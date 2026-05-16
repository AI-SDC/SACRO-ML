"""Test LiRA attack."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.target import Target


@pytest.fixture(name="dummy_classifier_setup")
def fixture_dummy_classifier_setup():
    """Set up common things for DummyClassifier."""
    dummy = LIRAAttack._DummyClassifier()
    X = np.array([[0.2, 0.8], [0.7, 0.3]])
    return dummy, X


def test_predict_proba(dummy_classifier_setup):
    """Test the predict_proba method."""
    dummy, X = dummy_classifier_setup
    pred_probs = dummy.predict_proba(X)
    assert np.array_equal(pred_probs, X)


def test_predict(dummy_classifier_setup):
    """Test the predict method."""
    dummy, X = dummy_classifier_setup
    expected_output = np.array([1, 0])
    pred = dummy.predict(X)
    assert np.array_equal(pred, expected_output)


@pytest.fixture(name="lira_classifier_setup")
def fixture_lira_classifier_setup():
    """Set up common things for LiRA."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    target_model = RandomForestClassifier(
        n_estimators=100, min_samples_split=2, min_samples_leaf=1
    )
    target_model.fit(X_train, y_train)
    target = Target(
        model=target_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    target.save(path="test_lira_target")
    return target


def _json_instance_for(output_dir: str, output: dict, instance: str = "instance_0"):
    """Return one instance from the just-written report.json entry.

    ``report.json`` is append-only (GenerateJSONModule keys each entry
    ``"<attack_name>_<log_id>"``), so the entry is selected by the attack's
    own log_id rather than by position, keeping the test robust to a reused
    output directory.
    """
    with open(os.path.join(output_dir, "report.json"), encoding="utf-8") as fp:
        json_data = json.load(fp)
    key = f"{output['metadata']['attack_name']}_{output['log_id']}"
    return json_data[key]["attack_experiment_logger"]["attack_instance_logger"][
        instance
    ]


@pytest.mark.parametrize(
    ("mode", "expect_error"),
    [
        ("offline", False),
        ("offline-carlini", False),
        ("online-carlini", False),
        ("blah", True),
    ],
)
@pytest.mark.parametrize("fix_variance", [True, False])
def test_lira_attack(lira_classifier_setup, mode, expect_error, fix_variance):
    """Test LiRA attack with different modes."""
    # create target
    target = lira_classifier_setup
    # create attack
    lira = LIRAAttack(
        output_dir="test_output_lira",
        write_report=True,
        n_shadow_models=20,
        p_thresh=0.05,
        mode=mode,
        fix_variance=fix_variance,
        report_individual=True,
    )

    # check unsupported modes raise an error
    if expect_error:
        with pytest.raises(ValueError, match="Unsupported LiRA mode.*"):
            output = lira.attack(target)
        return

    # run LiRA
    output = lira.attack(target)

    # check metadata
    metadata = output["metadata"]
    assert metadata["attack_name"] == "LiRA Attack"
    assert metadata["attack_params"]["n_shadow_models"] == 20
    assert metadata["attack_params"]["p_thresh"] == 0.05
    assert metadata["attack_params"]["mode"] == mode
    assert metadata["attack_params"]["fix_variance"] == fix_variance
    assert metadata["attack_params"]["report_individual"]

    # check global metrics
    global_metrics = metadata["global_metrics"]
    sig = "Significant at p=0.05"
    not_sig = "Not significant at p=0.05"

    if mode == "offline-carlini" and fix_variance:
        assert global_metrics["PDIF_sig"] == not_sig
    else:
        assert global_metrics["PDIF_sig"] == sig
    assert global_metrics["AUC_sig"] == sig

    # check instance metrics
    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert 0 <= metrics["TPR"] <= 1
    assert 0 <= metrics["FPR"] <= 1


def test_lira_arrays_externalised_and_json(lira_classifier_setup):
    """ROC + individual arrays are externalised to .npz and stripped from JSON.

    The already-computed arrays are written to a single sidecar by the
    generic report writer; no per-attack caching is involved.
    """
    target = lira_classifier_setup
    output_dir = "test_output_lira"
    lira = LIRAAttack(
        output_dir=output_dir,
        write_report=True,
        n_shadow_models=20,
        p_thresh=0.05,
        mode="offline",
        fix_variance=False,
        report_individual=True,
    )
    output = lira.attack(target)

    # In-memory output is untouched so PDF generation still has the arrays.
    instance = output["attack_experiment_logger"]["attack_instance_logger"][
        "instance_0"
    ]
    assert "fpr" in instance
    assert "individual" in instance

    # JSON drops the big keys and carries an arrays_file pointer instead.
    json_instance = _json_instance_for(output_dir, output)
    assert "fpr" not in json_instance
    assert "tpr" not in json_instance
    assert "roc_thresh" not in json_instance
    assert "individual" not in json_instance

    arrays_filename = json_instance["arrays_file"]
    assert arrays_filename.endswith("_instance_0.npz")
    arrays_path = os.path.join(output_dir, arrays_filename)
    assert os.path.exists(arrays_path)
    with np.load(arrays_path) as arrays:
        # ROC arrays and the per-record individual block share one sidecar.
        assert "fpr" in arrays
        assert "tpr" in arrays
        assert "roc_thresh" in arrays
        assert "individual.score" in arrays
        assert "individual.member" in arrays


def test_lira_roc_externalised_without_report_individual(lira_classifier_setup):
    """ROC arrays are still externalised when report_individual=False.

    Guards against losing the ROC curve when individual records aren't
    reported: fpr/tpr/roc_thresh must still be persisted to the sidecar so
    the curve stays recoverable downstream.
    """
    target = lira_classifier_setup
    output_dir = "test_output_lira_predictions_only"
    lira = LIRAAttack(
        output_dir=output_dir,
        write_report=True,
        n_shadow_models=20,
        mode="offline",
        report_individual=False,
    )
    output = lira.attack(target)

    json_instance = _json_instance_for(output_dir, output)
    assert "fpr" not in json_instance
    assert "individual" not in json_instance

    arrays_filename = json_instance["arrays_file"]
    with np.load(os.path.join(output_dir, arrays_filename)) as arrays:
        assert "fpr" in arrays
        assert "tpr" in arrays
        # No individual block was requested, so none is stored.
        assert not any(k.startswith("individual.") for k in arrays)


def test_lira_two_runs_same_dir_no_clobber(lira_classifier_setup):
    """Two LiRA runs into the same output_dir keep distinct .npz files."""
    target = lira_classifier_setup
    output_dir = "test_output_lira_two_runs"

    lira_a = LIRAAttack(
        output_dir=output_dir,
        write_report=True,
        n_shadow_models=20,
        mode="offline",
        report_individual=True,
    )
    lira_b = LIRAAttack(
        output_dir=output_dir,
        write_report=True,
        n_shadow_models=20,
        mode="offline-carlini",
        report_individual=True,
    )

    out_a = lira_a.attack(target)
    out_b = lira_b.attack(target)

    # Filenames embed each attack's stable log_id, so the runs don't clobber.
    fname_a = _json_instance_for(output_dir, out_a)["arrays_file"]
    fname_b = _json_instance_for(output_dir, out_b)["arrays_file"]

    assert fname_a != fname_b
    assert out_a["log_id"][:8] in fname_a
    assert out_b["log_id"][:8] in fname_b
    assert os.path.exists(os.path.join(output_dir, fname_a))
    assert os.path.exists(os.path.join(output_dir, fname_b))


def test_lira_multiclass(get_target_multiclass):
    """Test LIRAAttack with multiclass data."""
    target = get_target_multiclass
    lira = LIRAAttack(
        output_dir="test_output_lira",
        write_report=True,
        n_shadow_models=20,
        p_thresh=0.05,
        mode="offline",
        fix_variance=False,
        report_individual=False,
    )
    output = lira.attack(target)
    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert 0 <= metrics["TPR"] <= 1
    assert 0 <= metrics["FPR"] <= 1
