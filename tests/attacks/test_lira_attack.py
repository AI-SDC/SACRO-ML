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


def test_lira_individual_npz_and_json(lira_classifier_setup):
    """Test that individual results are saved to .npz and excluded from JSON."""
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

    instance = output["attack_experiment_logger"]["attack_instance_logger"][
        "instance_0"
    ]
    npz_filename = instance["individual_file"]
    assert npz_filename.startswith("lira_individual_")
    assert npz_filename.endswith("_instance_0.npz")

    npz_path = os.path.join(output_dir, npz_filename)
    assert os.path.exists(npz_path)

    data = np.load(npz_path)
    assert "y_pred_proba" in data
    assert "y_test" in data
    assert "score" in data
    assert "member" in data

    # Check JSON file does not contain fpr/tpr/roc_thresh/individual
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, encoding="utf-8") as fp:
        json_data = json.load(fp)

    lira_key = [k for k in json_data if k.startswith("LiRA")][0]
    json_instance = json_data[lira_key]["attack_experiment_logger"][
        "attack_instance_logger"
    ]["instance_0"]
    assert "fpr" not in json_instance
    assert "tpr" not in json_instance
    assert "roc_thresh" not in json_instance
    assert "individual" not in json_instance
    assert json_instance["individual_file"] == npz_filename


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

    fname_a = out_a["attack_experiment_logger"]["attack_instance_logger"]["instance_0"][
        "individual_file"
    ]
    fname_b = out_b["attack_experiment_logger"]["attack_instance_logger"]["instance_0"][
        "individual_file"
    ]

    assert fname_a != fname_b
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
