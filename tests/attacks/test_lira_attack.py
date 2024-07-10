"""Test LiRA attack."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks.likelihood_attack import LIRAAttack
from aisdc.attacks.target import Target


@pytest.fixture(name="dummy_classifier_setup")
def fixture_dummy_classifier_setup():
    """Set up common things for DummyClassifier."""
    dummy = LIRAAttack._DummyClassifier()  # pylint: disable=protected-access
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
    target = Target(target_model)
    target.add_processed_data(X_train, y_train, X_test, y_test)
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
