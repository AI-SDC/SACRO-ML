"""Test LiRA attack."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks.likelihood_attack import DummyClassifier, LIRAAttack
from aisdc.attacks.target import Target


@pytest.fixture(name="dummy_classifier_setup")
def fixture_dummy_classifier_setup():
    """Set up common things for DummyClassifier."""
    dummy = DummyClassifier()
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


def test_lira_attack(lira_classifier_setup):
    """Test LiRA attack."""
    target = lira_classifier_setup
    lira = LIRAAttack(n_shadow_models=20, output_dir="test_output_lira")
    lira.attack(target)
