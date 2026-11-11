"""Test threshold attack."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from sacroml.attacks.target import Target
from sacroml.attacks.threshold_attack import ThresholdAttack


def pytest_generate_tests(metafunc):
    """Generate target model for testing."""
    if "get_target" in metafunc.fixturenames:
        metafunc.parametrize("get_target", [RandomForestClassifier()], indirect=True)


@pytest.fixture(name="common_setup")
def fixture_common_setup(get_target):
    """Get ready to test some code."""
    target = get_target
    target.model.fit(target.X_train, target.y_train)
    return target


def test_insufficient_target():
    """Test insufficient target details to run."""
    target = Target()
    attack = ThresholdAttack()
    assert not attack.attack(target)


def test_threshold_attack(common_setup):
    """Test threshold attack runs."""
    target = common_setup
    attack = ThresholdAttack()

    result = attack.attack(target)
    assert result["metadata"]["attack_name"] == "Threshold attack"
    assert result["metadata"]["global_metrics"]["AUC"] == pytest.approx(0.67, abs=0.01)


def test_attack_from_signals_fails():
    """Test attack does not run from multi-dim signals."""
    rng = np.random.default_rng(1)
    train_signals = rng.random((50, 2))
    test_signals = rng.random((20, 2))

    target = Target(proba_train=train_signals, proba_test=test_signals)
    attack = ThresholdAttack()

    with pytest.raises(ValueError, match="signals require 1 float"):
        attack.attack(target)


def test_attack_from_signals_success():
    """Test attack runs from signals."""
    rng = np.random.default_rng(1)
    train_signals = rng.random((50,))
    test_signals = rng.random((20,))

    new_target = Target(proba_train=train_signals, proba_test=test_signals)
    attack = ThresholdAttack()

    result = attack.attack(new_target)
    assert result["metadata"]["attack_name"] == "Threshold attack"
    assert result["metadata"]["global_metrics"]["AUC"] == pytest.approx(0.45, abs=0.01)
