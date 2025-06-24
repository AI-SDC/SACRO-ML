"""Test attribute inference attacks."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier

from sacroml.attacks import attribute_attack
from sacroml.attacks.attribute_attack import (
    _get_bounds_risk,
    _infer_categorical,
    _unique_max,
)


def pytest_generate_tests(metafunc):
    """Generate target model for testing."""
    if "get_target" in metafunc.fixturenames:
        metafunc.parametrize(
            "get_target", [RandomForestClassifier(bootstrap=False)], indirect=True
        )


@pytest.fixture(name="common_setup")
def fixture_common_setup(get_target):
    """Get ready to test some code."""
    target = get_target
    target.model.fit(target.X_train, target.y_train)
    attack_obj = attribute_attack.AttributeAttack(n_cpu=7)
    return target, attack_obj


def test_attack_undefined_feats(common_setup):
    """Test attack when features have not been defined."""
    target, attack_obj = common_setup
    target.features = {}
    output = attack_obj.attack(target)
    assert output == {}


def test_unique_max():
    """Test the _unique_max helper function."""
    has_unique = (0.3, 0.5, 0.2)
    no_unique = (0.5, 0.5)
    assert _unique_max(has_unique, 0.0)
    assert not _unique_max(has_unique, 0.6)
    assert not _unique_max(no_unique, 0.0)


def test_categorical_via_modified_attack_brute_force(common_setup):
    """Test categoricals using code from brute_force."""
    target, _ = common_setup

    threshold = 0
    feature = 0
    # make predictions
    t_low = _infer_categorical(target, feature, threshold)
    t_low_correct = t_low["train"][0]
    t_low_total = t_low["train"][1]
    t_low_train_samples = t_low["train"][4]

    # Check the number of samples in the dataset
    assert len(target.X_train) == t_low_train_samples
    # Check that all samples are correct for this threshold
    assert t_low_correct == t_low_total

    # or don't because threshold is too high
    threshold = 999
    t_high = _infer_categorical(target, feature, threshold)
    t_high_correct = t_high["train"][0]
    t_high_train_samples = t_high["train"][4]
    assert len(target.X_train) == t_high_train_samples
    assert t_high_correct == 0


def test_continuous_via_modified_bounds_risk(common_setup):
    """Test continuous variables get_bounds_risk()."""
    target, _ = common_setup
    returned = _get_bounds_risk(target.model, "dummy", 8, target.X_train, target.X_test)
    # Check the number of parameters returned
    assert len(returned.keys()) == 3
    # Check the value of the returned parameters
    assert returned["train"] == 0
    assert returned["test"] == 0


def test_aia_on_nursery(common_setup):
    """Test attribute inference attack."""
    target, attack_obj = common_setup
    output = attack_obj.attack(target)
    keys = output["attack_experiment_logger"]["attack_instance_logger"][
        "instance_0"
    ].keys()
    assert "categorical" in keys


def test_aia_multiclass(get_target_multiclass):
    """Test AttributeAttack with multiclass data."""
    target = get_target_multiclass
    attack_obj = attribute_attack.AttributeAttack(n_cpu=7)
    output = attack_obj.attack(target)
    assert output
