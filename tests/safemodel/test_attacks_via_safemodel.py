"""Test attacks called via safemodel classes."""

from __future__ import annotations

import numpy as np
import pytest

from aisdc.attacks import attribute_attack, likelihood_attack, worst_case_attack
from aisdc.safemodel.classifiers import SafeDecisionTreeClassifier

RES_DIR = "RES"


@pytest.mark.parametrize(
    "get_target",
    [SafeDecisionTreeClassifier(random_state=1, max_depth=10, min_samples_leaf=1)],
    indirect=True,
)
def test_attacks_via_request_release(get_target):
    """Test vulnerable, hacked model then call request_release."""
    target = get_target
    assert target.__str__() == "nursery"  # pylint: disable=unnecessary-dunder-call
    target.model.fit(target.X_train, target.y_train)
    target.model.min_samples_leaf = 10
    target.model.request_release(path=RES_DIR, ext="pkl", target=target)


@pytest.mark.parametrize(
    "get_target",
    [SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=10)],
    indirect=True,
)
def test_run_attack_lira(get_target):
    """Test the lira attack via safemodel."""
    target = get_target
    target.model.fit(target.X_train, target.y_train)
    _, disclosive = target.model.preliminary_check()
    assert not disclosive
    print(np.unique(target.y_test, return_counts=True))
    print(np.unique(target.model.predict(target.X_test), return_counts=True))
    metadata = target.model.run_attack(target, "lira", RES_DIR)
    assert len(metadata) > 0  # something has been added


@pytest.mark.parametrize(
    "get_target",
    [SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=20)],
    indirect=True,
)
def test_run_attack_worstcase(get_target):
    """Test the worst case attack via safemodel."""
    target = get_target
    target.model.fit(target.X_train, target.y_train)
    _, disclosive = target.model.preliminary_check()
    assert not disclosive
    metadata = target.model.run_attack(target, "worstcase", RES_DIR)
    assert len(metadata) > 0  # something has been added


@pytest.mark.parametrize(
    "get_target",
    [SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=10)],
    indirect=True,
)
def test_run_attack_attribute(get_target):
    """Test the attribute attack via safemodel."""
    target = get_target
    target.model.fit(target.X_train, target.y_train)
    _, disclosive = target.model.preliminary_check()
    assert not disclosive
    metadata = target.model.run_attack(target, "attribute", RES_DIR)
    assert len(metadata) > 0  # something has been added


def test_attack_args():
    """Test the attack arguments class."""
    attack_obj = attribute_attack.AttributeAttack(output_dir="output_attribute")
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"

    attack_obj = likelihood_attack.LIRAAttack(output_dir="output_lira")
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"

    attack_obj = worst_case_attack.WorstCaseAttack(output_dir="output_worstcase")
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"


@pytest.mark.parametrize(
    "get_target",
    [SafeDecisionTreeClassifier(random_state=1, max_depth=5)],
    indirect=True,
)
def test_run_attack_unknown(get_target):
    """Test an unknown attack via safemodel."""
    target = get_target
    target.model.fit(target.X_train, target.y_train)
    metadata = target.model.run_attack(target, "unknown", RES_DIR)
    assert metadata["outcome"] == "unrecognised attack type requested"
