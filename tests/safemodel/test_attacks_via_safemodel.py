"""
Tests attacks called via safemodel classes
uses a subsampled nursery dataset as this tests more of the attack code
currently using random forests.
"""

from __future__ import annotations

import shutil

import numpy as np

from aisdc.attacks import attribute_attack, likelihood_attack, worst_case_attack
from aisdc.safemodel.classifiers import SafeDecisionTreeClassifier

from ..common import clean, get_target

# pylint: disable=duplicate-code,unnecessary-dunder-call

RES_DIR = "RES"


def test_attacks_via_request_release():
    """Make vulnerable,hacked model then call request_release."""
    # build a broken model and hack it so lots of reasons to fail and be vulnerable
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=10, min_samples_leaf=1)
    target = get_target(model)
    assert target.__str__() == "nursery"
    model.fit(target.x_train, target.y_train)
    model.min_samples_leaf = 10
    model.request_release(path=RES_DIR, ext="pkl", target=target)
    clean(RES_DIR)


def test_run_attack_lira():
    """Calls the lira attack via safemodel."""
    # build a model
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=10)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive

    print(np.unique(target.y_test, return_counts=True))
    print(np.unique(model.predict(target.x_test), return_counts=True))
    metadata = model.run_attack(target, "lira", RES_DIR, "lira_res")
    clean(RES_DIR)
    assert len(metadata) > 0  # something has been added


def test_run_attack_worstcase():
    """Calls the worst case attack via safemodel."""
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=20)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive
    metadata = model.run_attack(target, "worst_case", RES_DIR, "wc_res")
    clean(RES_DIR)
    assert len(metadata) > 0  # something has been added


def test_run_attack_attribute():
    """Calls the attribute  attack via safemodel."""
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=10)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive
    metadata = model.run_attack(target, "attribute", RES_DIR, "attr_res")
    clean(RES_DIR)
    assert len(metadata) > 0  # something has been added


def test_attack_args():
    """Tests the attack arguments class."""
    fname = "aia_example"
    attack_obj = attribute_attack.AttributeAttack(
        output_dir="output_attribute", report_name=fname
    )
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"
    assert fname == attack_obj.report_name

    fname = "liraa"
    attack_obj = likelihood_attack.LIRAAttack(
        output_dir="output_lira", report_name=fname
    )
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"
    assert fname == attack_obj.report_name

    fname = "wca"
    attack_obj = worst_case_attack.WorstCaseAttack(
        output_dir="output_worstcase", report_name=fname
    )
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"
    assert fname == attack_obj.report_name
    shutil.rmtree("output_attribute")
    shutil.rmtree("output_lira")
    shutil.rmtree("output_worstcase")


def test_run_attack_unknown():
    """Calls an unknown attack via safemodel."""
    # build a model
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    metadata = model.run_attack(target, "unknown", RES_DIR, "unk")
    clean(RES_DIR)
    assert metadata["outcome"] == "unrecognised attack type requested"
