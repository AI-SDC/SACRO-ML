"""Test_worst_case_attack.py
Copyright (C) Jim Smith 2023 <james.smith@uwe.ac.uk>.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

import aisdc.attacks.structural_attack as sa
from aisdc.attacks.target import Target

from ..common import clean


def get_target(modeltype: str, **kwparams: dict) -> Target:
    """Loads dataset and creates target of the desired type."""

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # these types should be handled
    if modeltype == "dt":
        target_model = DecisionTreeClassifier(**kwparams)
    elif modeltype == "rf":
        target_model = RandomForestClassifier(**kwparams)
    elif modeltype == "xgb":
        target_model = XGBClassifier(**kwparams)
    elif modeltype == "adaboost":
        target_model = AdaBoostClassifier(**kwparams)
    # should get polite error but not DoF yet
    elif modeltype == "svc":
        target_model = SVC(**kwparams)
    else:
        raise NotImplementedError("model type passed to get_model unknown")

    # Train the classifier
    target_model.fit(train_X, train_y)

    #  Wrap the model and data in a Target object
    target = Target(model=target_model)
    target.add_processed_data(train_X, train_y, test_X, test_y)

    return target


def test_unnecessary_risk():
    """Checking the unnecessary rules."""
    # non-tree we have no evidence yet
    model = SVC()
    assert sa.get_unnecessary_risk(model) == 0, "no risk without evidence"
    # decision tree next
    risky_param_dicts = [
        {
            "max_features": "log2",
            "max_depth": 8,
            "min_samples_leaf": 7,
            "min_samples_split": 14,
        },
        {
            "splitter": "best",
            "max_depth": 8,
            "min_samples_leaf": 7,
            "min_samples_split": 16,
        },
        {
            "splitter": "best",
            "max_depth": 8,
            "min_samples_leaf": 10,
            "max_features": None,
        },
        {
            "splitter": "best",
            "max_depth": 4,
            "max_features": None,
            "min_samples_leaf": 7,
        },
        {
            "splitter": "random",
            "max_depth": 8,
            "min_samples_leaf": 7,
            "max_features": None,
            "min_samples_split": 25,
        },
    ]
    for idx, paramdict in enumerate(risky_param_dicts):
        model = DecisionTreeClassifier(**paramdict)
        errstr = f" unnecessary risk with rule {idx}" f"params are {model.get_params()}"
        assert sa.get_unnecessary_risk(model) == 1, errstr
    model = DecisionTreeClassifier(max_depth=1, min_samples_leaf=150)
    assert (
        sa.get_unnecessary_risk(model) == 0
    ), f"should be non-disclosive with {model.get_params}"

    # now random forest
    risky_param_dicts = [
        {"max_depth": 50, "n_estimators": 50, "max_features": None},
        {
            "max_depth": 50,
            "n_estimators": 50,
            "min_samples_split": 5,
            "max_features": None,
            "bootstrap": True,
        },
        {
            "max_depth": 25,
            "n_estimators": 25,
            "min_samples_leaf": 5,
            "bootstrap": False,
        },
    ]
    for idx, paramdict in enumerate(risky_param_dicts):
        model = RandomForestClassifier(**paramdict)
        errstr = f" unnecessary risk with rule {idx}" f"params are {model.get_params()}"
        assert sa.get_unnecessary_risk(model) == 1, errstr
    model = RandomForestClassifier(max_depth=1, n_estimators=25, min_samples_leaf=150)
    assert (
        sa.get_unnecessary_risk(model) == 0
    ), f"should be non-disclosive with {model.get_params}"

    # finally xgboost
    risky_param_dicts = [
        {
            "max_depth": 5,  # > 3.5
            "n_estimators": 10,  #  and 3.5 < n_estimators <= 12.5
            "min_child_weight": 1,  #   and model.min_child_weight <= 1.5
        },
        {
            "max_depth": 5,  # > 3.5
            "n_estimators": 25,  # > 12.5
            "min_child_weight": 1,  # <= 3
        },
        {
            "max_depth": 5,  # > 3.5
            "n_estimators": 100,  # > 62.5
            "min_child_weight": 5,  #    and 3 < model.min_child_weight <= 6
        },
    ]
    for idx, paramdict in enumerate(risky_param_dicts):
        model = XGBClassifier(**paramdict)
        errstr = f" unnecessary risk with rule {idx}" f"params are {model.get_params()}"
        assert sa.get_unnecessary_risk(model) == 1, errstr
    model = XGBClassifier(min_child_weight=10)
    assert (
        sa.get_unnecessary_risk(model) == 0
    ), f"should be non-disclosive with {model.get_params}"


def test_non_trees():
    """Test  behaviour if model type not tree-based."""
    param_dict = {"probability": True}
    target = get_target("svc", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    # remove model
    target.model = None
    with pytest.raises(NotImplementedError):
        myattack2 = sa.StructuralAttack()
        myattack2.attack(target)


def test_dt():
    """Test for decision tree classifier."""

    # 'non' disclosive'
    param_dict = {"max_depth": 1, "min_samples_leaf": 150}
    target = get_target("dt", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert myattack.DoF_risk == 0, "should be no DoF risk with decision stump"
    assert (
        myattack.k_anonymity_risk == 0
    ), "should be no k-anonymity risk with min_samples_leaf 150"
    assert (
        myattack.class_disclosure_risk == 0
    ), "no class disclosure risk for stump with min samples leaf 150"
    assert myattack.unnecessary_risk == 0, "not unnecessary risk if max_depth < 3.5"

    # highly disclosive
    param_dict2 = {"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2}
    target = get_target("dt", **param_dict2)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert myattack.DoF_risk == 0, "should be no DoF risk with decision stump"
    assert (
        myattack.k_anonymity_risk == 1
    ), "should be  k-anonymity risk with unlimited depth and min_samples_leaf 5"
    assert (
        myattack.class_disclosure_risk == 1
    ), "should be class disclosure risk with unlimited depth and min_samples_leaf 5"
    assert (
        myattack.unnecessary_risk == 1
    ), " unnecessary risk with unlimited depth and min_samples_leaf 5"


def test_adaboost():
    """Test for adaboost classifier."""

    # 'non' disclosive'
    # - base estimator =None => DecisionTreeClassifier with max_depth 1
    # also set THRESHOLD to 4
    param_dict = {"n_estimators": 2, "estimator": None}
    target = get_target("adaboost", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.THRESHOLD = 2
    myattack.attack(target)
    assert myattack.DoF_risk == 0, "should be no DoF risk with just 2 decision stumps"
    assert (
        myattack.k_anonymity_risk == 0
    ), "should be no k-anonymity risk with only 2 stumps"
    assert myattack.class_disclosure_risk == 0, "no class disclosure risk for 2 stumps"
    assert myattack.unnecessary_risk == 0, " unnecessary risk not defined for adaboost"

    # highly disclosive
    kwargs = {"max_depth": None, "min_samples_leaf": 2}
    param_dict2 = {
        "estimator": DecisionTreeClassifier(**kwargs),
        "n_estimators": 1000,
    }
    target = get_target("adaboost", **param_dict2)
    myattack2 = sa.StructuralAttack()
    myattack2.attack(target)
    assert myattack2.DoF_risk == 1, "should be  DoF risk with adaboost of deep trees"
    assert (
        myattack2.k_anonymity_risk == 1
    ), "should be  k-anonymity risk with adaboost unlimited depth and min_samples_leaf 2"
    assert (
        myattack2.class_disclosure_risk == 1
    ), "should be class disclosure risk with adaboost unlimited depth and min_samples_leaf 2"
    assert myattack2.unnecessary_risk == 0, " unnecessary risk not define for adaboost"


def test_rf():
    """Test for random forest classifier."""

    # 'non' disclosive'
    param_dict = {"max_depth": 1, "min_samples_leaf": 150, "n_estimators": 10}
    target = get_target("rf", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert (
        myattack.DoF_risk == 0
    ), "should be no DoF risk with small forest of decision stumps"
    assert (
        myattack.k_anonymity_risk == 0
    ), "should be no k-anonymity risk with min_samples_leaf 150"
    assert (
        myattack.class_disclosure_risk == 0
    ), "no class disclosure risk for stumps with min samples leaf 150"
    assert myattack.unnecessary_risk == 0, "not unnecessary risk if max_depth < 3.5"

    # highly disclosive
    param_dict2 = {
        "max_depth": None,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "n_estimators": 1000,
    }
    target = get_target("rf", **param_dict2)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert myattack.DoF_risk == 1, "should be  DoF risk with forest of deep trees"
    assert (
        myattack.k_anonymity_risk == 1
    ), "should be  k-anonymity risk with unlimited depth and min_samples_leaf 5"
    assert (
        myattack.class_disclosure_risk == 1
    ), "should be class disclsoure risk with unlimited depth and min_samples_leaf 5"
    assert (
        myattack.unnecessary_risk == 1
    ), " unnecessary risk with unlimited depth and min_samples_leaf 5"


def test_xgb():
    """Test for xgboost."""
    # non-disclosive
    param_dict = {"max_depth": 1, "min_child_weight": 50, "n_estimators": 5}
    target = get_target("xgb", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert (
        myattack.DoF_risk == 0
    ), "should be no DoF risk with small xgb of decision stumps"
    assert (
        myattack.k_anonymity_risk == 0
    ), "should be no k-anonymity risk with min_samples_leaf 150"
    assert (
        myattack.class_disclosure_risk == 0
    ), "no class disclosure risk for stumps with min child weight 50"
    assert myattack.unnecessary_risk == 0, "not unnecessary risk if max_depth < 3.5"

    # highly disclosive
    param_dict2 = {"max_depth": 50, "n_estimators": 100, "min_child_weight": 1}
    target2 = get_target("xgb", **param_dict2)
    myattack2 = sa.StructuralAttack()
    myattack2.attack(target2)
    assert myattack2.DoF_risk == 1, "should be  DoF risk with xgb of deep trees"
    assert (
        myattack2.k_anonymity_risk == 1
    ), "should be  k-anonymity risk with depth 50 and min_child_weight 1"
    assert (
        myattack2.class_disclosure_risk == 1
    ), "should be class disclosure risk with xgb lots of deep trees"
    assert myattack2.unnecessary_risk == 1, " unnecessary risk with these xgb params"


def test_reporting():
    """Test reporting functionality."""
    param_dict = {"max_depth": 1, "min_samples_leaf": 150}
    target = get_target("dt", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    myattack.make_report()


def test_main_example():
    """Test command line example."""
    param_dict = {"max_depth": 1, "min_samples_leaf": 150}
    target = get_target("dt", **param_dict)
    target_path = "dt.sav"
    target.save(target_path)
    testargs = [
        "prog",
        "run-attack",
        "--target-path",
        target_path,
        "--output-dir",
        "test_output_sa",
        "--report-name",
        "commandline_structural_report",
    ]
    with patch.object(sys, "argv", testargs):
        sa.main()
    config = {
        "output_dir": "test_output_structural2",
        "report_name": "structural_test",
    }
    with open("config_structural_test.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(config))
    testargs = [
        "prog",
        "run-attack-from-configfile",
        "--attack-config-json-file-name",
        "config_structural_test.json",
        "--attack-target-folder-path",
        "dt.sav",
    ]
    with patch.object(sys, "argv", testargs):
        sa.main()

    clean("dt.sav")
    clean("test_output_sa")
    clean("config_structural_test.json")
    clean("outputs_structural")
