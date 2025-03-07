"""Test structural attacks."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

import sacroml.attacks.structural_attack as sa
from sacroml.attacks.target import Target


def get_target(modeltype: str, **kwparams: dict) -> Target:
    """Load dataset and create target of the desired type."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # these types should be handled
    if modeltype == "dt":
        target_model = DecisionTreeClassifier(**kwparams)
    elif modeltype == "rf":
        target_model = RandomForestClassifier(**kwparams)
    elif modeltype == "xgb":
        target_model = XGBClassifier(**kwparams)
    elif modeltype == "adaboost":
        target_model = AdaBoostClassifier(**kwparams)
    elif modeltype == "mlpclassifier":
        target_model = MLPClassifier(**kwparams)
    # should get polite error but not DoF yet
    elif modeltype == "svc":
        target_model = SVC(**kwparams)
    else:
        raise NotImplementedError("model type passed to get_model unknown")

    # Train the classifier
    target_model.fit(X_train, y_train)

    #  Wrap the model and data in a Target object
    target = Target(model=target_model)
    target.add_processed_data(X_train, y_train, X_test, y_test)

    return target


def test_unnecessary_risk():
    """Check the unnecessary rules."""
    # non-tree we have no evidence yet
    model = SVC()
    assert not sa.get_unnecessary_risk(model), "no risk without evidence"
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
        errstr = f" unnecessary risk with rule {idx}params are {model.get_params()}"
        assert sa.get_unnecessary_risk(model), errstr
    model = DecisionTreeClassifier(max_depth=1, min_samples_leaf=150)
    assert not sa.get_unnecessary_risk(model), (
        f"should be non-disclosive with {model.get_params}"
    )

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
        errstr = f" unnecessary risk with rule {idx}params are {model.get_params()}"
        assert sa.get_unnecessary_risk(model), errstr
    model = RandomForestClassifier(max_depth=1, n_estimators=25, min_samples_leaf=150)
    assert not sa.get_unnecessary_risk(model), (
        f"should be non-disclosive with {model.get_params}"
    )

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
        errstr = f" unnecessary risk with rule {idx}params are {model.get_params()}"
        assert sa.get_unnecessary_risk(model), errstr
    model = XGBClassifier(min_child_weight=10)
    assert not sa.get_unnecessary_risk(model), (
        f"should be non-disclosive with {model.get_params}"
    )


def test_non_trees():
    """Test behaviour if model type not tree-based."""
    param_dict = {"probability": True}
    target = get_target("svc", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    # remove model
    target.model = None
    myattack2 = sa.StructuralAttack()
    assert not myattack2.attack(target)


def test_dt():
    """Test for decision tree classifier."""
    # 'non' disclosive'
    param_dict = {"max_depth": 1, "min_samples_leaf": 150}
    target = get_target("dt", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert not myattack.dof_risk, "should be no DoF risk with decision stump"
    assert not myattack.k_anonymity_risk, (
        "should be no k-anonymity risk with min_samples_leaf 150"
    )
    assert not myattack.class_disclosure_risk, (
        "no class disclosure risk for stump with min samples leaf 150"
    )
    assert not myattack.unnecessary_risk, "not unnecessary risk if max_depth < 3.5"

    # highly disclosive
    param_dict2 = {"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2}
    target = get_target("dt", **param_dict2)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert not myattack.dof_risk, "should be no DoF risk with decision stump"
    assert myattack.k_anonymity_risk, (
        "should be  k-anonymity risk with unlimited depth and min_samples_leaf 5"
    )
    assert myattack.class_disclosure_risk, (
        "should be class disclosure risk with unlimited depth and min_samples_leaf 5"
    )
    assert myattack.unnecessary_risk, (
        " unnecessary risk with unlimited depth and min_samples_leaf 5"
    )


def test_adaboost():
    """Test for adaboost classifier."""
    # 'non' disclosive'
    # - base estimator =None => DecisionTreeClassifier with max_depth 1
    # also set THRESHOLD to 4
    np.random.seed(42)

    param_dict = {"n_estimators": 2, "estimator": None}
    target = get_target("adaboost", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.THRESHOLD = 2
    myattack.attack(target)
    assert not myattack.dof_risk, "should be no DoF risk with just 2 decision stumps"
    assert not myattack.k_anonymity_risk, (
        "should be no k-anonymity risk with only 2 stumps"
    )
    assert not myattack.class_disclosure_risk, "no class disclosure risk for 2 stumps"
    assert not myattack.unnecessary_risk, " unnecessary risk not defined for adaboost"

    # highly disclosive
    kwargs = {"max_depth": None, "min_samples_leaf": 2}
    param_dict2 = {
        "estimator": DecisionTreeClassifier(**kwargs),
        "n_estimators": 1000,
    }
    target = get_target("adaboost", **param_dict2)
    myattack2 = sa.StructuralAttack()
    myattack2.attack(target)
    assert myattack2.dof_risk, "should be  DoF risk with adaboost of deep trees"
    assert myattack2.k_anonymity_risk, (
        "should be k-anonymity risk with adaboost unlimited depth and min_samples_leaf 2"
    )
    assert myattack2.class_disclosure_risk, (
        "should be class risk with adaboost unlimited depth and min_samples_leaf 2"
    )
    assert not myattack2.unnecessary_risk, " unnecessary risk not define for adaboost"


def test_rf():
    """Test for random forest classifier."""
    # 'non' disclosive'
    param_dict = {"max_depth": 1, "min_samples_leaf": 150, "n_estimators": 10}
    target = get_target("rf", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert not myattack.dof_risk, (
        "should be no DoF risk with small forest of decision stumps"
    )
    assert not myattack.k_anonymity_risk, (
        "should be no k-anonymity risk with min_samples_leaf 150"
    )
    assert not myattack.class_disclosure_risk, (
        "no class disclosure risk for stumps with min samples leaf 150"
    )
    assert not myattack.unnecessary_risk, "not unnecessary risk if max_depth < 3.5"

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
    assert myattack.dof_risk, "should be  DoF risk with forest of deep trees"
    assert myattack.k_anonymity_risk, (
        "should be  k-anonymity risk with unlimited depth and min_samples_leaf 5"
    )
    assert myattack.class_disclosure_risk, (
        "should be class disclsoure risk with unlimited depth and min_samples_leaf 5"
    )
    assert myattack.unnecessary_risk, (
        " unnecessary risk with unlimited depth and min_samples_leaf 5"
    )


def test_xgb():
    """Test for xgboost."""
    # non-disclosive
    param_dict = {"max_depth": 1, "min_child_weight": 50, "n_estimators": 5}
    target = get_target("xgb", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    assert not myattack.dof_risk, (
        "should be no DoF risk with small xgb of decision stumps"
    )
    assert not myattack.k_anonymity_risk, (
        "should be no k-anonymity risk with min_samples_leaf 150"
    )
    assert not myattack.class_disclosure_risk, (
        "no class disclosure risk for stumps with min child weight 50"
    )
    assert myattack.unnecessary_risk == 0, "not unnecessary risk if max_depth < 3.5"

    # highly disclosive
    param_dict2 = {"max_depth": 50, "n_estimators": 100, "min_child_weight": 1}
    target2 = get_target("xgb", **param_dict2)
    myattack2 = sa.StructuralAttack()
    myattack2.attack(target2)
    assert myattack2.dof_risk, "should be  DoF risk with xgb of deep trees"
    assert myattack2.k_anonymity_risk, (
        "should be  k-anonymity risk with depth 50 and min_child_weight 1"
    )
    assert myattack2.class_disclosure_risk, (
        "should be class disclosure risk with xgb lots of deep trees"
    )
    assert myattack2.unnecessary_risk, " unnecessary risk with these xgb params"


def test_sklearnmlp():
    """Test for sklearn MLPClassifier."""
    # non-disclosive
    safeparams = {
        "hidden_layer_sizes": (10,),
        "random_state": 12345,
        "activation": "identity",
        "max_iter": 1,
    }
    target = get_target("mlpclassifier", **safeparams)
    myattack = sa.StructuralAttack()
    myattack.attack(target)
    paramstr = ""
    for key, val in safeparams.items():
        paramstr += f"{key}:{val}\n"
    assert not myattack.dof_risk, (
        f"should be no DoF risk with small mlp with params {paramstr}"
    )
    assert not myattack.k_anonymity_risk, (
        f"should be no k-anonymity risk with params {paramstr}"
    )
    assert myattack.class_disclosure_risk, (
        f"should be  class disclosure risk with params {paramstr}"
    )
    assert not myattack.unnecessary_risk, "not unnecessary risk for mlps at present"

    # highly disclosive
    unsafeparams = {
        "hidden_layer_sizes": (50, 50),
        "random_state": 12345,
        "activation": "relu",
        "max_iter": 100,
    }
    uparamstr = ""
    for key, val in unsafeparams.items():
        uparamstr += f"{key}:{val}\n"
    target2 = get_target("mlpclassifier", **unsafeparams)
    myattack2 = sa.StructuralAttack()
    myattack2.attack(target2)
    assert myattack2.dof_risk, f"should be DoF risk with this MLP:\n{uparamstr}"
    assert myattack2.k_anonymity_risk, (
        "559/560 records should be k-anonymity risk with this MLP:\n{uparamstr}"
    )
    assert myattack2.class_disclosure_risk, (
        "should be class disclosure risk with this MLP:\n{uparamstr}"
    )
    assert not myattack2.unnecessary_risk, "no unnecessary risk yet for MLPClassifiers"


def test_reporting():
    """Test reporting functionality."""
    param_dict = {"max_depth": 1, "min_samples_leaf": 150}
    target = get_target("dt", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)


def test_structural_multiclass(get_target_multiclass):
    """Test StructuralAttack with multiclass data."""
    target = get_target_multiclass
    attack_obj = sa.StructuralAttack()
    output = attack_obj.attack(target)
    assert output
