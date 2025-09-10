"""Test structural attacks."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

import sacroml.attacks.structural_attack as sa
from sacroml.attacks.target import Target

try:
    import torch

    from tests.attacks.pytorch_model import OverfitNet

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    OverfitNet = None
    TORCH_AVAILABLE = False


def get_target(modeltype: str, **kwparams: dict) -> Target:
    """Load dataset and create target of the desired type."""
    X, y = make_moons(
        n_samples=50,
        noise=0.5,
        random_state=12345,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    return Target(
        model=target_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


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


kwargs_dtsafe = {"max_depth": 1, "min_samples_leaf": 200}
kwargs_dtunsafe = {
    "max_depth": None,
    "splitter": "best",
    "min_samples_leaf": 1,
    "criterion": "entropy",
    #    "max_features":2
}


def test_dt_nondisclosive():
    """Test for safe decision tree classifier."""
    target_dtsafe = get_target("dt", **kwargs_dtsafe)
    myattack_dtsafe = sa.StructuralAttack()
    myattack_dtsafe.attack(target_dtsafe)
    assert not myattack_dtsafe.results.dof_risk, (
        "should be no DoF risk with decision stump"
    )
    assert not myattack_dtsafe.results.k_anonymity_risk, (
        "should be no k-anonymity risk with min_samples_leaf 150"
    )
    assert not myattack_dtsafe.results.class_disclosure_risk, (
        "no class disclosure risk for stump with min samples leaf 150"
    )
    assert not myattack_dtsafe.results.unnecessary_risk, (
        "not unnecessary risk if max_depth < 3.5"
    )


def test_dt_disclosive():
    """Test for risky decision tree classifier."""
    target_dtunsafe = get_target("dt", **kwargs_dtunsafe)
    myattack_dtunsafe = sa.StructuralAttack()
    myattack_dtunsafe.attack(target_dtunsafe)
    assert myattack_dtunsafe.results.dof_risk, (
        "should be  DoF risk with unlimited complexity decision tree"
    )
    assert myattack_dtunsafe.results.k_anonymity_risk, (
        "should be  k-anonymity risk with unlimited complexity decision tree"
    )
    assert myattack_dtunsafe.results.class_disclosure_risk, (
        "should be class disclosure risk with unlimited complexity decision tree"
    )
    assert myattack_dtunsafe.results.unnecessary_risk, (
        " unnecessary risk with unlimited unlimited complexity decision tree"
    )

    assert myattack_dtunsafe.results.smallgroup_risk, (
        "small group risk with unlimited complexity decision tree"
    )


def test_adaboost_nondisclosive():
    """Test for nondisclosive adaboost classifier."""
    param_dict_adasafe = {
        "n_estimators": 2,
        "estimator": DecisionTreeClassifier(**kwargs_dtsafe),
    }
    target = get_target("adaboost", **param_dict_adasafe)
    myattack_adasafe = sa.StructuralAttack()
    myattack_adasafe.THRESHOLD = 2
    myattack_adasafe.attack(target)
    assert not myattack_adasafe.results.dof_risk, (
        "should be no DoF risk with just 2 decision stumps"
    )
    assert not myattack_adasafe.results.k_anonymity_risk, (
        "should be no k-anonymity risk with only 2 stumps"
    )
    assert not myattack_adasafe.results.class_disclosure_risk, (
        "no class disclosure risk for 2 stumps"
    )
    assert not myattack_adasafe.results.unnecessary_risk, (
        " unnecessary risk not defined for adaboost"
    )
    assert not myattack_adasafe.results.smallgroup_risk, (
        "no small group risk with just 2 stumps"
    )


def test_adaboost_disclosive():
    """Test for disclosive adaboost classifier."""
    param_dict_adaunsafe = {
        "estimator": DecisionTreeClassifier(**kwargs_dtunsafe),
        "n_estimators": 1,
    }
    target_adaunsafe = get_target("adaboost", **param_dict_adaunsafe)
    myattack_adaunsafe = sa.StructuralAttack()
    myattack_adaunsafe.attack(target_adaunsafe)

    assert myattack_adaunsafe.results.dof_risk, (
        "should be  DoF risk with adaboost 1 unlimited trees\n"
    )

    assert (
        myattack_adaunsafe.results.k_anonymity_risk
        or myattack_adaunsafe.results.class_disclosure_risk
        or myattack_adaunsafe.results.smallgroup_risk
    ), "should be a risk with 1 risky decision tree"

    assert not myattack_adaunsafe.results.unnecessary_risk, (
        " unnecessary risk not define for adaboost"
    )


def test_rf_nondisclosive():
    """Test for safe random forest classifier."""
    param_dict_rfsafe = {"max_depth": 1, "min_samples_leaf": 150, "n_estimators": 2}
    target_rfsafe = get_target("rf", **param_dict_rfsafe)
    myattack_rfsafe = sa.StructuralAttack()
    myattack_rfsafe.attack(target_rfsafe)
    assert not myattack_rfsafe.results.dof_risk, (
        "should be no DoF risk with small forest of decision stumps"
    )
    assert not myattack_rfsafe.results.k_anonymity_risk, (
        "should be no k-anonymity risk with min_samples_leaf 150"
    )
    assert not myattack_rfsafe.results.class_disclosure_risk, (
        "no class disclosure risk for stumps with min samples leaf 150"
    )

    assert not myattack_rfsafe.results.smallgroup_risk, (
        "no small group risk with 2 stumps and min samples leaf 150"
    )
    assert not myattack_rfsafe.results.unnecessary_risk, (
        "not unnecessary risk if max_depth < 3.5"
    )


def test_rf_disclosive():
    """Test for unsafe random forest classifier."""
    param_dict_rfunsafe = {
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 1,
        "bootstrap": False,
    }
    target_rfunsafe = get_target("rf", **param_dict_rfunsafe)
    myattack_rfunsafe = sa.StructuralAttack()
    myattack_rfunsafe.attack(target_rfunsafe)

    assert (
        myattack_rfunsafe.results.dof_risk
        or myattack_rfunsafe.results.k_anonymity_risk
        or myattack_rfunsafe.results.class_disclosure_risk
        or myattack_rfunsafe.res.smallgroup_risk
    ), "should be a risk with a random forest made of one risky decision tree "


def test_rf_unnecessary():
    """Test for unsafe random forest classifier."""
    param_dict_rfunsafe = {
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 20,
        "bootstrap": False,
    }
    target_rfunsafe = get_target("rf", **param_dict_rfunsafe)
    myattack_rfunsafe = sa.StructuralAttack()
    myattack_rfunsafe.attack(target_rfunsafe)

    assert myattack_rfunsafe.results.unnecessary_risk, "should be unnecessary risk"


def test_xgb_nondisclosive():
    """Test for safe xgboost."""
    # non-disclosive
    param_dict_xgbsafe = {"max_depth": 1, "n_estimators": 1, "lambda": 1}
    target_xgbsafe = get_target("xgb", **param_dict_xgbsafe)
    myattack_xgbsafe = sa.StructuralAttack(report_individual=True)
    myattack_xgbsafe.attack(target_xgbsafe)
    assert not myattack_xgbsafe.results.dof_risk, (
        "should be no DoF risk with small xgb of decision stumps"
        f" results are:\n{myattack_xgbsafe.results}"
    )
    assert not myattack_xgbsafe.results.k_anonymity_risk, (
        "should be no k-anonymity risk with min_samples_leaf 150",
    )
    assert not myattack_xgbsafe.results.class_disclosure_risk, (
        "no class disclosure risk for stumps with min child weight 50"
    )

    assert myattack_xgbsafe.results.unnecessary_risk == 0, (
        "not unnecessary risk if max_depth < 3.5"
    )


def test_xgb_disclosive():
    """Test for unsafe xgboost."""
    # highly likely to be disclosive
    param_dict_xgbunsafe = {"max_depth": 50, "n_estimators": 1, "min_child_weight": 1}
    target_xgbunsafe = get_target("xgb", **param_dict_xgbunsafe)
    myattack_xgbunsafe = sa.StructuralAttack()
    myattack_xgbunsafe.attack(target_xgbunsafe)

    assert (
        myattack_xgbunsafe.results.dof_risk
        or myattack_xgbunsafe.results.k_anonymity_risk
        or myattack_xgbunsafe.results.class_disclosure_risk
        or myattack_xgbunsafe.results.smallgroup_risk
    ), "should be  risk with xgb one deep tree"


def test_xgb_unnecessary():
    """Test for unsafe xgboost."""
    # highly likely to be disclosive
    param_dict_xgbunsafe2 = {"max_depth": 10, "n_estimators": 10, "min_child_weight": 1}
    target_xgbunsafe2 = get_target("xgb", **param_dict_xgbunsafe2)
    myattack_xgbunsafe2 = sa.StructuralAttack()
    myattack_xgbunsafe2.attack(target_xgbunsafe2)

    assert myattack_xgbunsafe2.results.unnecessary_risk, (
        " unnecessary risk with these xgb params"
    )


def test_sklearnmlp_nondisclosive():
    """Test for safe sklearn MLPClassifier."""
    # non-disclosive
    params_mlpsafe = {
        "hidden_layer_sizes": (5,),
        "random_state": 12345,
        "activation": "relu",
        "max_iter": 100,
    }
    target_mlpsafe = get_target("mlpclassifier", **params_mlpsafe)
    myattack_mlpsafe = sa.StructuralAttack()
    myattack_mlpsafe.attack(target_mlpsafe)
    paramstr = ""
    for key, val in params_mlpsafe.items():
        paramstr += f"{key}:{val}\t"
    assert not myattack_mlpsafe.results.dof_risk, (
        f"should be no DoF risk with small mlp with params {paramstr}"
    )
    # shows k-anonymity risks because task is so noisy
    assert myattack_mlpsafe.results.k_anonymity_risk, (
        f"should be  k-anonymity risk with params {paramstr}"
    )
    assert not myattack_mlpsafe.results.class_disclosure_risk, (
        f"should be  class disclosure risk with params {paramstr}"
    )
    assert not myattack_mlpsafe.results.unnecessary_risk, (
        "not unnecessary risk for mlps at present"
    )


def test_sklearnmlp_disclosive():
    """Test for safe sklearn MLPClassifier."""
    # highly disclosive
    params_mlpunsafe = {
        "hidden_layer_sizes": (1000),
        "random_state": 12345,
        "activation": "relu",
        "max_iter": 100,
    }
    uparamstr = ""
    for key, val in params_mlpunsafe.items():
        uparamstr += f"{key}:{val}\n"
    target_mlpunsafe = get_target("mlpclassifier", **params_mlpunsafe)
    myattack_mlpunsafe = sa.StructuralAttack()
    myattack_mlpunsafe.attack(target_mlpunsafe)
    assert myattack_mlpunsafe.results.dof_risk, (
        f"should be DoF risk with this MLP:\n{uparamstr}"
    )
    assert (
        myattack_mlpunsafe.results.k_anonymity_risk
        or myattack_mlpunsafe.results.class_disclosure_risk
    ), "should be risks present with this massively overfit  MLP:\n{uparamstr}"
    assert not myattack_mlpunsafe.results.unnecessary_risk, (
        "no unnecessary risk yet for MLPClassifiers"
    )


def test_reporting():
    """Test reporting functionality."""
    param_dict = {"max_depth": 1, "min_samples_leaf": 200}
    target = get_target("dt", **param_dict)
    myattack = sa.StructuralAttack()
    myattack.attack(target)


def test_structural_multiclass(get_target_multiclass):
    """Test StructuralAttack with multiclass data."""
    target = get_target_multiclass
    attack_obj = sa.StructuralAttack()
    output = attack_obj.attack(target)
    assert output


def test_pytorch_parameter_counting():
    """Test PyTorch parameter counting functionality."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    # Create a PyTorch model using the existing test model
    model = OverfitNet(x_dim=10, y_dim=2, n_units=5)
    param_count = sa.get_model_param_count(model)

    # Expected parameters: (10*5 + 5) + (5*2 + 2) = 55 + 12 = 67
    expected_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count == expected_count, (
        f"Expected {expected_count}, got {param_count}"
    )


def test_xgb_empty_dataframe():
    """Test XGBoost param counting when trees_to_dataframe returns empty DataFrame."""
    # Create a mock XGBClassifier
    mock_model = Mock(spec=XGBClassifier)
    mock_booster = Mock()
    mock_booster.trees_to_dataframe.return_value = pd.DataFrame()  # Empty DataFrame
    mock_model.get_booster.return_value = mock_booster

    # Test that it returns 0 for empty dataframe (covers line 275)
    param_count = sa._get_model_param_count_xgb(mock_model)
    msg = "Should return 0 for XGBoost model with empty trees dataframe"
    assert param_count == 0, msg


def test_get_attack_metrics_instances_no_results():
    """Test _get_attack_metrics_instances when results is None."""
    attack = sa.StructuralAttack()
    attack.results = None

    # This should test the return {} path (covers line 548)
    metrics = attack._get_attack_metrics_instances()
    assert metrics == {}, f"Should return empty dict when results is None not {metrics}"


def test_torch_import_error():
    """Test torch ImportError handling during module import."""
    # Save the original module if it exists
    original_module = sys.modules.get("sacroml.attacks.structural_attack")

    try:
        # Remove the module from cache to force reimport
        if "sacroml.attacks.structural_attack" in sys.modules:
            del sys.modules["sacroml.attacks.structural_attack"]

        # Mock torch import to raise ImportError
        with patch("builtins.__import__") as mock_import:

            def side_effect(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return importlib.__import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            # Import the module which should trigger the ImportError
            reloaded_sa = importlib.import_module("sacroml.attacks.structural_attack")

            # Verify torch is None after ImportError
            assert reloaded_sa.torch is None, "torch should be None after ImportError"

    finally:
        # Restore the original module
        if original_module is not None:
            sys.modules["sacroml.attacks.structural_attack"] = original_module

        # Reload the original module to restore normal state
        importlib.reload(sa)


def get_target_xor(modeltype: str, reps: int = 10, **kwparams: dict) -> Target:
    """Load simple XOR-based dataset and create target of the desired type."""
    X = np.vstack(([[0, 0]] * reps, [[1, 0]] * reps, [[0, 1]] * reps, [[1, 1]] * reps))
    y = np.abs(X[:, 0] - X[:, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=1
    )

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
    return Target(
        model=target_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


kwargs_dtsafe = {"max_depth": 1, "min_samples_leaf": 200, "random_state": 0}
kwargs_dtunsafe = {
    "max_depth": None,
    "splitter": "best",
    "min_samples_leaf": 1,
    "criterion": "entropy",
    "random_state": 0,
}


def test_dt_nondisclosive_xor_reporting():
    """Test for safe decision tree classifier with full reporting xor."""
    target_dtsafe = get_target_xor("dt", reps=20, **kwargs_dtsafe)

    outputdir = "outputs"

    myattack_dtsafe = sa.StructuralAttack(report_individual=True, output_dir=outputdir)

    output = myattack_dtsafe.attack(target_dtsafe)
    assert not myattack_dtsafe.results.dof_risk, (
        "should be no DoF risk with decision stump"
    )
    assert not myattack_dtsafe.results.k_anonymity_risk, (
        "should be no k-anonymity risk with min_samples_leaf 150"
    )
    assert not myattack_dtsafe.results.class_disclosure_risk, (
        "no class disclosure risk for stump with min samples leaf 150"
    )
    assert not myattack_dtsafe.results.unnecessary_risk, (
        "not unnecessary risk if max_depth < 3.5"
    )
    gm = output["metadata"]["global_metrics"]
    inst = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    for metric in [
        "unnecessary_risk",
        "dof_risk",
        "k_anonymity_risk",
        "class_disclosure_risk",
        "smallgroup_risk",
    ]:
        assert not gm[metric], f"global metric {metric} should be false"
        assert not inst[metric], f"instance metric {metric} should be false"

    assert inst["individual"]["k_anonymity"] == [64] * 64, (
        "indiv records k_anon should all be 64"
    )
    for metric in [
        "unnecessary_risk",
        "dof_risk",
        "class_disclosure",
        "smallgroup_risk",
    ]:
        assert inst["individual"][metric] == [False] * 64, (
            f"individual records for {metric} should all be False"
        )


def test_dt_disclosive_xor_reporting():
    """Test for risky decision tree classifier."""
    target_dtunsafe = get_target_xor("dt", reps=2, **kwargs_dtunsafe)

    outputdir = "outputs"

    myattack_dtunsafe = sa.StructuralAttack(
        report_individual=True, output_dir=outputdir
    )
    output = myattack_dtunsafe.attack(target_dtunsafe)
    assert myattack_dtunsafe.results.dof_risk, (
        "should be  DoF risk with unlimited complexity decision tree"
    )
    assert myattack_dtunsafe.results.k_anonymity_risk, (
        "should be  k-anonymity risk with unlimited complexity decision tree"
    )
    assert myattack_dtunsafe.results.class_disclosure_risk, (
        "should be class disclosure risk with unlimited complexity decision tree"
    )
    assert myattack_dtunsafe.results.unnecessary_risk, (
        " unnecessary risk with unlimited unlimited complexity decision tree"
    )

    assert myattack_dtunsafe.results.smallgroup_risk, (
        "small group risk with unlimited complexity decision tree"
    )

    gm = output["metadata"]["global_metrics"]
    inst = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    for metric in [
        "unnecessary_risk",
        "dof_risk",
        "k_anonymity_risk",
        "class_disclosure_risk",
        "smallgroup_risk",
    ]:
        assert gm[metric], f"global metric {metric} should be True"
        assert inst[metric], f"instance metric {metric} should be True"

    assert inst["individual"]["k_anonymity"] == [2, 1, 2, 2, 1, 2], (
        "k_anon records should be [2, 1, 2, 2, 1, 2] "
        f"not {inst['individual']['k_anonymity']}"
    )
    for metric in [
        "unnecessary_risk",
        "dof_risk",
        "class_disclosure",
        "smallgroup_risk",
    ]:
        assert inst["individual"][metric] == [True] * 6, (
            f"individual records for {metric} should all be True"
        )
