"""Test LiRA attack."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks import likelihood_attack
from aisdc.attacks.likelihood_attack import DummyClassifier, LIRAAttack
from aisdc.attacks.target import Target

N_SHADOW_MODELS = 20
LR_CONFIG = os.path.normpath("tests/attacks/lrconfig.json")
LR_CMD_CONFIG = os.path.normpath("tests/attacks/lrconfig_cmd.json")


@pytest.fixture(name="dummy_classifier_setup")
def fixture_dummy_classifier_setup():
    """Setup common things for DummyClassifier."""
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
    """Setup common things for LiRA."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    target_model = RandomForestClassifier(
        n_estimators=100, min_samples_split=2, min_samples_leaf=1
    )
    target_model.fit(train_X, train_y)
    target = Target(target_model)
    target.add_processed_data(train_X, train_y, test_X, test_y)
    target.save(path="test_lira_target")
    # Dump training and test data to csv
    np.savetxt(
        "train_data.csv",
        np.hstack((train_X, train_y[:, None])),
        delimiter=",",
    )
    np.savetxt("test_data.csv", np.hstack((test_X, test_y[:, None])), delimiter=",")
    # dump the training and test predictions into files
    np.savetxt(
        "train_preds.csv",
        target_model.predict_proba(train_X),
        delimiter=",",
    )
    np.savetxt("test_preds.csv", target_model.predict_proba(test_X), delimiter=",")
    return target


def test_lira_attack(lira_classifier_setup):
    """Tests the lira code two ways."""
    target = lira_classifier_setup
    attack_obj = LIRAAttack(
        n_shadow_models=N_SHADOW_MODELS,
        output_dir="test_output_lira",
        attack_config_json_file_name=LR_CONFIG,
    )
    attack_obj.setup_example_data()
    attack_obj.attack_from_config()
    attack_obj.example()

    attack_obj2 = LIRAAttack(
        n_shadow_models=N_SHADOW_MODELS,
        output_dir="test_output_lira",
        report_name="lira_example1_report",
    )
    attack_obj2.attack(target)
    output2 = attack_obj2.make_report()
    n_shadow_models_trained = output2["attack_experiment_logger"][
        "attack_instance_logger"
    ]["instance_0"]["n_shadow_models_trained"]
    n_shadow_models = output2["metadata"]["experiment_details"]["n_shadow_models"]
    assert n_shadow_models_trained == n_shadow_models


def test_check_and_update_dataset(lira_classifier_setup):
    """Test removal from test set with classes not present in training set."""
    target = lira_classifier_setup
    attack_obj = LIRAAttack(n_shadow_models=N_SHADOW_MODELS)

    # now make test[0] have a  class not present in training set#
    local_test_y = np.copy(target.y_test)
    local_test_y[0] = 5
    local_target = Target(target.model)
    local_target.add_processed_data(
        target.x_train, target.y_train, target.x_test, local_test_y
    )
    unique_classes_pre = set(local_test_y)
    n_test_examples_pre = len(local_test_y)
    local_target = attack_obj._check_and_update_dataset(  # pylint: disable=protected-access
        local_target
    )

    unique_classes_post = set(local_target.y_test)
    n_test_examples_post = len(local_target.y_test)

    assert local_target.y_test[0] != 5
    assert (n_test_examples_pre - n_test_examples_post) == 1
    class_diff = unique_classes_pre - unique_classes_post
    assert class_diff == {5}

    #  Test command line example.
    testargs = [
        "prog",
        "run-example",
        "--output-dir",
        "test_output_lira",
        "--report-name",
        "commandline_lira_example2_report",
    ]
    with patch.object(sys, "argv", testargs):
        likelihood_attack.main()

    # Test command line with a config file.
    testargs = [
        "prog",
        "run-attack",
        "-j",
        LR_CONFIG,
        "--output-dir",
        "test_output_lira",
        "--report-name",
        "commandline_lira_example1_report",
    ]
    with patch.object(sys, "argv", testargs):
        likelihood_attack.main()

    #  Test command line with a config file.
    testargs = [
        "prog",
        "run-attack-from-configfile",
        "-j",
        LR_CMD_CONFIG,
        "-t",
        "test_lira_target",
    ]
    with patch.object(sys, "argv", testargs):
        likelihood_attack.main()

    # Test command line example data creation.
    testargs = [
        "prog",
        "setup-example-data",
    ]
    with patch.object(sys, "argv", testargs):
        likelihood_attack.main()


def test_lira_attack_failfast_example():
    """Tests the lira code two ways."""
    attack_obj = LIRAAttack(
        n_shadow_models=N_SHADOW_MODELS,
        output_dir="test_output_lira",
        attack_config_json_file_name=LR_CONFIG,
        shadow_models_fail_fast=True,
        n_shadow_rows_confidences_min=10,
    )
    attack_obj.setup_example_data()
    attack_obj.attack_from_config()
    attack_obj.example()


def test_lira_attack_failfast_from_scratch1(lira_classifier_setup):
    """Test by training a model from scratch."""
    target = lira_classifier_setup
    attack_obj = LIRAAttack(
        n_shadow_models=N_SHADOW_MODELS,
        output_dir="test_output_lira",
        report_name="lira_example2_failfast_report",
        attack_config_json_file_name=LR_CONFIG,
        shadow_models_fail_fast=True,
        n_shadow_rows_confidences_min=10,
    )
    attack_obj.attack(target)
    output = attack_obj.make_report()
    n_shadow_models_trained = output["attack_experiment_logger"][
        "attack_instance_logger"
    ]["instance_0"]["n_shadow_models_trained"]
    n_shadow_models = output["metadata"]["experiment_details"]["n_shadow_models"]
    assert n_shadow_models_trained == n_shadow_models


def test_lira_attack_failfast_from_scratch2(lira_classifier_setup):
    """Test by training a model from scratch."""
    target = lira_classifier_setup
    attack_obj = LIRAAttack(
        n_shadow_models=150,
        output_dir="test_output_lira",
        report_name="lira_example3_failfast_report",
        attack_config_json_file_name=LR_CONFIG,
        shadow_models_fail_fast=True,
        n_shadow_rows_confidences_min=10,
    )
    attack_obj.attack(target)
    output = attack_obj.make_report()
    n_shadow_models_trained = output["attack_experiment_logger"][
        "attack_instance_logger"
    ]["instance_0"]["n_shadow_models_trained"]
    n_shadow_models = output["metadata"]["experiment_details"]["n_shadow_models"]
    assert n_shadow_models_trained < n_shadow_models
