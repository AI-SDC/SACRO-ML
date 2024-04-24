"""Test_worst_case_attack.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from aisdc.attacks import worst_case_attack
from aisdc.attacks.target import Target


def clean_up(name):
    """Removes unwanted files or directory."""
    if os.path.exists(name):
        if os.path.isfile(name):
            os.remove(name)
        elif os.path.isdir(name):
            shutil.rmtree(name)


def test_config_file_arguments_parsin():
    """Tests reading parameters from the configuration file."""
    config = {
        "n_reps": 12,
        "n_dummy_reps": 2,
        "p_thresh": 0.06,
        "test_prop": 0.4,
        "output_dir": "test_output_worstcase",
        "report_name": "programmatically_worstcase_example1_test",
    }
    with open("config_worstcase_test.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(config))
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_config_json_file_name="config_worstcase_test.json",
    )
    assert attack_obj.n_reps == config["n_reps"]
    assert attack_obj.n_dummy_reps == config["n_dummy_reps"]
    assert attack_obj.p_thresh == config["p_thresh"]
    assert attack_obj.test_prop == config["test_prop"]
    assert attack_obj.report_name == config["report_name"]
    os.remove("config_worstcase_test.json")


def test_attack_from_predictions_cmd():
    """Running attack using configuration file and prediction files."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    model = SVC(gamma=0.1, probability=True)
    model.fit(train_X, train_y)

    ytr_pred = model.predict_proba(train_X)
    yte_pred = model.predict_proba(test_X)
    np.savetxt("ypred_train.csv", ytr_pred, delimiter=",")
    np.savetxt("ypred_test.csv", yte_pred, delimiter=",")

    target = Target(model=model)
    target.add_processed_data(train_X, train_y, test_X, test_y)

    target.save(path="test_worstcase_target")

    config = {
        "n_reps": 30,
        "n_dummy_reps": 2,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "output_dir": "test_output_worstcase",
        "report_name": "commandline_worstcase_example1_report",
        "training_preds_filename": "ypred_train.csv",
        "test_preds_filename": "ypred_test.csv",
        "attack_metric_success_name": "P_HIGHER_AUC",
        "attack_metric_success_thresh": 0.05,
        "attack_metric_success_comp_type": "lte",
        "attack_metric_success_count_thresh": 2,
        "attack_fail_fast": True,
    }

    with open("config_worstcase_cmd.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(config))
    os.system(
        f"{sys.executable} -m aisdc.attacks.worst_case_attack run-attack-from-configfile "
        "--attack-config-json-file-name config_worstcase_cmd.json "
        "--attack-target-folder-path test_worstcase_target "
    )
    os.remove("config_worstcase_cmd.json")
    os.remove("ypred_train.csv")
    os.remove("ypred_test.csv")


def test_report_worstcase():
    """Tests worst case attack directly."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(train_X, train_y)
    _ = model.predict_proba(train_X)
    _ = model.predict_proba(test_X)

    target = Target(model=model)
    target.add_processed_data(train_X, train_y, test_X, test_y)

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename=None,
        test_preds_filename=None,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    attack_obj.attack(target)
    # attack_obj.make_dummy_data() cause exception when used like this!
    _ = attack_obj.make_report()

    # with one rep
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=1,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename=None,
        test_preds_filename=None,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    attack_obj.attack(target)
    _ = attack_obj.make_report()


def test_attack_with_correct_feature():
    """Test the attack when the model correctness feature is used."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(train_X, train_y)

    target = Target(model=model)
    target.add_processed_data(train_X, train_y, test_X, test_y)

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=1,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename=None,
        test_preds_filename=None,
        test_prop=0.5,
        report_name="test-1rep-programmatically_worstcase_example4_test",
        include_model_correct_feature=True,
    )
    attack_obj.attack(target)

    # Check that attack_metrics has the Yeom metrics
    assert "yeom_tpr" in attack_obj.attack_metrics[0]
    assert "yeom_fpr" in attack_obj.attack_metrics[0]
    assert "yeom_advantage" in attack_obj.attack_metrics[0]


def test_attack_from_predictions():
    """Checks code that runs attacks from predictions."""

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(train_X, train_y)
    ytr_pred = model.predict_proba(train_X)
    yte_pred = model.predict_proba(test_X)
    np.savetxt("ypred_train.csv", ytr_pred, delimiter=",")
    np.savetxt("ypred_test.csv", yte_pred, delimiter=",")

    target = Target(model=model)
    target.add_processed_data(train_X, train_y, test_X, test_y)

    attack_obj = worst_case_attack.WorstCaseAttack(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename="ypred_train.csv",
        test_preds_filename="ypred_test.csv",
        test_prop=0.5,
        output_dir="test_output_worstcase",
        report_name="test-10reps-programmatically_worstcase_example5_test",
    )

    assert attack_obj.training_preds_filename == "ypred_train.csv"

    # with multiple reps
    attack_obj.attack_from_prediction_files()


def test_attack_from_predictions_no_dummy():
    """Checks code that runs attacks from predictions."""

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(train_X, train_y)
    ytr_pred = model.predict_proba(train_X)
    yte_pred = model.predict_proba(test_X)
    np.savetxt("ypred_train.csv", ytr_pred, delimiter=",")
    np.savetxt("ypred_test.csv", yte_pred, delimiter=",")

    target = Target(model=model)
    target.add_processed_data(train_X, train_y, test_X, test_y)

    attack_obj = worst_case_attack.WorstCaseAttack(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=10,
        n_dummy_reps=0,
        p_thresh=0.05,
        training_preds_filename="ypred_train.csv",
        test_preds_filename="ypred_test.csv",
        test_prop=0.5,
        output_dir="test_output_worstcase",
        report_name="test-10reps-programmatically_worstcase_example6_test",
    )

    assert attack_obj.training_preds_filename == "ypred_train.csv"
    print(attack_obj)
    # with multiple reps
    attack_obj.attack_from_prediction_files()


def test_dummy_data():
    """Test functionality around creating dummy data."""
    attack_obj = worst_case_attack.WorstCaseAttack(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename="ypred_train.csv",
        test_preds_filename="ypred_test.csv",
        test_prop=0.5,
        output_dir="test_output_worstcase",
        report_name="test-10reps-programmatically_worstcase_example7_test",
    )

    attack_obj.make_dummy_data()


def test_attack_data_prep():
    """Test the method that prepares the attack data."""
    attack_obj = worst_case_attack.WorstCaseAttack()
    train_preds = np.array([[1, 0], [0, 1]], int)
    test_preds = np.array([[2, 0], [0, 2]], int)

    mi_x, mi_y = attack_obj._prepare_attack_data(  # pylint: disable=protected-access
        train_preds, test_preds
    )
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], int))
    # Test the x data produced. Each row should be sorted in descending order
    np.testing.assert_array_equal(mi_x, np.array([[1, 0], [1, 0], [2, 0], [2, 0]]))
    # With sort_probs = False, the rows of x should not be sorted
    attack_obj = worst_case_attack.WorstCaseAttack(sort_probs=False)
    mi_x, mi_y = attack_obj._prepare_attack_data(  # pylint: disable=protected-access
        train_preds, test_preds
    )
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], int))
    np.testing.assert_array_equal(mi_x, np.array([[1, 0], [0, 1], [2, 0], [0, 2]]))


def test_attack_data_prep_with_correct_feature():
    """Test the method that prepares the attack data.
    This time, testing that the model correctness values are added, are always
    the final feature, and are not included in the sorting.
    """
    attack_obj = worst_case_attack.WorstCaseAttack(include_model_correct_feature=True)
    train_preds = np.array([[1, 0], [0, 1]], int)
    test_preds = np.array([[2, 0], [0, 2]], int)
    train_correct = np.array([1, 0], int)
    test_correct = np.array([0, 1], int)

    mi_x, mi_y = attack_obj._prepare_attack_data(  # pylint: disable=protected-access
        train_preds, test_preds, train_correct=train_correct, test_correct=test_correct
    )
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], int))
    # Test the x data produced. Each row should be sorted in descending order
    np.testing.assert_array_equal(
        mi_x, np.array([[1, 0, 1], [1, 0, 0], [2, 0, 0], [2, 0, 1]])
    )

    # With sort_probs = False, the rows of x should not be sorted
    attack_obj = worst_case_attack.WorstCaseAttack(
        sort_probs=False, include_model_correct_feature=True
    )
    mi_x, mi_y = attack_obj._prepare_attack_data(  # pylint: disable=protected-access
        train_preds, test_preds, train_correct=train_correct, test_correct=test_correct
    )
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], int))
    np.testing.assert_array_equal(
        mi_x, np.array([[1, 0, 1], [0, 1, 0], [2, 0, 0], [0, 2, 1]])
    )


def test_non_rf_mia():
    """Tests that it is possible to set the attack model via the args
    In this case, we set as a SVC. But we set probability to false. If the code does
    indeed try and use the SVC (as we want) it will fail as it will try and access
    the predict_proba which won't work if probability=False. Hence, if the code throws
    an AttributeError we now it *is* trying to use the SVC.
    """

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(train_X, train_y)
    ytr_pred = model.predict_proba(train_X)
    yte_pred = model.predict_proba(test_X)

    target = Target(model=model)
    target.add_processed_data(train_X, train_y, test_X, test_y)

    attack_obj = worst_case_attack.WorstCaseAttack(
        mia_attack_model=SVC,
        mia_attack_model_hyp={"kernel": "rbf", "probability": False},
    )
    with pytest.raises(AttributeError):
        attack_obj.attack_from_preds(ytr_pred, yte_pred)


def test_main():
    """Test invocation via command line."""

    # option 1
    testargs = ["prog", "make-dummy-data"]
    with patch.object(sys, "argv", testargs):
        worst_case_attack.main()

    # option 2
    testargs = ["prog", "run-attack"]
    with patch.object(sys, "argv", testargs):
        worst_case_attack.main()

    # wrong args

    # testargs = ["prog", "run-attack","--no-such-arg"]
    # with patch.object(sys, 'argv', testargs):
    #    worst_case_attack.main()


def test_cleanup():
    """Gets rid of files created during tests."""
    names = [
        "test_output_worstcase",
        "output_worstcase",
        "test_worstcase_target",
        "test_preds.csv",
        "train_preds.csv",
        "ypred_test.csv",
        "ypred_train.csv",
    ]
    for name in names:
        clean_up(name)
