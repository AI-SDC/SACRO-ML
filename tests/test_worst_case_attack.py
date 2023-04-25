"""test_worst_case_attack.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from aisdc.attacks import dataset, worst_case_attack  # pylint: disable = import-error


def clean_up(name):
    """removes unwanted files or directory"""
    if os.path.exists(name) and os.path.isfile(name):
        os.remove(name)


def test_report_worstcase():
    """tests worst case attack directly"""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    dataset_obj = dataset.Data()
    dataset_obj.add_processed_data(train_X, train_y, test_X, test_y)

    target_model = SVC(gamma=0.1, probability=True)
    target_model.fit(train_X, train_y)
    _ = target_model.predict_proba(train_X)
    _ = target_model.predict_proba(test_X)

    args = worst_case_attack.WorstCaseAttackArgs(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        in_sample_filename=None,
        out_sample_filename=None,
        test_prop=0.5,
        report_name="test-10reps",
    )

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(args)
    attack_obj.attack(dataset_obj, target_model)
    # attack_obj.make_dummy_data() cause exception when used like this!
    _ = attack_obj.make_report()

    # with one rep
    args = worst_case_attack.WorstCaseAttackArgs(
        n_reps=1,
        n_dummy_reps=1,
        p_thresh=0.05,
        in_sample_filename=None,
        out_sample_filename=None,
        test_prop=0.5,
        report_name="test-1rep",
    )

    attack_obj = worst_case_attack.WorstCaseAttack(args)
    attack_obj.attack(dataset_obj, target_model)
    _ = attack_obj.make_report()


def test_attack_with_correct_feature():
    """Test the attack when the model correctness feature is used"""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    dataset_obj = dataset.Data()
    dataset_obj.add_processed_data(train_X, train_y, test_X, test_y)

    target_model = SVC(gamma=0.1, probability=True)
    target_model.fit(train_X, train_y)

    args = worst_case_attack.WorstCaseAttackArgs(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=1,
        n_dummy_reps=1,
        p_thresh=0.05,
        in_sample_filename=None,
        out_sample_filename=None,
        test_prop=0.5,
        report_name="test-1rep",
        include_model_correct_feature=True,
    )

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(args)
    attack_obj.attack(dataset_obj, target_model)

    # Check that attack_metrics has the Yeom metrics
    assert "yeom_tpr" in attack_obj.attack_metrics[0]
    assert "yeom_fpr" in attack_obj.attack_metrics[0]
    assert "yeom_advantage" in attack_obj.attack_metrics[0]


def test_attack_from_predictions():
    """checks code that runs attacks from predictions"""

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    dataset_obj = dataset.Data()
    dataset_obj.add_processed_data(train_X, train_y, test_X, test_y)

    target_model = SVC(gamma=0.1, probability=True)
    target_model.fit(train_X, train_y)
    ytr_pred = target_model.predict_proba(train_X)
    yte_pred = target_model.predict_proba(test_X)
    np.savetxt("ypred_train.csv", ytr_pred, delimiter=",")
    np.savetxt("ypred_test.csv", yte_pred, delimiter=",")

    args = worst_case_attack.WorstCaseAttackArgs(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        in_sample_filename="ypred_train.csv",
        out_sample_filename="ypred_test.csv",
        test_prop=0.5,
        report_name="test-10reps",
    )

    assert args.get_args()["in_sample_filename"] == "ypred_train.csv"
    print(args)

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(args)
    attack_obj.attack_from_prediction_files()


def test_attack_from_predictions_no_dummy():
    """checks code that runs attacks from predictions"""

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    dataset_obj = dataset.Data()
    dataset_obj.add_processed_data(train_X, train_y, test_X, test_y)

    target_model = SVC(gamma=0.1, probability=True)
    target_model.fit(train_X, train_y)
    ytr_pred = target_model.predict_proba(train_X)
    yte_pred = target_model.predict_proba(test_X)
    np.savetxt("ypred_train.csv", ytr_pred, delimiter=",")
    np.savetxt("ypred_test.csv", yte_pred, delimiter=",")

    args = worst_case_attack.WorstCaseAttackArgs(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=10,
        n_dummy_reps=0,
        p_thresh=0.05,
        in_sample_filename="ypred_train.csv",
        out_sample_filename="ypred_test.csv",
        test_prop=0.5,
        report_name="test-10reps",
    )

    assert args.get_args()["in_sample_filename"] == "ypred_train.csv"
    print(args)

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(args)
    attack_obj.attack_from_prediction_files()


def test_dummy_data():
    """test functionality around creating dummy data"""
    args = worst_case_attack.WorstCaseAttackArgs(
        # How many attacks to run -- in each the attack model is trained on a different
        # subset of the data
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        in_sample_filename="ypred_train.csv",
        out_sample_filename="ypred_test.csv",
        test_prop=0.5,
        report_name="test-10reps",
    )

    attack_obj = worst_case_attack.WorstCaseAttack(args)
    attack_obj.make_dummy_data()


def test_attack_data_prep():
    """test the method that prepares the attack data"""
    args = worst_case_attack.WorstCaseAttackArgs()
    attack_obj = worst_case_attack.WorstCaseAttack(args)
    train_preds = np.array([[1, 0], [0, 1]], int)
    test_preds = np.array([[2, 0], [0, 2]], int)

    mi_x, mi_y = attack_obj._prepare_attack_data(  # pylint: disable=protected-access
        train_preds, test_preds
    )
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], np.int))
    # Test the x data produced. Each row should be sorted in descending order
    np.testing.assert_array_equal(mi_x, np.array([[1, 0], [1, 0], [2, 0], [2, 0]]))

    # With sort_probs = False, the rows of x should not be sorted
    args = worst_case_attack.WorstCaseAttackArgs(sort_probs=False)
    attack_obj = worst_case_attack.WorstCaseAttack(args)
    mi_x, mi_y = attack_obj._prepare_attack_data(  # pylint: disable=protected-access
        train_preds, test_preds
    )
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], np.int))
    np.testing.assert_array_equal(mi_x, np.array([[1, 0], [0, 1], [2, 0], [0, 2]]))


def test_attack_data_prep_with_correct_feature():
    """test the method that prepares the attack data.
    This time, testing that the model correctness values are added, are always
    the final feature, and are not included in the sorting"""
    args = worst_case_attack.WorstCaseAttackArgs(include_model_correct_feature=True)
    attack_obj = worst_case_attack.WorstCaseAttack(args)
    train_preds = np.array([[1, 0], [0, 1]], int)
    test_preds = np.array([[2, 0], [0, 2]], int)
    train_correct = np.array([1, 0], int)
    test_correct = np.array([0, 1], int)

    mi_x, mi_y = attack_obj._prepare_attack_data(  # pylint: disable=protected-access
        train_preds, test_preds, train_correct=train_correct, test_correct=test_correct
    )
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], np.int))
    # Test the x data produced. Each row should be sorted in descending order
    np.testing.assert_array_equal(
        mi_x, np.array([[1, 0, 1], [1, 0, 0], [2, 0, 0], [2, 0, 1]])
    )

    # With sort_probs = False, the rows of x should not be sorted
    args = worst_case_attack.WorstCaseAttackArgs(
        sort_probs=False, include_model_correct_feature=True
    )
    attack_obj = worst_case_attack.WorstCaseAttack(args)
    mi_x, mi_y = attack_obj._prepare_attack_data(  # pylint: disable=protected-access
        train_preds, test_preds, train_correct=train_correct, test_correct=test_correct
    )
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], np.int))
    np.testing.assert_array_equal(
        mi_x, np.array([[1, 0, 1], [0, 1, 0], [2, 0, 0], [0, 2, 1]])
    )


def test_non_rf_mia():
    """Tests that it is possible to set the attack model via the args
    In this case, we set as a SVC. But we set probability to false. If the code does
    indeed try and use the SVC (as we want) it will fail as it will try and access
    the predict_proba which won't work if probability=False. Hence, if the code throws
    an AttributeError we now it *is* trying to use the SVC"""

    args = worst_case_attack.WorstCaseAttackArgs(
        mia_attack_model=SVC,
        mia_attack_model_hyp={"kernel": "rbf", "probability": False},
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    dataset_obj = dataset.Data()
    dataset_obj.add_processed_data(train_X, train_y, test_X, test_y)

    target_model = SVC(gamma=0.1, probability=True)
    target_model.fit(train_X, train_y)
    ytr_pred = target_model.predict_proba(train_X)
    yte_pred = target_model.predict_proba(test_X)

    attack_obj = worst_case_attack.WorstCaseAttack(args)
    with pytest.raises(AttributeError):
        attack_obj.attack_from_preds(ytr_pred, yte_pred)


def test_main():
    """test invocation via command line"""

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
    """gets rid of files created during tests"""
    names = [
        "worstcase_report.pdf",
        "log_roc.png",
        "worstcase_report.json",
        "test_preds.csv",
        "train_preds.csv",
        "ypred_test.csv",
        "ypred_train.csv",
        "test-1rep.pdf",
        "test-1rep.json",
        "test-10reps.pdf",
        "test-10reps.json",
    ]
    for name in names:
        clean_up(name)
