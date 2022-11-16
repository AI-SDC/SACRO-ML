"""test_worst_case_attack.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>
"""
import os
import sys
from unittest.mock import patch

import numpy as np
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
