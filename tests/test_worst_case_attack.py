"""Examples for using the 'worst case' attack code.

This code simulates a MIA attack providing the attacker with as much information as possible.
i.e. they have a subset of rows that they _know_ were used for training. And a subset that they
know were not. They also have query access to the target model.

They pass the training and non-training rows through the target model to get the predictive
probabilities. These are then used to train an _attack model_. And the attack model is evaluated
to see how well it can predict whether or not other examples were in the training set or not.

The code can be called from the command line, or accessed programmatically. Examples of both
are shown below.

In the code, [Researcher] and [TRE] are used in comments to denote which bit is done by whom

Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.worst_case_attack_example

"""
import os
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from attacks import dataset, worst_case_attack  # pylint: disable = import-error


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
    from unittest.mock import patch

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
