"""Test worst case attack."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from aisdc.attacks import worst_case_attack
from aisdc.attacks.target import Target


def test_report_worstcase():
    """Test worst case attack directly."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    X_train, X_test, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(X_train, train_y)
    _ = model.predict_proba(X_train)
    _ = model.predict_proba(X_test)

    target = Target(model=model)
    target.add_processed_data(X_train, train_y, X_test, test_y)

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename=None,
        test_preds_filename=None,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    attack_obj.attack(target)

    # with one rep
    attack_obj = worst_case_attack.WorstCaseAttack(
        reproduce_split=[5, 5],
        n_reps=1,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename=None,
        test_preds_filename=None,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    attack_obj.attack(target)


def test_attack_with_correct_feature():
    """Test the attack when the model correctness feature is used."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    X_train, X_test, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(X_train, train_y)

    target = Target(model=model)
    target.add_processed_data(X_train, train_y, X_test, test_y)

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=1,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename=None,
        test_preds_filename=None,
        test_prop=0.5,
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
    X_train, X_test, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(X_train, train_y)
    ytr_pred = model.predict_proba(X_train)
    yte_pred = model.predict_proba(X_test)
    np.savetxt("ypred_train.csv", ytr_pred, delimiter=",")
    np.savetxt("ypred_test.csv", yte_pred, delimiter=",")

    target = Target(model=model)
    target.add_processed_data(X_train, train_y, X_test, test_y)

    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename="ypred_train.csv",
        test_preds_filename="ypred_test.csv",
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )

    assert attack_obj.training_preds_filename == "ypred_train.csv"

    # with multiple reps
    attack_obj.attack_from_prediction_files()


def test_attack_from_predictions_no_dummy():
    """Checks code that runs attacks from predictions."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    X_train, X_test, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = SVC(gamma=0.1, probability=True)
    model.fit(X_train, train_y)
    ytr_pred = model.predict_proba(X_train)
    yte_pred = model.predict_proba(X_test)
    np.savetxt("ypred_train.csv", ytr_pred, delimiter=",")
    np.savetxt("ypred_test.csv", yte_pred, delimiter=",")

    target = Target(model=model)
    target.add_processed_data(X_train, train_y, X_test, test_y)

    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=0,
        p_thresh=0.05,
        training_preds_filename="ypred_train.csv",
        test_preds_filename="ypred_test.csv",
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )

    assert attack_obj.training_preds_filename == "ypred_train.csv"
    print(attack_obj)
    # with multiple reps
    attack_obj.attack_from_prediction_files()


def test_dummy_data():
    """Test functionality around creating dummy data."""
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        training_preds_filename="ypred_train.csv",
        test_preds_filename="ypred_test.csv",
        test_prop=0.5,
        output_dir="test_output_worstcase",
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
