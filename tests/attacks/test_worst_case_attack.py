"""Test worst case attack."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import SVC

from sacroml.attacks import worst_case_attack
from sacroml.attacks.target import Target


def pytest_generate_tests(metafunc):
    """Generate target model for testing."""
    if "get_target" in metafunc.fixturenames:
        metafunc.parametrize(
            "get_target", [SVC(gamma=0.1, probability=True)], indirect=True
        )


@pytest.fixture(name="common_setup")
def fixture_common_setup(get_target):
    """Get ready to test some code."""
    target = get_target
    target.model.fit(target.X_train, target.y_train)
    return target


def test_insufficient_target():
    """Test insufficient target details to run."""
    target = Target()
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    output = attack_obj.attack(target)
    assert not output


def test_report_worstcase(common_setup):
    """Test worst case attack directly."""
    target = common_setup

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
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
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    attack_obj.attack(target)


def test_attack_with_correct_feature(common_setup):
    """Test the attack when the model correctness feature is used."""
    target = common_setup

    # with multiple reps
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=1,
        n_dummy_reps=1,
        p_thresh=0.05,
        test_prop=0.5,
        include_model_correct_feature=True,
    )
    attack_obj.attack(target)

    # Check that attack_metrics has the Yeom metrics
    assert "yeom_tpr" in attack_obj.attack_metrics[0]
    assert "yeom_fpr" in attack_obj.attack_metrics[0]
    assert "yeom_advantage" in attack_obj.attack_metrics[0]


def test_attack_from_predictions(common_setup):
    """Checks code that runs attacks from predictions."""
    target = common_setup

    ytr_pred = target.model.predict_proba(target.X_train)
    yte_pred = target.model.predict_proba(target.X_test)

    new_target = Target(model=target.model, proba_train=ytr_pred, proba_test=yte_pred)

    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    attack_obj.attack(new_target)


def test_attack_from_predictions_no_dummy(common_setup):
    """Checks code that runs attacks from predictions."""
    target = common_setup

    ytr_pred = target.model.predict_proba(target.X_train)
    yte_pred = target.model.predict_proba(target.X_test)

    new_target = Target(model=target.model, proba_train=ytr_pred, proba_test=yte_pred)

    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=0,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    attack_obj.attack(new_target)


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


def test_wc_multiclass(get_target_multiclass):
    """Test WorstCaseAttack with multiclass data."""
    target = get_target_multiclass
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=5,
        n_dummy_reps=0,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    output = attack_obj.attack(target)
    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert 0 <= metrics["TPR"] <= 1
    assert 0 <= metrics["FPR"] <= 1
