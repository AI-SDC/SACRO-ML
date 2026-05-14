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

    mi_x, mi_y = attack_obj._prepare_attack_data(train_preds, test_preds)
    np.testing.assert_array_equal(mi_y, np.array([1, 1, 0, 0], int))
    # Test the x data produced. Each row should be sorted in descending order
    np.testing.assert_array_equal(mi_x, np.array([[1, 0], [1, 0], [2, 0], [2, 0]]))
    # With sort_probs = False, the rows of x should not be sorted
    attack_obj = worst_case_attack.WorstCaseAttack(sort_probs=False)
    mi_x, mi_y = attack_obj._prepare_attack_data(train_preds, test_preds)
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

    mi_x, mi_y = attack_obj._prepare_attack_data(
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
    mi_x, mi_y = attack_obj._prepare_attack_data(
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


def test_tuning_default_grid(common_setup):
    """The "default" shortcut tunes the RF attack model and reports best params."""
    target = common_setup
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=3,
        n_dummy_reps=0,
        test_prop=0.5,
        output_dir="test_output_worstcase",
        attack_model_param_grid="default",
    )
    output = attack_obj.attack(target)

    assert len(attack_obj.attack_metrics) == 3
    tuning = output["metadata"]["tuning"]
    assert tuning["search_type"] == "grid"
    assert tuning["tuning_metric"] == "AUC"
    # RF default grid keys should drive best_params.
    assert set(tuning["best_params"]).issubset(
        {"min_samples_split", "min_samples_leaf", "max_depth"}
    )
    assert tuning["n_candidates"] == 3 * 3 * 3
    assert attack_obj._tuned_params is not None
    # cv_results summary: one entry per candidate, with score + rank columns.
    assert len(tuning["cv_results"]) == tuning["n_candidates"]
    expected_cols = {"params", "mean_test_score", "std_test_score", "rank_test_score"}
    assert set(tuning["cv_results"][0]) == expected_cols
    ranks = sorted(c["rank_test_score"] for c in tuning["cv_results"])
    assert ranks[0] == 1  # best candidate has rank 1
    # The candidate with rank 1 should carry the reported best_params.
    best_candidate = next(c for c in tuning["cv_results"] if c["rank_test_score"] == 1)
    assert best_candidate["params"] == tuning["best_params"]
    # Per-fold scores for the winner: one per rep, all finite floats.
    per_fold = tuning["best_candidate_per_fold_scores"]
    assert len(per_fold) == 3
    assert all(isinstance(s, float) for s in per_fold)


def test_tuning_custom_grid_reused_by_dummies(common_setup):
    """A custom dict grid is honoured and tuned params are reused for dummies."""
    target = common_setup
    grid = {
        "min_samples_split": [2, 20],
        "max_depth": [3, None],
    }
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=2,
        n_dummy_reps=1,
        test_prop=0.5,
        output_dir="test_output_worstcase",
        attack_model_param_grid=grid,
    )
    output = attack_obj.attack(target)

    assert len(attack_obj.attack_metrics) == 2
    assert len(attack_obj.dummy_attack_metrics) == 1
    tuning = output["metadata"]["tuning"]
    assert set(tuning["best_params"]).issubset({"min_samples_split", "max_depth"})
    assert tuning["n_candidates"] == 4
    # Dummies must reuse the same tuned params; tuning should run exactly once.
    assert attack_obj._tuning_info is tuning or attack_obj._tuning_info == tuning


def test_tuning_random_search_with_alt_scorer(common_setup):
    """Randomised search with a non-default MIA scorer."""
    target = common_setup
    grid = {
        "min_samples_split": [2, 5, 10, 20],
        "max_depth": [3, 5, 10, None],
    }
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=3,
        n_dummy_reps=0,
        test_prop=0.5,
        output_dir="test_output_worstcase",
        attack_model_param_grid=grid,
        search_type="random",
        search_n_iter=3,
        tuning_metric="TPR@0.1",
    )
    output = attack_obj.attack(target)

    tuning = output["metadata"]["tuning"]
    assert tuning["search_type"] == "random"
    assert tuning["tuning_metric"] == "TPR@0.1"
    assert tuning["n_candidates"] == 3


def test_tuning_requires_n_reps_at_least_two(common_setup):
    """A single rep cannot drive cross-validation."""
    target = common_setup
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=1,
        n_dummy_reps=0,
        test_prop=0.5,
        output_dir="test_output_worstcase",
        attack_model_param_grid="default",
    )
    with pytest.raises(ValueError, match="n_reps >= 2"):
        attack_obj.attack(target)


def test_tuning_default_for_unknown_model_raises():
    """The "default" shortcut errors when no default exists for the model."""
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_model="sklearn.svm.SVC",
        attack_model_param_grid="default",
    )
    with pytest.raises(ValueError, match="No default param grid"):
        attack_obj._resolve_param_grid()


def test_tuning_invalid_grid_string_raises():
    """Any string other than "default" is rejected."""
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_model_param_grid="not-a-real-shortcut",
    )
    with pytest.raises(ValueError, match="must be a dict"):
        attack_obj._resolve_param_grid()


def test_tuning_invalid_search_type_raises(common_setup):
    """An unrecognised search_type raises a clear error."""
    target = common_setup
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=2,
        n_dummy_reps=0,
        test_prop=0.5,
        output_dir="test_output_worstcase",
        attack_model_param_grid={"min_samples_split": [2, 20]},
        search_type="bogus",
    )
    with pytest.raises(ValueError, match="Unknown search_type"):
        attack_obj.attack(target)


def test_init_rejects_bad_search_n_iter():
    """`search_n_iter` must be a positive integer."""
    with pytest.raises(ValueError, match="search_n_iter"):
        worst_case_attack.WorstCaseAttack(search_n_iter=0)
    with pytest.raises(ValueError, match="search_n_iter"):
        worst_case_attack.WorstCaseAttack(search_n_iter=-1)


def test_init_bad_tuning_metric_falls_back_to_auc(caplog):
    """An unknown `tuning_metric` warns and falls back to AUC at construction."""
    with caplog.at_level("WARNING", logger="sacroml.attacks.worst_case_attack"):
        attack_obj = worst_case_attack.WorstCaseAttack(
            tuning_metric="not-a-real-metric"
        )
    assert attack_obj.tuning_metric == "AUC"
    assert callable(attack_obj._resolved_tuning_scorer)
    assert any("not-a-real-metric" in rec.message for rec in caplog.records)


def test_no_tuning_leaves_metadata_clean(common_setup):
    """When no grid is supplied, metadata has no "tuning" key."""
    target = common_setup
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=2,
        n_dummy_reps=0,
        test_prop=0.5,
        output_dir="test_output_worstcase",
    )
    output = attack_obj.attack(target)
    assert "tuning" not in output["metadata"]
    assert attack_obj._tuned_params is None
    assert attack_obj._tuning_info is None
