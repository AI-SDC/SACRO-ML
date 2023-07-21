"""Test_worst_case_attack.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>.
"""
from aisdc.attacks import failfast, worst_case_attack  # pylint: disable = import-error


def test_parse_boolean_argument():
    """Test all comparison operators and both options for attack
    being successful and not successful given a metric and
    comparison operator with a threshold value.
    """
    metrics = {}
    metrics["ACC"] = 0.9
    metrics["AUC"] = 0.8
    metrics["P_HIGHER_AUC"] = 0.05

    # Option 1
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.04,
        attack_metric_success_comp_type="lte",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert not failfast_Obj.check_attack_success(metrics)

    # Option 2
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.06,
        attack_metric_success_comp_type="lte",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert failfast_Obj.check_attack_success(metrics)

    # Option 3
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.04,
        attack_metric_success_comp_type="lt",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert not failfast_Obj.check_attack_success(metrics)

    # Option 4
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.06,
        attack_metric_success_comp_type="lt",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert failfast_Obj.check_attack_success(metrics)

    # Option 5
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.04,
        attack_metric_success_comp_type="gte",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert failfast_Obj.check_attack_success(metrics)

    # Option 6
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.06,
        attack_metric_success_comp_type="gte",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert not failfast_Obj.check_attack_success(metrics)

    # Option 7
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.04,
        attack_metric_success_comp_type="gt",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert failfast_Obj.check_attack_success(metrics)

    # Option 8
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.06,
        attack_metric_success_comp_type="gt",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert not failfast_Obj.check_attack_success(metrics)

    # Option 9
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.05,
        attack_metric_success_comp_type="eq",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert failfast_Obj.check_attack_success(metrics)

    # Option 10
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.06,
        attack_metric_success_comp_type="eq",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert not failfast_Obj.check_attack_success(metrics)

    # Option 11
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.05,
        attack_metric_success_comp_type="not_eq",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert not failfast_Obj.check_attack_success(metrics)

    # Option 12
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.06,
        attack_metric_success_comp_type="not_eq",
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    assert failfast_Obj.check_attack_success(metrics)

    assert failfast_Obj.get_fail_count() == 0


def test_attack_success_fail_counts_and_overall_attack_success():
    """Test success and fail counts of attacks for a given threshold
    of a given metric based on a given comparison operation and
    also test overall attack successes using
    count threshold of attack being successful or not successful.
    """
    metrics = {}
    metrics["ACC"] = 0.9
    metrics["AUC"] = 0.8
    metrics["P_HIGHER_AUC"] = 0.08
    attack_obj = worst_case_attack.WorstCaseAttack(
        attack_metric_success_name="P_HIGHER_AUC",
        attack_metric_success_thresh=0.05,
        attack_metric_success_comp_type="lte",
        attack_metric_success_count_thresh=3,
    )
    failfast_Obj = failfast.FailFast(attack_obj)
    _ = failfast_Obj.check_attack_success(metrics)
    metrics["P_HIGHER_AUC"] = 0.07
    _ = failfast_Obj.check_attack_success(metrics)
    metrics["P_HIGHER_AUC"] = 0.03
    _ = failfast_Obj.check_attack_success(metrics)
    assert not failfast_Obj.check_overall_attack_success(attack_obj)
    metrics["P_HIGHER_AUC"] = 0.02
    _ = failfast_Obj.check_attack_success(metrics)
    metrics["P_HIGHER_AUC"] = 0.01
    _ = failfast_Obj.check_attack_success(metrics)
    assert failfast_Obj.get_success_count() == 3
    assert failfast_Obj.get_fail_count() == 2
    assert failfast_Obj.check_overall_attack_success(attack_obj)
