"""test_worst_case_attack.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>
"""
from aisdc.attacks import failfast, worst_case_attack  # pylint: disable = import-error


def test_parse_boolean_argument():
    """removes unwanted files or directory"""
    metrics = {}
    metrics["ACC"] = 0.9
    metrics["AUC"] = 0.8

    # Option 1
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="lte",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is False

    # Option 2
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.99,
        attack_metric_success_comp_type="lte",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is True

    # Option 3
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="lte",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is False

    # Option 4
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.99,
        attack_metric_success_comp_type="lte",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is True

    # Option 5
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="gte",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is True

    # Option 6
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.99,
        attack_metric_success_comp_type="gte",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is False

    # Option 7
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="gt",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is True

    # Option 8
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.99,
        attack_metric_success_comp_type="gt",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is False

    # Option 9
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.8,
        attack_metric_success_comp_type="eq",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is True

    # Option 10
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="eq",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is False

    # Option 11
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.8,
        attack_metric_success_comp_type="not_eq",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is False

    # Option 12
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="not_eq",
    )
    failfast_Obj = failfast.FailFast(args)
    assert failfast_Obj.check_attack_success(metrics) is True

    assert failfast_Obj.get_fail_count() == 0
