"""test_worst_case_attack.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>
"""
from unittest.mock import patch

import pytest

from aisdc.attacks import failfast.FailFast,
from aisdc.attacks import worst_case_attack,

def test_check_attack_success():
    """removes unwanted files or directory"""
    metrics={}
    metrics["AUC"]=0.80
    metrics["ACC"]=0.90

    # Option 1: AUC with greater than or equal to
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.99,
        attack_metric_success_comp_type="gte",
    )
    failfast_metric_summary = failast.FailFast(args)
    assert failfast_metric_summary.check_attack_success(metrics) is False

    # Option 2: AUC with greater than or equal to
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="gte",
    )
    failfast_metric_summary = FailFast(args)
    assert failfast_metric_summary.check_attack_success(metrics) is True

    # Option 3: AUC with greater than or equal to
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="lte",
    )
    failfast_metric_summary = FailFast(args)
    assert failfast_metric_summary.check_attack_success(metrics) is False

    # Option 4: AUC with greater than or equal to
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.99,
        attack_metric_success_comp_type="lte",
    )
    failfast_metric_summary = FailFast(args)
    assert failfast_metric_summary.check_attack_success(metrics) is True
