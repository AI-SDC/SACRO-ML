"""test_worst_case_attack.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>
"""
from unittest.mock import patch
import pytest
<<<<<<< HEAD
from aisdc.attacks import worst_case_attack, failfast.FailFast  # pylint: disable = import-error

def test_check_attack_success():
    """removes unwanted files or directory"""
    metrics={}
    metrics["AUC"]=0.80
    metrics["ACC"]=0.90
    
    # Option 1: AUC with greater than or equal to 
=======
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from aisdc.attacks import (  # pylint: disable = import-error
    dataset,
    failfast,
    worst_case_attack,
)


def test_check_attack_success():
    """removes unwanted files or directory"""
    metrics["AUC"] = 0.80
    metrics["ACC"] = 0.90

    # Option 1: AUC with greater than or equal to
>>>>>>> c9f5d7340616a28375d5fbd5db3b3f34a6a2aedd
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.6,
        attack_metric_success_comp_type="gte",
    )
    failfast_metric_summary = FailFast(args)
    assert failfast_metric_summary.check_attack_success(metrics) is True

    # Option 1: AUC with greater than or equal to
    args = worst_case_attack.WorstCaseAttackArgs(
        attack_metric_success_name="AUC",
        attack_metric_success_thresh=0.99,
        attack_metric_success_comp_type="gte",
    )
    failfast_metric_summary = FailFast(args)
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
