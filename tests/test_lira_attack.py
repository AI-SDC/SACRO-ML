"""test_lira_attack.py
Copyright (C) Jim Smith2022  <james.smith@uwe.ac.uk>
"""
# pylint: disable = duplicate-code

# import json
from unittest.mock import patch
import sys

# import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from attacks import likelihood_attack
from attacks.dataset import Data  # pylint: disable = import-error
from attacks.likelihood_attack import (  # pylint: disable = import-error
    LIRAAttack,
    LIRAAttackArgs,
)

# from sklearn.svm import SVC


def test_lira_attack():
    """tests the lira code two ways"""
    args = LIRAAttackArgs(n_shadow_models=50, report_name="lira_example_report")
    attack_obj = LIRAAttack(args)
    attack_obj.setup_example_data()
    attack_obj.attack_from_config()
    attack_obj.example()

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    dataset = Data()
    dataset.add_processed_data(train_X, train_y, test_X, test_y)

    # target_model = SVC(gamma=0.1, probability=True)
    target_model = RandomForestClassifier(
        n_estimators=100, min_samples_split=2, min_samples_leaf=1
    )
    target_model.fit(train_X, train_y)

    args2 = LIRAAttackArgs(n_shadow_models=50, report_name="lira_example2_report")
    attack_obj2 = LIRAAttack(args2)
    attack_obj2.attack(dataset, target_model)
    output2 = attack_obj2.make_report()  # also makes .pdf and .json files
    _ = output2["attack_metrics"][0]


def test_main():
    """test invocation via command line"""

    # option 1
    testargs = ["prog", "run-example"]
    with patch.object(sys, "argv", testargs):
        likelihood_attack.main()

    # option 2
    testargs = ["prog", "run-attack", "--j", "lrconfig.json"]
    with patch.object(sys, "argv", testargs):
        likelihood_attack.main()
