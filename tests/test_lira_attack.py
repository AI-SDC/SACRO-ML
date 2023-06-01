"""test_lira_attack.py
Copyright (C) Jim Smith2022  <james.smith@uwe.ac.uk>
"""
# pylint: disable = duplicate-code

import logging
import os
import sys
from unittest import TestCase

# import json
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks import likelihood_attack
from aisdc.attacks.dataset import Data  # pylint: disable = import-error
from aisdc.attacks.likelihood_attack import (  # pylint: disable = import-error
    DummyClassifier,
    LIRAAttack,
    LIRAAttackArgs,
)

N_SHADOW_MODELS = 20

logger = logging.getLogger(__file__)


def clean_up(name):
    """removes unwanted files or directory"""
    if os.path.exists(name) and os.path.isfile(name):
        os.remove(name)
        logger.info("Removed %s", name)


class TestDummyClassifier(TestCase):
    """Test the dummy classifier class"""

    @classmethod
    def setUpClass(cls):
        """Create a dummy classifier object"""
        cls.dummy = DummyClassifier()
        cls.X = np.array([[0.2, 0.8], [0.7, 0.3]])

    def test_predict_proba(self):
        """Test the predict_proba method"""
        pred_probs = self.dummy.predict_proba(self.X)
        assert np.array_equal(pred_probs, self.X)

    def test_predict(self):
        """Test the predict method"""
        expected_output = np.array([1, 0])
        pred = self.dummy.predict(self.X)
        assert np.array_equal(pred, expected_output)


class TestLiraAttack(TestCase):
    """Test the LIRA attack code"""

    @classmethod
    def setUpClass(cls):
        """Setup the common things for the class"""
        logger.info("Setting up test class")
        X, y = load_breast_cancer(return_X_y=True, as_frame=False)
        cls.train_X, cls.test_X, cls.train_y, cls.test_y = train_test_split(
            X, y, test_size=0.3
        )
        cls.dataset = Data()
        cls.dataset.add_processed_data(cls.train_X, cls.train_y, cls.test_X, cls.test_y)
        cls.target_model = RandomForestClassifier(
            n_estimators=100, min_samples_split=2, min_samples_leaf=1
        )
        cls.target_model.fit(cls.train_X, cls.train_y)

        # Dump training and test data to csv
        np.savetxt(
            "train_data.csv",
            np.hstack((cls.train_X, cls.train_y[:, None])),
            delimiter=",",
        )
        np.savetxt(
            "test_data.csv", np.hstack((cls.test_X, cls.test_y[:, None])), delimiter=","
        )
        # dump the training and test predictions into files
        np.savetxt(
            "train_preds.csv",
            cls.target_model.predict_proba(cls.train_X),
            delimiter=",",
        )
        np.savetxt(
            "test_preds.csv", cls.target_model.predict_proba(cls.test_X), delimiter=","
        )

    def test_lira_attack(self):
        """tests the lira code two ways"""
        args = LIRAAttackArgs(
            n_shadow_models=N_SHADOW_MODELS,
            report_name="lira_example_report",
            attack_config_json_file_name="tests/lrconfig.json",
        )
        attack_obj = LIRAAttack(args)
        attack_obj.setup_example_data()
        attack_obj.attack_from_config()
        attack_obj.example()

        args2 = LIRAAttackArgs(
            n_shadow_models=N_SHADOW_MODELS, report_name="lira_example2_report"
        )
        attack_obj2 = LIRAAttack(args2)
        attack_obj2.attack(self.dataset, self.target_model)
        output2 = attack_obj2.make_report()  # also makes .pdf and .json files
        n_shadow_models_trained = output2["attack_experiment_logger"][
            "attack_instance_logger"
        ]["instance_0"]["n_shadow_models_trained"]
        n_shadow_models = output2["metadata"]["experiment_details"]["n_shadow_models"]
        assert n_shadow_models_trained == n_shadow_models

    def test_check_and_update_dataset(self):
        """test the code that removes items from test set with classes
        not present in training set"""
        args = LIRAAttackArgs(
            n_shadow_models=N_SHADOW_MODELS, report_name="lira_example_report"
        )
        attack_obj = LIRAAttack(args)

        # now make test[0] have a  class not present in training set#
        local_test_y = np.copy(self.test_y)
        local_test_y[0] = 5
        local_dataset = Data()
        local_dataset.add_processed_data(
            self.train_X, self.train_y, self.test_X, local_test_y
        )
        unique_classes_pre = set(local_test_y)
        n_test_examples_pre = len(local_test_y)
        local_dataset = (
            attack_obj._check_and_update_dataset(  # pylint:disable=protected-access
                local_dataset, self.target_model
            )
        )

        unique_classes_post = set(local_dataset.y_test)
        n_test_examples_post = len(local_dataset.y_test)

        self.assertNotEqual(local_dataset.y_test[0], 5)
        self.assertEqual(n_test_examples_pre - n_test_examples_post, 1)
        class_diff = unique_classes_pre - unique_classes_post  # set diff
        self.assertSetEqual(class_diff, {5})

    def test_main_example(self):
        """test command line example"""
        testargs = ["prog", "run-example"]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

    def test_main_config(self):
        """test command line with a config file"""
        testargs = ["prog", "run-attack", "-j", "tests/lrconfig.json"]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

    def test_main_from_configfile(self):
        """test command line with a config file"""
        testargs = [
            "prog",
            "run-attack-from-configfile",
            "-j",
            "tests/lrconfig_cmd.json",
        ]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

    def test_main_example_data(self):
        """test command line example data creation"""
        testargs = ["prog", "setup-example-data"]  # , "--j", "tests/lrconfig.json"]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

    def test_lira_attack_failfast_example(self):
        """tests the lira code two ways"""
        args = LIRAAttackArgs(
            n_shadow_models=N_SHADOW_MODELS,
            report_name="lira_example_report",
            attack_config_json_file_name="tests/lrconfig.json",
            shadow_models_fail_fast=True,
            n_shadow_rows_confidences_min=10,
        )
        attack_obj = LIRAAttack(args)
        attack_obj.setup_example_data()
        attack_obj.attack_from_config()
        attack_obj.example()

    def test_lira_attack_failfast_from_scratch(self):
        """Test by training a model from scratch"""
        args = LIRAAttackArgs(
            n_shadow_models=N_SHADOW_MODELS,
            report_name="lira_example3_failfast_report",
            attack_config_json_file_name="tests/lrconfig.json",
            shadow_models_fail_fast=True,
            n_shadow_rows_confidences_min=10,
        )
        attack_obj = LIRAAttack(args)
        attack_obj.attack(self.dataset, self.target_model)
        output = attack_obj.make_report()  # also makes .pdf and .json files
        n_shadow_models_trained = output["attack_experiment_logger"][
            "attack_instance_logger"
        ]["instance_0"]["n_shadow_models_trained"]
        n_shadow_models = output["metadata"]["experiment_details"]["n_shadow_models"]
        assert n_shadow_models_trained == n_shadow_models

    @classmethod
    def tearDownClass(cls):
        """cleans up various files made during the tests"""
        names = [
            "lr_report.pdf",
            "log_roc.png",
            "lr_report.json",
            "lira_example2_report.json",
            "lira_example2_report.pdf",
            "test_preds.csv",
            "config.json",
            "train_preds.csv",
            "test_data.csv",
            "train_data.csv",
        ]
        for name in names:
            clean_up(name)
