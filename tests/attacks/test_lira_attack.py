"""Test_lira_attack.py
Copyright (C) Jim Smith2022  <james.smith@uwe.ac.uk>.
"""

# pylint: disable = duplicate-code

from __future__ import annotations

import logging
import os
import shutil
import sys
from unittest import TestCase

# import json
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks import likelihood_attack
from aisdc.attacks.likelihood_attack import DummyClassifier, LIRAAttack
from aisdc.attacks.target import Target

N_SHADOW_MODELS = 20

logger = logging.getLogger(__file__)


def clean_up(name):
    """Removes unwanted files or directory."""
    if os.path.exists(name):
        if os.path.isfile(name):
            os.remove(name)
        elif os.path.isdir(name):
            shutil.rmtree(name)
        logger.info("Removed %s", name)


class TestDummyClassifier(TestCase):
    """Test the dummy classifier class."""

    @classmethod
    def setUpClass(cls):
        """Create a dummy classifier object."""
        cls.dummy = DummyClassifier()
        cls.X = np.array([[0.2, 0.8], [0.7, 0.3]])

    def test_predict_proba(self):
        """Test the predict_proba method."""
        pred_probs = self.dummy.predict_proba(self.X)
        assert np.array_equal(pred_probs, self.X)

    def test_predict(self):
        """Test the predict method."""
        expected_output = np.array([1, 0])
        pred = self.dummy.predict(self.X)
        assert np.array_equal(pred, expected_output)


class TestLiraAttack(TestCase):
    """Test the LIRA attack code."""

    @classmethod
    def setUpClass(cls):
        """Setup the common things for the class."""
        logger.info("Setting up test class")
        X, y = load_breast_cancer(return_X_y=True, as_frame=False)
        cls.train_X, cls.test_X, cls.train_y, cls.test_y = train_test_split(
            X, y, test_size=0.3
        )
        cls.target_model = RandomForestClassifier(
            n_estimators=100, min_samples_split=2, min_samples_leaf=1
        )
        cls.target_model.fit(cls.train_X, cls.train_y)
        cls.target = Target(cls.target_model)
        cls.target.add_processed_data(cls.train_X, cls.train_y, cls.test_X, cls.test_y)
        cls.target.save(path="test_lira_target")

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
        """Tests the lira code two ways."""
        attack_obj = LIRAAttack(
            n_shadow_models=N_SHADOW_MODELS,
            output_dir="test_output_lira",
            attack_config_json_file_name=os.path.join("tests", "lrconfig.json"),
        )
        attack_obj.setup_example_data()
        attack_obj.attack_from_config()
        attack_obj.example()

        attack_obj2 = LIRAAttack(
            n_shadow_models=N_SHADOW_MODELS,
            output_dir="test_output_lira",
            report_name="lira_example1_report",
        )
        attack_obj2.attack(self.target)
        output2 = attack_obj2.make_report()  # also makes .pdf and .json files
        n_shadow_models_trained = output2["attack_experiment_logger"][
            "attack_instance_logger"
        ]["instance_0"]["n_shadow_models_trained"]
        n_shadow_models = output2["metadata"]["experiment_details"]["n_shadow_models"]
        assert n_shadow_models_trained == n_shadow_models

    def test_check_and_update_dataset(self):
        """Test the code that removes items from test set with classes
        not present in training set.
        """

        attack_obj = LIRAAttack(n_shadow_models=N_SHADOW_MODELS)

        # now make test[0] have a  class not present in training set#
        local_test_y = np.copy(self.test_y)
        local_test_y[0] = 5
        local_target = Target(self.target_model)
        local_target.add_processed_data(
            self.train_X, self.train_y, self.test_X, local_test_y
        )
        unique_classes_pre = set(local_test_y)
        n_test_examples_pre = len(local_test_y)
        local_target = (
            attack_obj._check_and_update_dataset(  # pylint:disable=protected-access
                local_target
            )
        )

        unique_classes_post = set(local_target.y_test)
        n_test_examples_post = len(local_target.y_test)

        self.assertNotEqual(local_target.y_test[0], 5)
        self.assertEqual(n_test_examples_pre - n_test_examples_post, 1)
        class_diff = unique_classes_pre - unique_classes_post  # set diff
        self.assertSetEqual(class_diff, {5})

    def test_main_example(self):
        """Test command line example."""
        testargs = [
            "prog",
            "run-example",
            "--output-dir",
            "test_output_lira",
            "--report-name",
            "commandline_lira_example2_report",
        ]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

    def test_main_config(self):
        """Test command line with a config file."""
        testargs = [
            "prog",
            "run-attack",
            "-j",
            os.path.join("tests", "lrconfig.json"),
            "--output-dir",
            "test_output_lira",
            "--report-name",
            "commandline_lira_example1_report",
        ]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

    def test_main_from_configfile(self):
        """Test command line with a config file."""
        testargs = [
            "prog",
            "run-attack-from-configfile",
            "-j",
            os.path.join("tests", "lrconfig_cmd.json"),
            "-t",
            "test_lira_target",
        ]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

    def test_main_example_data(self):
        """Test command line example data creation."""
        testargs = [
            "prog",
            "setup-example-data",
        ]  # , "--j", os.path.join("tests","lrconfig.json")]
        with patch.object(sys, "argv", testargs):
            likelihood_attack.main()

    def test_lira_attack_failfast_example(self):
        """Tests the lira code two ways."""
        attack_obj = LIRAAttack(
            n_shadow_models=N_SHADOW_MODELS,
            output_dir="test_output_lira",
            attack_config_json_file_name=os.path.join("tests", "lrconfig.json"),
            shadow_models_fail_fast=True,
            n_shadow_rows_confidences_min=10,
        )
        attack_obj.setup_example_data()
        attack_obj.attack_from_config()
        attack_obj.example()

    def test_lira_attack_failfast_from_scratch1(self):
        """Test by training a model from scratch."""
        attack_obj = LIRAAttack(
            n_shadow_models=N_SHADOW_MODELS,
            output_dir="test_output_lira",
            report_name="lira_example2_failfast_report",
            attack_config_json_file_name=os.path.join("tests", "lrconfig.json"),
            shadow_models_fail_fast=True,
            n_shadow_rows_confidences_min=10,
        )
        attack_obj.attack(self.target)
        output = attack_obj.make_report()  # also makes .pdf and .json files
        n_shadow_models_trained = output["attack_experiment_logger"][
            "attack_instance_logger"
        ]["instance_0"]["n_shadow_models_trained"]
        n_shadow_models = output["metadata"]["experiment_details"]["n_shadow_models"]
        assert n_shadow_models_trained == n_shadow_models

    def test_lira_attack_failfast_from_scratch2(self):
        """Test by training a model from scratch."""
        attack_obj = LIRAAttack(
            n_shadow_models=150,
            output_dir="test_output_lira",
            report_name="lira_example3_failfast_report",
            attack_config_json_file_name=os.path.join("tests", "lrconfig.json"),
            shadow_models_fail_fast=True,
            n_shadow_rows_confidences_min=10,
        )
        attack_obj.attack(self.target)
        output = attack_obj.make_report()  # also makes .pdf and .json files
        n_shadow_models_trained = output["attack_experiment_logger"][
            "attack_instance_logger"
        ]["instance_0"]["n_shadow_models_trained"]
        n_shadow_models = output["metadata"]["experiment_details"]["n_shadow_models"]
        assert n_shadow_models_trained < n_shadow_models

    @classmethod
    def tearDownClass(cls):
        """Cleans up various files made during the tests."""
        names = [
            "test_output_lira",
            "output_lira",
            "outputs_lira",
            "test_lira_target",
            "test_preds.csv",
            "config.json",
            "train_preds.csv",
            "test_data.csv",
            "train_data.csv",
            "test_lira_target",
        ]
        for name in names:
            clean_up(name)
