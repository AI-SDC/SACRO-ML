"""
Example demonstrating the attribute inference attacks.

Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.attribute_inference_example
"""

from __future__ import annotations

import json
import os
import sys
import unittest

from sklearn.ensemble import RandomForestClassifier

from aisdc.attacks import attribute_attack
from aisdc.attacks.attribute_attack import (
    _get_bounds_risk,
    _infer_categorical,
    _unique_max,
)

from ..common import clean, get_target

# pylint: disable = duplicate-code


class TestAttributeInferenceAttack(unittest.TestCase):
    """Class which tests the AttributeInferenceAttack module."""

    def _common_setup(self):
        """Basic commands to get ready to test some code."""
        model = RandomForestClassifier(bootstrap=False)
        target = get_target(model)
        model.fit(target.x_train, target.y_train)
        attack_obj = attribute_attack.AttributeAttack(n_cpu=7, report_name="aia_report")
        return target, attack_obj

    def test_attack_args(self):
        """Tests methods in the attack_args class."""
        _, attack_obj = self._common_setup()
        attack_obj.__dict__["newkey"] = True
        thedict = attack_obj.__dict__

        self.assertTrue(thedict["newkey"])

    def test_unique_max(self):
        """Tests the _unique_max helper function."""
        has_unique = (0.3, 0.5, 0.2)
        no_unique = (0.5, 0.5)
        self.assertTrue(_unique_max(has_unique, 0.0))
        self.assertFalse(_unique_max(has_unique, 0.6))
        self.assertFalse(_unique_max(no_unique, 0.0))

    def test_categorical_via_modified_attack_brute_force(self):
        """Test lots of functionality for categoricals
        using code from brute_force but without multiprocessing.
        """
        target, _ = self._common_setup()

        threshold = 0
        feature = 0
        # make predictions
        t_low = _infer_categorical(target, feature, threshold)
        t_low_correct = t_low["train"][0]
        t_low_total = t_low["train"][1]
        t_low_train_samples = t_low["train"][4]

        # Check the number of samples in the dataset
        self.assertEqual(len(target.x_train), t_low_train_samples)

        # Check that all samples are correct for this threshold
        self.assertEqual(t_low_correct, t_low_total)

        # or don't because threshold is too high
        threshold = 999
        t_high = _infer_categorical(target, feature, threshold)
        t_high_correct = t_high["train"][0]
        t_high_train_samples = t_high["train"][4]

        self.assertEqual(len(target.x_train), t_high_train_samples)
        self.assertEqual(0, t_high_correct)

    def test_continuous_via_modified_bounds_risk(self):
        """Tests a lot of the code for continuous variables
        via a copy of the _get_bounds_risk()
        modified not to use multiprocessing.
        """
        target, _ = self._common_setup()
        returned = _get_bounds_risk(
            target.model, "dummy", 8, target.x_train, target.x_test
        )

        # Check the number of parameters returned
        self.assertEqual(3, len(returned.keys()))

        # Check the value of the returned parameters
        self.assertEqual(0.0, returned["train"])
        self.assertEqual(0.0, returned["test"])
        clean("output_attribute")

    # test below covers a lot of the plotting etc.
    def test_AIA_on_nursery(self):
        """Tests running AIA on the nursery data
        with an added continuous feature.
        """
        target, attack_obj = self._common_setup()
        attack_obj.attack(target)

        output = attack_obj.make_report()
        keys = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ].keys()

        self.assertIn("categorical", keys)

    def test_AIA_on_nursery_from_cmd(self):
        """Tests running AIA on the nursery data
        with an added continuous feature.
        """
        target, _ = self._common_setup()
        target.save(path="tests/test_aia_target")

        config = {
            "n_cpu": 7,
            "report_name": "commandline_aia_exampl1_report",
        }
        with open(
            os.path.join("tests", "test_config_aia_cmd.json"), "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(config))

        cmd_json = os.path.join("tests", "test_config_aia_cmd.json")
        aia_target = os.path.join("tests", "test_aia_target")
        os.system(
            f"{sys.executable} -m aisdc.attacks.attribute_attack run-attack-from-configfile "
            f"--attack-config-json-file-name {cmd_json} "
            f"--attack-target-folder-path {aia_target} "
        )

    def test_cleanup(self):
        """Tidies up any files created."""
        files_made = (
            "delete-me.json",
            "aia_example.json",
            "aia_example.pdf",
            "aia_report_cat_frac.png",
            "aia_report_cat_risk.png",
            "aia_report_quant_risk.png",
            "aia_report.pdf",
            "aia_report.json",
            "aia_attack_from_configfile.json",
            "test_attribute_attack.json",
            os.path.join("tests", "test_config_aia_cmd.json"),
            os.path.join("tests", "test_aia_target/"),
            "output_attribute",
        )
        for fname in files_made:
            clean(fname)
