"""
Example demonstrating the attribute inference attacks.

Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.attribute_inference_example
"""
import glob
import json
import os
import shutil
import sys
import unittest

# ignore unused imports because it depends on whether data file is present
from sklearn.datasets import fetch_openml  # pylint:disable=unused-import
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (  # pylint:disable=unused-import
    LabelEncoder,
    OneHotEncoder,
)

from aisdc.attacks import attribute_attack  # pylint: disable = import-error
from aisdc.attacks.attribute_attack import (
    _get_bounds_risk,
    _infer_categorical,
    _unique_max,
)
from tests.test_attacks_via_safemodel import get_target

# pylint: disable = duplicate-code


<<<<<<< HEAD
class TestAttributeInferenceAttack(unittest.TestCase):
    """
    Class which tests the AttributeInferenceAttack module
    """

    def _cleanup_file(self, name: str):
        """removes unwanted files or directory"""
        if os.path.exists(name) and os.path.isfile(name):  # h5
            os.remove(name)
        elif os.path.exists(name) and os.path.isdir(name):  # tf
            files = glob.glob(name + "*")
            for f in files:
                os.remove(f)
            shutil.rmtree(name)

    def _common_setup(self):
        """basic commands to get ready to test some code"""
        model = RandomForestClassifier(bootstrap=False)
        target = get_target(model)
        model.fit(target.x_train, target.y_train)
        attack_obj = attribute_attack.AttributeAttack(n_cpu=7, report_name="aia_report")
        return target, attack_obj

    def test_attack_args(self):
        """tests methods in the attack_args class"""
        _, attack_obj = self._common_setup()
        attack_obj.__dict__["newkey"] = True
        thedict = attack_obj.__dict__

        self.assertTrue(thedict["newkey"])

    def test_unique_max(self):
        """tests the _unique_max helper function"""
        has_unique = (0.3, 0.5, 0.2)
        no_unique = (0.5, 0.5)
        self.assertTrue(_unique_max(has_unique, 0.0))
        self.assertFalse(_unique_max(has_unique, 0.6))
        self.assertFalse(_unique_max(no_unique, 0.0))

    def test_categorical_via_modified_attack_brute_force(self):
        """test lots of functionality for categoricals
        using code from brute_force but without multiprocessing
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
        """tests a lot of the code for continuous variables
        via a copy of the _get_bounds_risk()
        modified not to use multiprocessing
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

    # test below covers a lot of the plotting etc.
    def test_AIA_on_nursery(self):
        """tests running AIA on the nursery data
        with an added continuous feature"""
        target, attack_obj = self._common_setup()
        attack_obj.attack(target)

        output = attack_obj.make_report()
        keys = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ].keys()

        self.assertIn("categorical", keys)

    def test_AIA_on_nursery_from_cmd(self):
        """tests running AIA on the nursery data
        with an added continuous feature"""
        target, _ = self._common_setup()
        target.save(path="tests/test_aia_target")

        config = {
            "n_cpu": 7,
            "report_name": "commandline_aia_exampl1_report",
        }
        with open("tests/test_config_aia_cmd.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(config))

        os.system(
            f"{sys.executable} -m aisdc.attacks.attribute_attack run-attack-from-configfile "
            "--attack-config-json-file-name tests/test_config_aia_cmd.json "
            "--attack-target-folder-path tests/test_aia_target "
        )

        assert os.path.exists("commandline_aia_exampl1_report.pdf") is True

    def test_cleanup(self):
        """tidies up any files created"""
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
            "commandline_aia_exampl1_report.pdf",
            "tests/test_config_aia_cmd.json",
            "tests/test_aia_target/",
        )
        for fname in files_made:
            self._cleanup_file(fname)
=======
def cleanup_file(name: str):
    """Removes unwanted files or directory."""
    if os.path.exists(name):
        if os.path.isfile(name):
            os.remove(name)
        elif os.path.isdir(name):
            shutil.rmtree(name)


def common_setup():
    """Basic commands to get ready to test some code."""
    model = RandomForestClassifier(bootstrap=False)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    attack_obj = attribute_attack.AttributeAttack(
        n_cpu=7,
        output_dir="test_output_aia",
        report_name="test_attribute_attack",
    )
    return target, attack_obj


def test_attack_args():
    """Tests methods in the attack_args class."""
    _, attack_obj = common_setup()
    attack_obj.__dict__["newkey"] = True
    thedict = attack_obj.__dict__
    assert thedict["newkey"] is True


def test_unique_max():
    """Tests the _unique_max helper function."""
    has_unique = (0.3, 0.5, 0.2)
    no_unique = (0.5, 0.5)
    assert _unique_max(has_unique, 0.0) is True
    assert _unique_max(has_unique, 0.6) is False
    assert _unique_max(no_unique, 0.0) is False


def test_categorical_via_modified_attack_brute_force():
    """Test lots of functionality for categoricals
    using code from brute_force but without multiprocessing.
    """
    target, _ = common_setup()

    threshold = 0
    feature = 0
    # make predictions
    _infer_categorical(target, feature, threshold)
    # or don't because threshold is too high
    threshold = 999
    _infer_categorical(target, feature, threshold)


def test_continuous_via_modified_bounds_risk():
    """Tests a lot of the code for continuous variables
    via a copy of the _get_bounds_risk()
    modified not to use multiprocessing.
    """
    target, _ = common_setup()
    _ = _get_bounds_risk(target.model, "dummy", 8, target.x_train, target.x_test)


# test below covers a lot of the plotting etc.
def test_AIA_on_nursery():
    """Tests running AIA on the nursery data
    with an added continuous feature.
    """
    target, attack_obj = common_setup()
    attack_obj.attack(target)

    output = attack_obj.make_report()
    output = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]


def test_AIA_on_nursery_from_cmd():
    """Tests running AIA on the nursery data
    with an added continuous feature.
    """
    target, _ = common_setup()
    target.save(path="test_aia_target")

    config = {
        "n_cpu": 7,
        "output_dir": "test_output_aia",
        "report_name": "commandline_aia_exampl1_report",
    }
    with open("tests/test_config_aia_cmd.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(config))

    os.system(
        f"{sys.executable} -m aisdc.attacks.attribute_attack run-attack-from-configfile "
        "--attack-config-json-file-name tests/test_config_aia_cmd.json "
        "--attack-target-folder-path test_aia_target "
    )


def test_cleanup():
    """Tidies up any files created."""
    files_made = (
        "test_output_aia/",
        "test_aia_target/",
        "test_attribute_attack.json",
        "tests/test_config_aia_cmd.json",
    )
    for fname in files_made:
        cleanup_file(fname)
>>>>>>> development
