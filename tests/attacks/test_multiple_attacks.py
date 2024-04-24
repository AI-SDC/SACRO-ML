"""
Test to run multiple attacks (MIA and AIA) using a single configuration file
having different configuration settings (i.e. attack type or configuration parameters).

Running
-------

Invoke this code from the root AI-SDC folder.
However to run this test file, it will be required to install pytest package
using 'pip install pytest' and then run following
python -m pytest .\tests\test_multiple_attacks.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys

# ignore unused imports because it depends on whether data file is present
from sklearn.datasets import fetch_openml  # pylint:disable=unused-import
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (  # pylint:disable=unused-import
    LabelEncoder,
    OneHotEncoder,
)

from aisdc.attacks.multiple_attacks import ConfigFile, MultipleAttacks

from ..common import get_target

# pylint: disable = duplicate-code


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
    attack_obj = MultipleAttacks(
        config_filename="test_single_config.json",
    )
    return target, attack_obj


def create_single_config_file():
    """Creates single configuration file using multiple attack configuration."""
    # instantiating a configfile object to add configurations
    configfile_obj = ConfigFile(
        filename="test_single_config.json",
    )

    # Example 1: Adding three different worst case configuration dictionaries to the JSON file
    config = {
        "n_reps": 10,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
    }
    configfile_obj.add_config(config, "worst_case")

    config = {
        "n_reps": 20,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
    }
    configfile_obj.add_config(config, "worst_case")

    config = {
        "n_reps": 10,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "attack_metric_success_name": "P_HIGHER_AUC",
        "attack_metric_success_thresh": 0.05,
        "attack_metric_success_comp_type": "lte",
        "attack_metric_success_count_thresh": 2,
        "attack_fail_fast": True,
    }
    configfile_obj.add_config(config, "worst_case")

    # Adding two different lira attack configuration dictionaries to the JSON file
    config = {
        "n_shadow_models": 100,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }
    configfile_obj.add_config(config, "lira")

    config = {
        "n_shadow_models": 150,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
        "shadow_models_fail_fast": True,
        "n_shadow_rows_confidences_min": 10,
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }
    configfile_obj.add_config(config, "lira")
    # adding explicitly wrong attack name to cover codecov test
    configfile_obj.add_config(config, "lirrra")

    # Example 3: Adding a lira JSON configuration file to a configuration file
    # having multiple attack configurations
    config = {
        "n_shadow_models": 120,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
        "shadow_models_fail_fast": True,
        "n_shadow_rows_confidences_min": 10,
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }
    with open("test_lira_config.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(config))
    configfile_obj.add_config("test_lira_config.json", "lira")

    # Example 4: Adding a attribute configuration dictionary
    # from an existing configuration file to the JSON configuration file
    config = {
        "n_cpu": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
    }
    configfile_obj.add_config(config, "attribute")
    os.remove("test_lira_config.json")
    return configfile_obj


def test_configfile_number():
    """Tests number of attack configurations in a configuration file."""
    configfile_obj = create_single_config_file()
    configfile_data = configfile_obj.read_config_file()
    assert len(configfile_data) == 8
    os.remove("test_single_config.json")


def test_multiple_attacks_programmatic():
    """Tests programmatically running attacks using a single configuration configuration file."""
    target, attack_obj = common_setup()
    _ = create_single_config_file()
    attack_obj.attack(target)
    print(attack_obj)
    os.remove("test_single_config.json")


def test_multiple_attacks_cmd():
    """Tests running multiple attacks (MIA and AIA) on the nursery data
    with an added continuous feature.
    """
    target, _ = common_setup()
    target.save(path=os.path.join("tests", "test_multiple_target"))
    _ = create_single_config_file()

    multiple_target = os.path.join("tests", "test_multiple_target")
    os.system(
        f"{sys.executable} -m aisdc.attacks.multiple_attacks run-attack-from-configfile "
        "--attack-config-json-file-name test_single_config.json "
        f"--attack-target-folder-path {multiple_target} "
    )


def test_cleanup():
    """Tidies up any files created."""
    files_made = (
        "test_single_config.json",
        "outputs_multiple_attacks",
        os.path.join("tests", "test_multiple_target"),
    )
    for fname in files_made:
        cleanup_file(fname)
