"""Test multiple attacks (MIA and AIA) using a single configuration file."""

from __future__ import annotations

import json
import os
import sys

import pytest
from sklearn.ensemble import RandomForestClassifier

from aisdc.attacks.multiple_attacks import ConfigFile, MultipleAttacks


def pytest_generate_tests(metafunc):
    """Generate target model for testing."""
    if "get_target" in metafunc.fixturenames:
        metafunc.parametrize(
            "get_target", [RandomForestClassifier(bootstrap=False)], indirect=True
        )


@pytest.fixture(name="common_setup")
def fixture_common_setup(get_target):
    """Get ready to test some code."""
    target = get_target
    target.model.fit(target.X_train, target.y_train)
    attack_obj = MultipleAttacks(config_filename="test_single_config.json")
    return target, attack_obj


def create_single_config_file():
    """Create single config file using multiple attack configuration."""
    configfile_obj = ConfigFile(filename="test_single_config.json")

    # Example 1: Add 3 different worst case configuration dictionaries to JSON
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

    # Add 2 different lira attack configuration dictionaries to JSON
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
    # add explicitly wrong attack name to cover codecov test
    configfile_obj.add_config(config, "lirrra")

    # Example 3: Add a lira JSON configuration file to a configuration file
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

    # Example 4: Add an attribute configuration dictionary
    # from an existing configuration file to JSON
    config = {
        "n_cpu": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
    }
    configfile_obj.add_config(config, "attribute")
    os.remove("test_lira_config.json")
    return configfile_obj


def test_configfile_number():
    """Test attack configurations in a configuration file."""
    configfile_obj = create_single_config_file()
    configfile_data = configfile_obj.read_config_file()
    assert len(configfile_data) == 8
    os.remove("test_single_config.json")


def test_multiple_attacks_programmatic(common_setup):
    """Test programmatically running attacks using a single config file."""
    target, attack_obj = common_setup
    _ = create_single_config_file()
    attack_obj.attack(target)
    print(attack_obj)
    os.remove("test_single_config.json")


def test_multiple_attacks_cmd(common_setup):
    """Test multiple attacks (MIA and AIA) with a continuous feature."""
    target, _ = common_setup
    target.save(path=os.path.join("tests", "test_multiple_target"))
    _ = create_single_config_file()

    multiple_target = os.path.join("tests", "test_multiple_target")
    os.system(
        f"{sys.executable} -m aisdc.attacks.multiple_attacks run-attack-from-configfile "
        "--attack-config-json-file-name test_single_config.json "
        f"--attack-target-folder-path {multiple_target} "
    )
