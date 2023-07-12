"""
Example demonstrating the attribute inference attacks.

Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.attribute_inference_example

"""
import json
import os
import sys

# ignore unused imports because it depends on whether data file is present
from sklearn.datasets import fetch_openml  # pylint:disable=unused-import
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (  # pylint:disable=unused-import
    LabelEncoder,
    OneHotEncoder,
)

from aisdc.attacks.multiple_attacks import ConfigFile  # pylint: disable = import-error
from aisdc.attacks.multiple_attacks import (
    MultipleAttacks,  # pylint: disable = import-error
)
from aisdc.attacks.attack_report_formatter import GenerateJSONModule
from aisdc.attacks.attribute_attack import (
    _get_bounds_risk,
    _infer_categorical,
    _unique_max,
)
from tests.test_attacks_via_safemodel import get_target

# pylint: disable = duplicate-code


def cleanup_file(name: str):
    """removes unwanted files or directory"""
    if os.path.exists(name) and os.path.isfile(name):  # h5
        os.remove(name)


def common_setup():
    """basic commands to get ready to test some code"""
    model = RandomForestClassifier(bootstrap=False)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    config = 
    attack_obj = multiple_attacks.MultipleAttacks(
        config_filename="test_single_config.json",
        output_filename="test_single_output.json",
    )
    return target, attack_obj

def create_single_config_file():
    configfile_obj = ConfigFile(
        filename="single_config.json",
    )

    # Example 1: Adding a configuration dictionary to the JSON file
    config = {
        "n_reps": 10,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "report_name": "worstcase_example1_report",
    }
    configfile_obj.add_config(config, "worst_case")

    # Example 2: Adding a configuration dictionary to the JSON file
    config = {
        "n_reps": 20,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "report_name": "worstcase_example2_report",
    }
    configfile_obj.add_config(config, "worst_case")

    # Example 3: Adding a configuration dictionary to the JSON file
    config = {
        "n_reps": 10,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "report_name": "worstcase_example3_report",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "attack_metric_success_name": "P_HIGHER_AUC",
        "attack_metric_success_thresh": 0.05,
        "attack_metric_success_comp_type": "lte",
        "attack_metric_success_count_thresh": 2,
        "attack_fail_fast": True,
    }
    configfile_obj.add_config(config, "worst_case")

    # Example 4: Adding a configuration dictionary to the JSON file
    config = {
        "n_shadow_models": 100,
        "report_name": "lira_example1_report",
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }
    configfile_obj.add_config(config, "lira")

    # Example 5: Adding a configuration dictionary to the JSON file
    config = {
        "n_shadow_models": 150,
        "report_name": "lira_example2_report",
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

    # Example 5: Adding an existing configuration file to a single JSON configuration file
    config = {
        "n_shadow_models": 120,
        "report_name": "lira_example3_report",
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
    configfile_obj.add_config("lira_config.json", "lira")

    # Example 6: Adding a configuration dictionary to the JSON file
    config = {
        "n_cpu": 2,
        "report_name": "aia_exampl1_report",
    }
    configfile_obj.add_config(config, "attribute")
    return configfile_obj

def test_configfile_number():
    configfile_obj = create_single_config_file()
    _, n = configfile_obj.read_config_file()
    assert n == 7

def test_programmatic_multiple_attacks():
    target, attack_obj = common_setup()
    attack_obj.attack(target)


def test_cmd_multiple_attacks():
    """tests running AIA on the nursery data
    with an added continuous feature"""
    target, _ = common_setup()
    target.save(path="tests/test_multiple_target")

    config = {
        "n_cpu": 7,
        "report_name": "commandline_aia_exampl1_report",
    }
    with open("tests/test_config_aia_cmd.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(config))

    os.system(
        f"{sys.executable} -m aisdc.attacks.attribute_attack run-attack-from-configfile "
        "--attack-config-json-file-name tests/test_single_config_cmd.json "
        "--attack-target-folder-path tests/test_multiple_target "
        "--attack-output-json-file-name tests/test_single_output_cmd.json "
    )


def test_cleanup():
    """tidies up any files created"""
    files_made = (
        "test_lira_config.json",
        "test_single_config.json",
        "test_single_output.json",
        "worstcase_example1_report.pdf",
        "worstcase_example2_report.pdf",
        "worstcase_example3_report.pdf",
        "lira_example1_report.pdf",
        "lira_example2_report.pdf",
        "lira_example3_report.pdf",        
        "aia_exampl1_report.pdf",
        "tests/test_single_config_cmd.json",
        "tests/test_single_output_cmd.json",
    )
    for fname in files_made:
        cleanup_file(fname)