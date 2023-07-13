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

from aisdc.attacks.multiple_attacks import (  # pylint: disable = import-error
    ConfigFile,
    MultipleAttacks,
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
    attack_obj = MultipleAttacks(
        config_filename="test_single_config.json",
        output_filename="test_single_output.json",
    )
    return target, attack_obj


def create_single_config_file():
    """creates single configuration file using multiple attack configuration"""
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
        "report_name": "worstcase_example1_report",
    }
    configfile_obj.add_config(config, "worst_case")

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

    # Adding two different lira attack configuration dictionaries to the JSON file
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

    # Example 3: Adding a lira JSON configuration file to a configuration file
    # having multiple attack configurations
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
    configfile_obj.add_config("test_lira_config.json", "lira")

    # Example 4: Adding a attribute configuration dictionary
    # from an existing configuration file to the JSON configuration file
    config = {
        "n_cpu": 2,
        "report_name": "aia_exampl1_report",
    }
    configfile_obj.add_config(config, "attribute")
    return configfile_obj


def test_configfile_number():
    """tests number of attack configurations in a configuration file"""
    configfile_obj = create_single_config_file()
    _, n = configfile_obj.read_config_file()
    assert n == 7


def test_multiple_attacks_programmatic():
    """tests programmatically running attacks using a single configuration configuration file"""
    target, attack_obj = common_setup()
    attack_obj.attack(target)


def test_multiple_attacks_cmd():
    """tests running multiple attacks (MIA and AIA) on the nursery data
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
