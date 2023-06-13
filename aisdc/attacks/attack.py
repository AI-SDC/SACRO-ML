"""attack.py - base class for an attack object"""

import json

import multiprocess as mp
from sklearn.ensemble import RandomForestClassifier

from aisdc.attacks.target import Target


class Attack:
    """Base (abstract) class to represent an attack"""

    def attack(self, target: Target) -> None:
        """Method to run an attack"""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


def load_config_file_into_dict(config_filename: str, attack_args_dict: dict) -> None:
    """Reads a configuration file and loads it into a dictionary object"""
    with open(config_filename, encoding="utf-8") as f:
        config = json.loads(f.read())
    for _, k in enumerate(config):
        attack_args_dict[k] = config[k]


def load_default_worstcase_dict(args: dict) -> None:
    """Initialise dictionary items with default values for worst case attack"""
    args["n_reps"] = 10
    args["p_thresh"] = 0.05
    args["n_dummy_reps"] = 1
    args["train_beta"] = 2
    args["test_beta"] = 2
    args["test_prop"] = 0.3
    args["n_rows_in"] = 1000
    args["n_rows_out"] = 1000
    args["training_preds_filename"] = None
    args["test_preds_filename"] = None
    args["report_name"] = None
    args["include_model_correct_feature"] = False
    args["sort_probs"] = True
    args["mia_attack_model"] = RandomForestClassifier
    args["mia_attack_model_hyp"] = {
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_depth": 5,
    }
    args["attack_metric_success_name"] = "P_HIGHER_AUC"
    args["attack_metric_success_thresh"] = 0.05
    args["attack_metric_success_comp_type"] = "lte"
    args["attack_metric_success_count_thresh"] = 5
    args["attack_fail_fast"] = False
    args["attack_config_json_file_name"] = None


def load_default_lira_dict(args: dict) -> None:
    """Initialise dictionary items with default values for worst case attack"""
    args["n_shadow_models"] = 100
    args["n_shadow_rows_confidences_min"] = 10
    args["p_thresh"] = 0.05
    args["report_name"] = None
    args["training_data_filename"] = None
    args["test_data_filename"] = None
    args["training_preds_filename"] = None
    args["test_preds_filename"] = None
    args["target_model"] = None
    args["target_model_hyp"] = None
    args["attack_config_json_file_name"] = None
    args["shadow_models_fail_fast"] = False


def load_default_attribute_dict(args: dict) -> None:
    """Initialise dictionary items with default values for attribute inference attack"""
    args["report_name"] = None
    args["n_cpu"] = max(1, mp.cpu_count() - 1)
