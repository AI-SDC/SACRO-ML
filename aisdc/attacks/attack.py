"""attack.py - base class for an attack object"""

import json

import sklearn

from aisdc.attacks.dataset import Data


class Attack:
    """Base (abstract) class to represent an attack"""

    def attack(self, dataset: Data, target_model: sklearn.base.BaseEstimator) -> None:
        """Method to run an attack"""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ConfigFile:  # pylint: disable=too-few-public-methods
    """ConfigFile class to load parameters from json configuration file"""

    def __init__(self, config_filename):
        self.config_filename = config_filename

    def load_config_file_into_dict(self, attack_args_dict: dict) -> None:
        """Reads a configuration file and loads it into a dictionary object"""
        with open(self.config_filename, encoding="utf-8") as f:
            config = json.loads(f.read())
        for _, k in enumerate(config):
            attack_args_dict[k] = config[k]
