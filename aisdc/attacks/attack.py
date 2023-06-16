"""attack.py - base class for an attack object"""

import json

from aisdc.attacks.target import Target


class Attack:
    """Base (abstract) class to represent an attack"""

    def __init__(self, **kwargs):
        self.attack_config_json_file_name = None

    def attack(self, target: Target) -> None:
        """Method to run an attack"""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def update_params_from_config_file(self) -> None:
        """Reads a configuration file and loads it into a dictionary object"""
        with open(self.attack_config_json_file_name, encoding="utf-8") as f:
            config = json.loads(f.read())
        for key, value in config.items():
            setattr(self, key, value)


def load_config_file_into_dict(config_filename: str, attack_args_dict: dict) -> None:
    """Reads a configuration file and loads it into a dictionary object"""
    with open(config_filename, encoding="utf-8") as f:
        config = json.loads(f.read())
    for _, k in enumerate(config):
        attack_args_dict[k] = config[k]
