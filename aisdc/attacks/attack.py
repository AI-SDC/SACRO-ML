"""attack.py - base class for an attack object"""

import json

from aisdc.attacks.target import Target


class Attack:
    """Base (abstract) class to represent an attack"""

    def __init__(self):
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

    def _exclude_keys_from_dict(self, keys_to_exclude: list) -> dict:
        """Exclude keys from a given dictionary"""
        return {k: self.__dict__[k] for k in self.__dict__.keys() if k not in keys_to_exclude}
