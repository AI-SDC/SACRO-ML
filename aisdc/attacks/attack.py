"""Attack.py - base class for an attack object."""

import inspect
import json

from aisdc.attacks.target import Target


class Attack:
    """Base (abstract) class to represent an attack."""

    def __init__(self):
        self.attack_config_json_file_name = None

    def attack(self, target: Target) -> None:
        """Method to run an attack."""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def _update_params_from_config_file(self) -> None:
        """Reads a configuration file and loads it into a dictionary object."""
        with open(self.attack_config_json_file_name, encoding="utf-8") as f:
            config = json.loads(f.read())
        for key, value in config.items():
            setattr(self, key, value)

    @classmethod
    def _get_param_names(cls):
        """Get parameter names."""
        init_signature = inspect.signature(cls.__init__)
        parameters = [
            p.name
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        return parameters

    def get_params(self):
        """
        Get parameters for this attack.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out: dict = {}
        for key in self._get_param_names():
            out[key] = getattr(self, key)
        return out
