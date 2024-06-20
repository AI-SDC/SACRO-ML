"""Base class for an attack object."""

from __future__ import annotations

import importlib
import inspect
import os

from aisdc.attacks.target import Target


class Attack:
    """Base class to represent an attack."""

    def __init__(self, output_dir: str = "outputs", make_report: bool = True) -> None:
        """Instantiate an attack.

        Parameters
        ----------
        output_dir : str
            name of the directory where outputs are stored
        make_report : bool
            Whether to generate a JSON and PDF report.
        """
        self.output_dir: str = output_dir
        self.make_report: bool = make_report
        self.attack_metrics: dict | list = {}
        self.metadata: dict = {}
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def attack(self, target: Target) -> dict:
        """Run an attack."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return the string representation of an attack."""
        raise NotImplementedError

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Get parameter names."""
        init_signature = inspect.signature(cls.__init__)
        return [
            p.name
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

    def get_params(self) -> dict:
        """Get parameters for this attack.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out: dict = {}
        for key in self._get_param_names():
            out[key] = getattr(self, key)
        return out


def get_class_by_name(class_path: str):
    """Return a class given its name."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
