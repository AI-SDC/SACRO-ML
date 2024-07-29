"""Prompt to generate valid attack config."""

from __future__ import annotations

import logging

import yaml

from sacroml.attacks import factory
from sacroml.config import utils


def _get_defaults(name: str) -> dict:
    """Return an attack parameters and their defaults."""
    attack = factory.create_attack(name)
    return attack.get_params()


def _prompt_for_params(params: dict) -> None:
    """Prompt user to change parameter values."""
    for key, val in params.items():
        print(f"The current value for '{key}' is {val}.")
        if utils.get_bool("Do you want to change it?"):
            while True:
                new_val = input(f"Enter new value for '{key}': ").strip()
                try:
                    params[key] = type(val)(new_val)
                    break
                except ValueError:
                    print(f"Please enter a value of type {type(val).__name__}.")


def _get_attack(name: str) -> dict:
    """Get an attack configuration."""
    params: dict = _get_defaults(name)
    if not utils.get_bool("Use all defaults?"):
        _prompt_for_params(params)
    return {"name": name, "params": params}


def _prompt_for_attacks() -> list[dict]:
    """Prompt user for individual attack configurations."""
    attacks: list[dict] = []
    names: list[str] = list(factory.registry.keys())
    while utils.get_bool("Would you like to add an attack?"):
        while True:
            print(f"Attacks available: {', '.join(names)}")
            name: str = input("Which attack?: ")
            if name in names:
                attack = _get_attack(name)
                attacks.append(attack)
                print(f"{name} attack added.")
                break
            print("Please enter one of the available attacks.")
    return attacks


def _default_config() -> list[dict]:
    """Return a default configuration with all attacks."""
    attacks: list[dict] = []
    names: list[str] = list(factory.registry.keys())
    for name in names:
        params: dict = _get_defaults(name)
        attack: dict = {"name": name, "params": params}
        attacks.append(attack)
    return attacks


def prompt_for_attack() -> None:
    """Prompt user for information to generate attack config."""
    logging.disable(logging.ERROR)  # suppress info/warnings

    # get attack configuration
    if utils.get_bool("Generate default config with all attacks?"):
        attacks = _default_config()
    else:
        attacks = _prompt_for_attacks()

    # write to file
    filename: str = "attack.yaml"
    with open(filename, "w", encoding="utf-8") as fp:
        yaml.dump({"attacks": attacks}, fp)
    print(f"{filename} has been generated.")
