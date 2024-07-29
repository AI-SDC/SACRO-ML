"""Factory for running attacks."""

import logging

import yaml

from sacroml.attacks.attribute_attack import AttributeAttack
from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.structural_attack import StructuralAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

registry: dict = {
    "attribute": AttributeAttack,
    "lira": LIRAAttack,
    "structural": StructuralAttack,
    "worstcase": WorstCaseAttack,
}


def create_attack(attack_name: str, **kwargs: dict) -> None:
    """Instantiate an attack."""
    if attack_name in registry:
        return registry[attack_name](**kwargs)
    raise ValueError(f"Unknown Attack: {attack_name}")


def attack(target: Target, attack_name: str, **kwargs: dict) -> dict:
    """Create and execute an attack on a target."""
    attack_obj = create_attack(attack_name, **kwargs)
    return attack_obj.attack(target)


def run_attacks(target_dir: str, attack_filename: str) -> None:
    """Run attacks given a target and attack configuration.

    Parameters
    ----------
    target_dir : str
        Name of a directory containing target.yaml.
    attack_filename : str
        Name of a YAML file containing an attack configuration.
    """
    logger.info("Preparing Target")
    target = Target()
    target.load(target_dir)

    logger.info("Preparing Attacks")
    with open(attack_filename, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Running Attacks")

    for attack_cfg in config["attacks"]:
        name = attack_cfg["name"]
        params = attack_cfg["params"]
        attack(target=target, attack_name=name, **params)

    logger.info("Finished running attacks")
