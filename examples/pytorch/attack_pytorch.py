"""Run attacks on an example Pytorch classifier."""

import logging

from sacroml.attacks import (
    attribute_attack,
    likelihood_attack,
    structural_attack,
    worst_case_attack,
)
from sacroml.attacks.target import Target

output_dir = "output_pytorch"
target_dir = "target_pytorch"


if __name__ == "__main__":
    logging.info("Loading Target object from '%s'", target_dir)
    target = Target()
    target.load(target_dir)

    logging.info("Running attacks...")

    attack = worst_case_attack.WorstCaseAttack(n_reps=10, output_dir=output_dir)
    output = attack.attack(target)

    attack = structural_attack.StructuralAttack(output_dir=output_dir)
    output = attack.attack(target)

    attack = attribute_attack.AttributeAttack(n_cpu=2, output_dir=output_dir)
    output = attack.attack(target)

    attack = likelihood_attack.LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    output = attack.attack(target)
