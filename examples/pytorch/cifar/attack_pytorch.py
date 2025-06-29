"""Run attacks on an example Pytorch classifier."""

import logging

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack

output_dir = "output_pytorch"
target_dir = "target_pytorch"


if __name__ == "__main__":
    logging.info("Loading Target object from '%s'", target_dir)
    target = Target()
    target.load(target_dir)

    logging.info("Running attacks...")

    attack = WorstCaseAttack(n_reps=10, output_dir=output_dir)
    output = attack.attack(target)

    attack = LIRAAttack(n_shadow_models=20, output_dir=output_dir)
    output = attack.attack(target)
