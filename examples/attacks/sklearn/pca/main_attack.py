"""Example of how to run attacks on a model saved with the Target wrapper."""

import logging

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.target import Target
from sacroml.attacks.threshold_attack import ThresholdAttack

output_dir: str = "output_pca"
target_dir: str = "target_pca"

if __name__ == "__main__":
    logging.info("Loading Target object from '%s'", target_dir)
    target = Target()
    target.load(target_dir)

    logging.info("Running Threshold attack")
    attack = ThresholdAttack(output_dir=output_dir)
    attack.attack(target)

    logging.info("Running LiRA attack")
    attack = LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    attack.attack(target)

    logging.info("Report available in directory: '%s'", output_dir)
