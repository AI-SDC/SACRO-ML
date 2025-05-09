"""Example running an attribute inference attack.

The steps are as follows:

1. Researcher trains their model, e.g., `train_rf_nursery.py`
2. Researcher and/or TRE runs the attacks
     1. The TRE calls the attack code.
     2. The TRE computes and inspects attack metrics.
"""

import logging

from sacroml.attacks import attribute_attack
from sacroml.attacks.target import Target

output_dir = "outputs_aia"
target_dir = "target_rf_nursery"

if __name__ == "__main__":
    logging.info("Loading Target object from '%s'", target_dir)
    target = Target()
    target.load(target_dir)

    logging.info("Creating attribute inference attack")
    attack_obj = attribute_attack.AttributeAttack(n_cpu=8, output_dir=output_dir)

    logging.info("Running attribute inference attack")
    output = attack_obj.attack(target)

    logging.info("Accessing attack metrics and metadata")
    output = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    logging.info(attribute_attack.report_categorical(output))
    logging.info(attribute_attack.report_quantitative(output))

    logging.info("Report available in directory: '%s'", output_dir)
