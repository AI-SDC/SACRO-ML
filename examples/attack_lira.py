"""Example running a LiRA membership inference attack programmatically.

This code simulates a MIA attack providing the attacker with as much
information as possible. That is, they have a subset of rows that they _know_
were used for training. And a subset that they know were not. They also have
query access to the target model.

The attack proceeds as described in this paper:
https://arxiv.org/pdf/2112.03570.pdf

The steps are as follows:

1. Researcher trains their model, e.g., `train_rf_breast_cancer.py`
2. Researcher and/or TRE runs the attacks
     1. The TRE calls the attack code.
     2. The TRE computes and inspects attack metrics.
"""

import logging

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.target import Target

output_dir = "outputs_lira"
target_dir = "target_rf_breast_cancer"

if __name__ == "__main__":
    logging.info("Loading Target object from '%s'", target_dir)
    target = Target()
    target.load(target_dir)

    logging.info("Creating LiRA attack")
    attack_obj = LIRAAttack(n_shadow_models=100, output_dir=output_dir)

    logging.info("Running LiRA attack")
    output = attack_obj.attack(target)

    logging.info("Accessing attack metrics and metadata")
    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    metadata = output["metadata"]

    logging.info("*******************")
    logging.info("Attack metrics:")
    logging.info("*******************")
    for key, value in metrics.items():
        try:
            logging.info("%s: %s", key, str(value))
        except TypeError:
            logging.info("Cannot print %s", key)

    logging.info("*******************")
    logging.info("Global metrics")
    logging.info("*******************")
    for key, value in metadata["global_metrics"].items():
        logging.info("%s: %s", key, str(value))

    logging.info("Report available in directory: '%s'", output_dir)
