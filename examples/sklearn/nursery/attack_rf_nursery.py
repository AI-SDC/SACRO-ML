"""Example programmatically running attacks."""

import logging

from sacroml.attacks.attribute_attack import AttributeAttack
from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.structural_attack import StructuralAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack

output_dir = "output_rf_nursery"
target_dir = "target_rf_nursery"

if __name__ == "__main__":
    logging.info("Loading Target object from '%s'", target_dir)
    target = Target()
    target.load(target_dir)

    logging.info("Running LiRA attack")
    attack = LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    output = attack.attack(target)

    logging.info("Running worst case attack")
    attack = WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        train_beta=5,
        test_beta=2,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir=output_dir,
    )
    output = attack.attack(target)

    logging.info("Running structural attack")
    attack = StructuralAttack(output_dir=output_dir)
    output = attack.attack(target)

    logging.info("Running attribute attack")
    attack = AttributeAttack(n_cpu=8, output_dir=output_dir)
    output = attack.attack(target)

    logging.info("Report available in directory: '%s'", output_dir)
