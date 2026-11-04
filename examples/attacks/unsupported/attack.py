"""Example of how to run attacks with saved predicted probabilities.

Note: a limited number of attacks can run in this scenario.
"""

import logging

import numpy as np

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.structural_attack import StructuralAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack

logging.basicConfig(level=logging.INFO)

output_dir = "output_rf_breast_cancer"

if __name__ == "__main__":
    logging.info("Loading predicted probabilities")
    proba_train = np.loadtxt("proba_train.csv", delimiter=",")
    proba_test = np.loadtxt("proba_test.csv", delimiter=",")
    target = Target(proba_train=proba_train, proba_test=proba_test)

    logging.info("Attempting to run LiRA attack... can't run in this scenario")
    attack = LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    attack.attack(target)

    logging.info("Running worst case attack")
    # Note: specifying the attack classifier is optional
    attack_model = "sklearn.linear_model.LogisticRegression"
    attack_model_params = {
        "solver": "lbfgs",
        "max_iter": 200,
        "class_weight": "balanced",
    }

    attack = WorstCaseAttack(
        attack_model=attack_model,
        attack_model_params=attack_model_params,
        n_reps=10,
        n_dummy_reps=1,
        train_beta=5,
        test_beta=2,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir=output_dir,
    )
    attack.attack(target)

    logging.info("Attempting to run structural attack... can't run in this scenario")
    attack = StructuralAttack(output_dir=output_dir)
    attack.attack(target)

    logging.info("Report available in directory: '%s'", output_dir)
