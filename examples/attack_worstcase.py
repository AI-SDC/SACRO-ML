"""Example running a worst case membership inference attack programmatically.

This code simulates a MIA attack providing the attacker with as much
information as possible. That is, they have a subset of rows that they _know_
were used for training. And a subset that they know were not. They also have
query access to the target model.

They pass the training and non-training rows through the target model to get
the predictive probabilities. These are then used to train an _attack model_.
And the attack model is evaluated to see how well it can predict whether or not
other examples were in the training set or not.

To compare the results obtained with those expected by chance, the attack runs
some baseline experiments too.

The steps are as follows:

1. Researcher trains their model, e.g., `train_rf_breast_cancer.py`
2. Researcher and/or TRE runs the attacks
     1. The TRE calls the attack code.
     2. The TRE computes and inspects attack metrics.
"""

import logging

from sacroml.attacks import worst_case_attack
from sacroml.attacks.target import Target

output_dir = "outputs_worstcase"
target_dir = "target_rf_breast_cancer"

if __name__ == "__main__":
    logging.info("Loading Target object from '%s'", target_dir)
    target = Target()
    target.load(target_dir)

    logging.info("Creating worst case attack")
    attack_obj = worst_case_attack.WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        train_beta=5,
        test_beta=2,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir=output_dir,
    )

    logging.info("Running worst case attack")
    output = attack_obj.attack(target)

    logging.info("Accessing attack metrics and metadata")
    metadata = output["metadata"]

    logging.info(
        "Number of significant AUC values (raw): %d/%d",
        metadata["global_metrics"]["n_sig_auc_p_vals"],
        attack_obj.n_reps,
    )

    logging.info(
        "Number of significant AUC values (FDR corrected): %d/%d",
        metadata["global_metrics"]["n_sig_auc_p_vals_corrected"],
        attack_obj.n_reps,
    )

    logging.info(
        "Number of significant PDIF values (proportion of 0.1), raw: %d/%d",
        metadata["global_metrics"]["n_sig_pdif_vals"],
        attack_obj.n_reps,
    )

    logging.info(
        "Number of significant PDIF values (proportion of 0.1), FDR corrected: %d/%d",
        metadata["global_metrics"]["n_sig_pdif_vals_corrected"],
        attack_obj.n_reps,
    )

    logging.info(
        "(dummy) Number of significant AUC values (raw): %d/%d",
        metadata["baseline_global_metrics"]["n_sig_auc_p_vals"],
        attack_obj.n_reps,
    )

    logging.info(
        "(dummy) Number of significant AUC values (FDR corrected): %d/%d",
        metadata["baseline_global_metrics"]["n_sig_auc_p_vals_corrected"],
        attack_obj.n_reps,
    )

    logging.info(
        "(dummy) Number of significant PDIF values (proportion of 0.1), raw: %d/%d",
        metadata["baseline_global_metrics"]["n_sig_pdif_vals"],
        attack_obj.n_reps,
    )

    logging.info(
        "(dummy) Number of significant PDIF values (proportion of 0.1) FDR corrected: %d/%d",
        metadata["baseline_global_metrics"]["n_sig_pdif_vals_corrected"],
        attack_obj.n_reps,
    )

    logging.info("Report available in directory: '%s'", output_dir)
