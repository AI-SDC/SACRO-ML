JSON Output for Attacks
=======================

JSON output has been standardised where possible. A generic JSON output structure is presented as under:

General Structure
-----------------

Key components of JSON output across attacks will be::

    log_id: Log identifier - a random unique id for each entry
    log_time: the time when the log was created
    metadata: standardised variables related to a specific attack type
    attack_experiment_logger: Attack experiment logger - maintains instances of metrics computed across iterations

Worst-Case Attack
-----------------

A worst case attack will have the following components in a metadata component of JSON output.

metadata::

    attack_name: Name of the attack
    attack_params: Attack parameters
    target_model: Name of the target model
    target_model_params: Target model parameters
    global_metrics: The following global metrics are computed for attack repetitions
        null_auc_3sd_range: A three standard deviation range from the mean for the observed p_value
        n_sig_auc_p_vals: Number of significant p values given a p_thresh value
        n_sig_auc_p_vals_corrected: Number of significant p values given a p_thresh value given applying testing corrections
        n_sig_pdif_vals: Number of significant pdif given a p_thresh value
        n_sig_pdif_vals_corrected: Number of significant p values given a p_thresh value given applying testing corrections

    baseline_global_metric: The following global metrics are computed for attack repetitions across all experiments of baseline (dummy) experiments
        null_auc_3sd_range: A three standard deviation range from the mean for the observed p_value
        n_sig_auc_p_vals: Number of significant p values given a p_thresh value
        n_sig_auc_p_vals_corrected: Number of significant p values given a p_thresh value given applying testing corrections
        n_sig_pdif_vals: Number of significant pdif given a p_thresh value
        n_sig_pdif_vals_corrected: Number of significant p values given a p_thresh value given applying testing corrections

A worst case attack will have experiment logger and baseline (dummy) experiments logger which is unique to worst case attack only.

attack_experiment_logger::

    attack_instance_logger: Stores metrics computed across all iteration of attacks (i.e. n_reps)
        instance_0:
            TPR: value of true positive rate
            FPR: value of false positive rate
            ...
            ...
            n_pos_test_examples:
            n_neg_test_examples:

        instance_1:
            ... all metric values computed similar to instance_0

        instance_n:
            ... n will be n_reps-1 representing iterations of attacks

dummy_attack_experiments_logger::

    dummy_attack_metrics_experiment_0:
       attack_instance_logger: stores metrics computed across all iteration of attacks (i.e. n_reps)
            instance_0:
                TPR: value of true positive rate
                FPR: value of false positive rate
                ...
                ...
                n_pos_test_examples:
                n_neg_test_examples:

            instance_1:
                ... all metric values computed similar to instance_0

            instance_n: n will be n_reps-1 representing iterations of attacks
                ...
    dummy_attack_metrics_experiment_1:
        ...
        ...
    dummy_attack_metrics_experiment_n: n will be n_dummy_reps-1 representing iterations of attacks
        ...

Example JSON output for worst case attack is accessible from :download:`link <report_example_worstcase.json>`

LiRA Attack
-----------

A LiRA attack will have the following components in a metadata component of JSON output.

metadata::

    attack_name: Name of the attack
    attack_params: Attack parameters
    target_model: Name of the target model
    target_model_params: Target model parameters
    global_metric: The following global metrics are computed for attack repetitions
        null_auc_3sd_range: A three standard deviation range from the mean for the observed p_value
        AUC_sig: Significant AUC at given p value
        PDIF_sig: Significant PDIF at given p value

A LiRA attack will have experiment logger with only one instance.

attack_experiment_logger::

    attack_instance_logger: stores metrics computed across all iteration of attacks (i.e. n_reps)
        instance_0: For a lira attack type, this will have a single instance
            TPR: value of true positive rate
            FPR: value of false positive rate
            ...
            ...
            n_pos_test_examples:
            n_neg_test_examples:

Example JSON output for LiRA attack is accessible from :download:`link <report_example_lira.json>`
