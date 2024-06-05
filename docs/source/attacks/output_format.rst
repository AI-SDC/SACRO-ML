JSON Output for MIA attacks
===========================

We standaridised the JSON output both for worst_case and LIRA attacks where possible. A generic JSON output structure is presented as under:

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

    experiment_details: this will have attack type parameters
        n_reps: number of attacks to run -- in each iteration an attack model is trained on a different subset of the data
        p_thresh: threshold to determine significance of things. For instance auc_p_value and pdif_vals
        n_dummy_reps: number of baseline (dummy) experiments to do
        train_beta: value of b for beta distribution used to sample the in-sample (training) probabilities
        test_beta: value of b for beta distribution used to sample the out-of-sample (test) probabilities
        test_prop: proportion of data to use as a test set for the attack model
        n_rows_in: number of rows for in-sample (training data)
        n_rows_out: number of rows for out-of-sample (test data)
        training_preds_filename: name of the file to keep predictions of the training data (in-sample)
        test_preds_filename: name of the file to keep predictions of the test data (out-of-sample)
        report_name: name of the JSON report
        include_model_correct_feature: inclusion of additional feature to hold whether or not the target model made a correct prediction for each example
        sort_probs: true in case require to sort combine preds (from training and test) to have highest probabilities in the first column
        mia_attack_model: name of the attack model suchas RandomForestClassifier
        mia_attack_model_hyp: list of hyper parameters for the mia_attack_model such as min_sample_split, min_samples_leaf, max_depth etc
        attack_metric_success_name: the name of metric to compute for the attack being successful
        attack_metric_success_thresh: threshold for a given metric to measure attack being successful or not
        attack_metric_success_comp_type: threshold comparison operator (i.e., gte: greater than or equal to, gt: greater than, lte: less than or equal to, lt: less than, eq: equal to and not_eq: not equal to)
        attack_metric_success_count_thresh: a counter to record how many times an attack was successful given that the threshold has fulfilled criteria for a given comparison type
        attack_fail_fast: If true it stops repetitions earlier based on the given attack metric (i.e., attack_metric_success_name) considering the comparison type (attack_metric_success_comp_type) satisfying a threshold (i.e., attack_metric_success_thresh) for n (attack_metric_success_count_thresh) number of times

    attack: name of the attack type ('WorstCase attack')

    global_metric: the following global metrics are computed for attack repetitions
        null_auc_3sd_range: a three standard deviation range from the mean for the observed p_value
        n_sig_auc_p_vals: number of significant p values given a p_thresh value
        n_sig_auc_p_vals_corrected: number of significant p values given a p_thresh value given applying testing corrections
        n_sig_pdif_vals: number of significant pdif given a p_thresh value
        n_sig_pdif_vals_corrected: number of significant p values given a p_thresh value given applying testing corrections

    baseline_global_metric: the following global metrics are computed for attack repetitions across all experiments of baseline (dummy) experiments
        null_auc_3sd_range: a three standard deviation range from the mean for the observed p_value
        n_sig_auc_p_vals: number of significant p values given a p_thresh value
        n_sig_auc_p_vals_corrected: number of significant p values given a p_thresh value given applying testing corrections
        n_sig_pdif_vals: number of significant pdif given a p_thresh value
        n_sig_pdif_vals_corrected: number of significant p values given a p_thresh value given applying testing corrections

A worst case attack will have experiment logger and baseline (dummy) experiments logger which is unique to worst case attack only.

attack_experiment_logger::

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

        instance_n:
            ... n will be n_reps-1 representing iterations of attacks
    attack_metric_failfast_summary:
        succcess_count: number of attacks being successful given the attack success criteria demonstrated in metadata
        fail_count: number of attacks being not successful

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
        attack_metric_failfast_summary:
            succcess_count: number of attacks being successful given the attack success criteria demonstrated in metadata
            fail_count: number of attacks being not successful
    dummy_attack_metrics_experiment_1:
        ...
        ...
    dummy_attack_metrics_experiment_n: n will be n_dummy_reps-1 representing iterations of attacks
        ...

Example JSON output for worst case attack is accessible from :download:`link <programmatically_worstcase_example1_report.json>`

LIRA Attack
-----------

A LIRA attack will have the following components in a metadata component of JSON output.

metadata::

    experiment_details: this will have attack type parameters
        n_shadow_models: number of shadow models to be trained
        p_thresh: threshold to determine significance of things. For instance auc_p_value and pdif_vals
        report_name: name of the JSON report
        training_data_filename: name of the data file for the training data (in-sample)
        test_data_filename: name of the file for the test data (out-of-sample)
        training_preds_filename: name of the file to keep predictions of the training data (in-sample)
        test_preds_filename: name of the file to keep predictions of the test data (out-of-sample)
        target_model: name of the attack model suchas RandomForestClassifier
        target_model_hyp: list of hyper parameters for the mia_attack_model such as min_sample_split, min_samples_leaf etc
        n_shadow_rows_confidences_min: number of minimum number of confidences calculated for each row in test data (out-of-sample)
        attack_fail_fast: If true it stops repetitions earlier based on the given minimum number of confidences for each row in the test data

    attack: name of the attack type ('WorstCase attack')

    global_metric: the following global metrics are computed for attack repetitions
        null_auc_3sd_range: a three standard deviation range from the mean for the observed p_value
        AUC_sig: significant AUC at given p value
        PDIF_sig: significant PDIF at given p value

A LIRA attack will have experiment logger with only one instance.

attack_experiment_logger::

    attack_instance_logger: stores metrics computed across all iteration of attacks (i.e. n_reps)
        instance_0: For a lira attack type, this will have a single instance
            TPR: value of true positive rate
            FPR: value of false positive rate
            ...
            ...
            n_pos_test_examples:
            n_neg_test_examples:
            n_shadow_models_trained: this represent number of actual models trained. For a case where attack_fail_fast is true and minimum number of confidences computed for each row in the test data, there is likely to be a chance to have less number of shadow models trained satisfying the given criteria

Example JSON output for LIRA attack is accessible from :download:`link <lira_example1_report.json>`

Running MIA Attacks from Config File
====================================

Both for worst case and LIRA attacks, examples presented `worst_case_attack_example <https://github.com/AI-SDC/AI-SDC/blob/development/examples/worst_case_attack_example.py/>`_
and `lira_attack_example <https://github.com/AI-SDC/AI-SDC/blob/development/examples/lira_attack_example.py/>`_ in the AI-SDC explains most of the possible use of configuration files.
