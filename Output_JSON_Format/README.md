# JSON output Standardisation
We standaridised the JSON output both for worst_case and LIRA attacks where possible. A generic JSON output strcture is presented as under:

## General Structure
Key components of JSON output across attacks will be:

````
log_id: Log identifier - a random unique id for each entry 
log_time: the time when the log was created
metadata: standardised variables related to a specific attack type
attack_experiment_logger: Attack experiment logger - maintains instances of metrics computed across iterations
````

### Worst-Case Attack
A worst case attack will have a following components in their metadata component of JSON output.
````
metadata:
    experiment_details: 
        n_reps: number of attacks to run -- in each iteration an attack model is trained on a different subset of the data
        p_thresh: threshold to determine significance of things. For instance auc_p_value and pdif_vals
        n_dummy_reps: number of baseline (dummy) experiments to do
        train_beta: value of b for beta distribution used to sample the in-sample (training) probabilities
        test_beta: value of b for beta distibution used to sample the out-of-sample (test) probabilities
        test_prop: proportion of data to use as a test set for the attack model
        n_rows_in:
        n_rows_out:
        training_preds_filename: name of the file keep predictions of the training data (in-sample)
        test_preds_filename: name of the file keep predictions of the test data (out-of-sample)
        report_name:
        include_model_correct_feature:
        sort_probs:
        mia_attack_model:
        mia_attack_model_hyp:
        attack_metric_success_name:
        attack_metric_success_comp_type:
        attack_metric_success_count_thresh:
        attack_fail_fast:
    attack:
    global_metric:
        null_auc_3sd_range:
        n_sig_auc_p_vals:
        n_sig_auc_p_vals_corrected:
        n_sig_pdif_vals:
        n_sig_pdif_vals_corrected:
    baseline_global_metric:
        null_auc_3sd_range:
        n_sig_auc_p_vals:
        n_sig_auc_p_vals_corrected:
        n_sig_pdif_vals:
        n_sig_pdif_vals_corrected:    
````