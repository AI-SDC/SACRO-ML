````
log_id: Log identifier - unique for each entry
timestamp: When the log was created
model_filename: filename of model under attack
data_filename: filename of input data (exact format to be determined)
attack_type: type of attack
attack_type_version: used if code is released and code versions get updated
fail_fast_threshold: threshold for the fail-fast parameter
attack_instance_logger: a list of dictionaries - each time an attack is run, parameters about the attack are stored (one dictionary per run)
    true_labels_distribution: count of how many of each class there are in the 'real' labels
    predicted_labels_distribution: count of how many of each class there are in the predicted labels
    metrics: comes from metrics.py
attack_metric_success_summary: 
    params:
	metric_name: name of the metric
	metric_succ_thresh: threshold for a metric to compare
	comp_type: type of comparison (based on relational operators such as gt, gte, lt, lte etc)
	fail_fast: true value will force to stop further attack repetitions based on success_count_thresh value
	success_count_thresh: threshold counter for attack being successful
    output:
	success_count: attack being successful across repetitions
	fail_count: attack not being successful across repetitions    
model: class name of model under attack
model_params: model parameters - comes directly from the model (unprocessed)
````
