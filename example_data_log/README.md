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
model: class name of model under attack
model_params: model parameters - comes directly from the model (unprocessed)