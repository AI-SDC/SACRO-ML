"""Examples for using the 'worst case' attack code.

This code simulates a MIA attack providing the attacker with as much information as possible.
i.e. they have a subset of rows that they _know_ were used for training. And a subset that they
know were not. They also have query access to the target model.

They pass the training and non-training rows through the target model to get the predictive
probabilities. These are then used to train an _attack model_. And the attack model is evaluated
to see how well it can predict whether or not other examples were in the training set or not.

The code can be called from the command line, or accessed programmatically. Examples of both
are shown below.

In the code, [Researcher] and [TRE] are used in comments to denote which bit is done by whom

Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.worst_case_attack_example

"""
import json
import os
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from aisdc.attacks import worst_case_attack  # pylint: disable = import-error
from aisdc.attacks.attack_report_formatter import (
    GenerateJSONModule,  # pylint: disable = import-error
)
from aisdc.attacks.target import Target  # pylint: disable = import-error

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# [Researcher] Access a dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=False)

# [Researcher] Split into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

# [Researcher] Define the classifier
target_model = SVC(gamma=0.1, probability=True)

# [Researcher] Train the classifier
target_model.fit(train_X, train_y)

# [Researcher] Provide the model and the train and test data to the TRE

# [TRE] Compute the predictions on the training and test sets
train_preds = target_model.predict_proba(train_X)
test_preds = target_model.predict_proba(test_X)

# [TRE / Researcher] Wrap the model and data in a Target object
target = Target(model=target_model)
target.add_processed_data(train_X, train_y, test_X, test_y)

# [TRE] Create the attack object
attack_obj = worst_case_attack.WorstCaseAttack(
    # How many attacks to run -- in each the attack model is trained on a different
    # subset of the data
    n_reps=10,
    # number of baseline (dummy) experiments to do
    n_dummy_reps=1,
    # value of b for beta distribution used to sample the in-sample probabilities
    train_beta=5,
    # value of b for beta distribution used to sample the out-of-sample probabilities
    test_beta=2,
    # Threshold to determine significance of things
    p_thresh=0.05,
    # Filename arguments needed by the code, meaningless if run programmatically
    training_preds_filename=None,
    test_preds_filename=None,
    # Proportion of data to use as a test set for the attack model;
    test_prop=0.5,
    # If Report name is given so it creates Json file; however when it is None -
    # don't make json file
    report_name="programmatically_worstcase_example1_report",
    # Setting the name of metric to compute failures
    attack_metric_success_name="P_HIGHER_AUC",
    # threshold for a given metric for failure/success counters
    attack_metric_success_thresh=0.05,
    # threshold comparison operator (i.e., gte: greater than or equal to, gt: greater than, lte:
    # less than or equal to, lt: less than, eq: equal to and not_eq: not equal to)
    attack_metric_success_comp_type="lte",
    # fail fast counter to stop further repetitions of the test
    attack_metric_success_count_thresh=2,
    # If true it stop repetitions earlier based on the given attack metric
    # (i.e., attack_metric_success_name) considering the comparison type
    # (attack_metric_success_comp_type) satisfying a threshold (i.e., attack_metric_success_thresh)
    #  for n (attack_metric_success_count_thresh) number of times
    attack_fail_fast=True,
)

# [TRE] Run the attack
attack_obj.attack(target)

# [TRE] Grab the output
output = attack_obj.make_report(GenerateJSONModule("worst_case_attack.json"))
metadata = output["metadata"]
# [TRE] explore the metrics
# For how many of the reps is the AUC p-value significant, with and without FDR correction. A
# significant P-value means that the attack was statistically successful at predicting rows at
# belonging in the training set

print(
    "Number of significant AUC values (raw):",
    f"{metadata['global_metrics']['n_sig_auc_p_vals']}/{attack_obj.n_reps}",
)

print(
    "Number of significant AUC values (FDR corrected):",
    f"{metadata['global_metrics']['n_sig_auc_p_vals_corrected']}/{attack_obj.n_reps}",
)

# Or the number of repetitions in which the PDIF (0.1) was significant
print(
    "Number of significant PDIF values (proportion of 0.1), raw:",
    f"{metadata['global_metrics']['n_sig_pdif_vals']}/{attack_obj.n_reps}",
)

print(
    "Number of significant PDIF values (proportion of 0.1), FDR corrected:",
    f"{metadata['global_metrics']['n_sig_pdif_vals_corrected']}/{attack_obj.n_reps}",
)

# [TRE] to compare the results obtained with those expected by chance, the attack runs some
# Baseline experiments too

# [TRE] looks at the metric values to compare with those for the model
print(
    "(dummy) Number of significant AUC values (raw):",
    f"{metadata['baseline_global_metrics']['n_sig_auc_p_vals']}/{attack_obj.n_reps}",
)

print(
    "(dummy) Number of significant AUC values (FDR corrected):",
    f"{metadata['baseline_global_metrics']['n_sig_auc_p_vals_corrected']}/{attack_obj.n_reps}",
)

# Or the number of repetitions in which the PDIF (0.1) was significant
print(
    "(dummy) Number of significant PDIF values (proportion of 0.1), raw:",
    f"{metadata['baseline_global_metrics']['n_sig_pdif_vals']}/{attack_obj.n_reps}",
)

print(
    "(dummy) Number of significant PDIF values (proportion of 0.1), FDR corrected:",
    f"{metadata['baseline_global_metrics']['n_sig_pdif_vals_corrected']}/{attack_obj.n_reps}",
)

print("Programmatic example1 finished")
print("****************************")

# Example 2: Use of configuration file name to pass through and load parameters
# and running attack programmatically
config = {
    "n_reps": 10,
    "n_dummy_reps": 1,
    "p_thresh": 0.05,
    "test_prop": 0.5,
    "train_beta": 5,
    "test_beta": 2,
    "report_name": "programmatically_worstcase_example2_report",
}

with open("config_worstcase.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config))

# [TRE] Create the attack object
attack_obj = worst_case_attack.WorstCaseAttack(
    # name of the configuration file in JSON format to load parameters
    attack_config_json_file_name="config_worstcase.json",
)

# [TRE] Run the attack
attack_obj.attack(target)

# [TRE] Grab the output
output = attack_obj.make_report(GenerateJSONModule("worst_case_attack.json"))
metadata = output["metadata"]
# [TRE] explore the metrics
# For how many of the reps is the AUC p-value significant, with and without FDR correction. A
# significant P-value means that the attack was statistically successful at predicting rows at
# belonging in the training set

print(
    "Number of significant AUC values (raw):",
    f"{metadata['global_metrics']['n_sig_auc_p_vals']}/{attack_obj.n_reps}",
)

print(
    "Number of significant AUC values (FDR corrected):",
    f"{metadata['global_metrics']['n_sig_auc_p_vals_corrected']}/{attack_obj.n_reps}",
)

# Or the number of repetitions in which the PDIF (0.1) was significant
print(
    "Number of significant PDIF values (proportion of 0.1), raw:",
    f"{metadata['global_metrics']['n_sig_pdif_vals']}/{attack_obj.n_reps}",
)

print(
    "Number of significant PDIF values (proportion of 0.1), FDR corrected:",
    f"{metadata['global_metrics']['n_sig_pdif_vals_corrected']}/{attack_obj.n_reps}",
)

# [TRE] to compare the results obtained with those expected by chance, the attack runs some
# Baseline experiments too

# [TRE] looks at the metric values to compare with those for the model
print(
    "(dummy) Number of significant AUC values (raw):",
    f"{metadata['baseline_global_metrics']['n_sig_auc_p_vals']}/{attack_obj.n_reps}",
)

print(
    "(dummy) Number of significant AUC values (FDR corrected):",
    f"{metadata['baseline_global_metrics']['n_sig_auc_p_vals_corrected']}/{attack_obj.n_reps}",
)

# Or the number of repetitions in which the PDIF (0.1) was significant
print(
    "(dummy) Number of significant PDIF values (proportion of 0.1), raw:",
    f"{metadata['baseline_global_metrics']['n_sig_pdif_vals']}/{attack_obj.n_reps}",
)

print(
    "(dummy) Number of significant PDIF values (proportion of 0.1), FDR corrected:",
    f"{metadata['baseline_global_metrics']['n_sig_pdif_vals_corrected']}/{attack_obj.n_reps}",
)

print("Programmatic example2 finished")
print("****************************")

print()
print()
print("Command line example starting")
print("*****************************")
# Command line version. The same functionality as above, but the attack is run from
# the command line rather than programmatically

# [Researcher] Dump the training and test predictions to .csv files
np.savetxt("train_preds.csv", train_preds, delimiter=",")
np.savetxt("test_preds.csv", test_preds, delimiter=",")

# [Researcher] Dump the target model and target data
target.save(path="worstcase_target")

# [TRE] Runs the attack. This would be done on the command line, here we do that with os.system
# [TRE] First they access the help to work out which parameters they need to set
os.system(f"{sys.executable} -m aisdc.attacks.worst_case_attack run-attack --help")

# [TRE] Then they run the attack
# Example 1: Worstcase attack through commandline by passing parameters
os.system(
    f"{sys.executable} -m aisdc.attacks.worst_case_attack run-attack "
    "--training-preds-filename train_preds.csv "
    "--test-preds-filename test_preds.csv "
    "--n-reps 10 "
    "--report-name commandline_worstcase_example1_report "
    "--n-dummy-reps 1 "
    "--test-prop 0.1 "
    "--train-beta 5 "
    "--test-beta 2 "
    "--attack-metric-success-name P_HIGHER_AUC "
    "--attack-metric-success-thresh 0.05 "
    "--attack-metric-success-comp-type lte "
    "--attack-metric-success-count-thresh 2 "
    "--attack-fail-fast "
)

# [TRE] Runs the attack. This would be done on the command line, here we do that with os.system
# [TRE] First they access the help to work out which parameters they need to set
os.system(f"{sys.executable} -m aisdc.attacks.worst_case_attack run-attack-from-configfile --help")

# Example 2: Worstcase attack by passing a configuratation file name for loading parameters
config = {
    "n_reps": 10,
    "n_dummy_reps": 1,
    "p_thresh": 0.05,
    "test_prop": 0.5,
    "train_beta": 5,
    "test_beta": 2,
    "report_name": "commandline_worstcase_example2_report",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
    "attack_metric_success_name": "P_HIGHER_AUC",
    "attack_metric_success_thresh": 0.05,
    "attack_metric_success_comp_type": "lte",
    "attack_metric_success_count_thresh": 2,
    "attack_fail_fast": True,
}

with open("config_worstcase_cmd.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config))

os.system(
    f"{sys.executable} -m aisdc.attacks.worst_case_attack run-attack-from-configfile "
    "--attack-config-json-file-name config_worstcase_cmd.json "
    "--attack-target-folder-path worstcase_target "
)

# Example 3: Worstcase attack by passing a configuratation file name for loading parameters
config = {
    "n_reps": 10,
    "n_dummy_reps": 1,
    "p_thresh": 0.05,
    "test_prop": 0.5,
    "train_beta": 5,
    "test_beta": 2,
    "report_name": "commandline_worstcase_example3_report",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
}

with open("config_worstcase_cmd.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config))

os.system(
    f"{sys.executable} -m aisdc.attacks.worst_case_attack run-attack-from-configfile "
    "--attack-config-json-file-name config_worstcase_cmd.json "
    "--attack-target-folder-path worstcase_target "
)

# [TRE] The code produces a .pdf report (example_report.pdf) and a .json file (example_report.json)
# that can be injesetd by the shiny app
