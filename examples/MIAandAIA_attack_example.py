"""Examples for using the 'Membership Inferene Attack' attack code.

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
python -m examples.MIA_and_AIA_single_entry_example

"""

import json
import os
import sys

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from aisdc.attacks.multiple_attacks import ConfigFile  # pylint: disable = import-error
from aisdc.attacks.multiple_attacks import (
    MultipleAttacks,  # pylint: disable = import-error
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

target.save(path="MIA_target")

# Example 1: Adding multiple configurations to a configuration file
# and then running attacks programmatically
# creating an instance of Multiple Attacks object
# requiring name of the configuration file and
# output json file name
# MultipleAttacks class has a method add_config
# to add an attack configuration to a configuration file and
# attack method then runs attacks based on the specifications given in the configuration file
configfile_obj = ConfigFile(
    filename="single_config.json",
)

# Example 1: Adding a configuration dictionary to the JSON file
config = {
    "n_reps": 10,
    "n_dummy_reps": 1,
    "p_thresh": 0.05,
    "test_prop": 0.5,
    "train_beta": 5,
    "test_beta": 2,
    "report_name": "programmatically_worstcase_example1_report",
}
configfile_obj.add_config(config, "worst_case")

# Example 2: Adding a configuration dictionary to the JSON file
config = {
    "n_reps": 20,
    "n_dummy_reps": 1,
    "p_thresh": 0.05,
    "test_prop": 0.5,
    "train_beta": 5,
    "test_beta": 2,
    "report_name": "programmatically_worstcase_example2_report",
}
configfile_obj.add_config(config, "worst_case")

# Example 3: Adding a configuration dictionary to the JSON file
config = {
    "n_reps": 10,
    "n_dummy_reps": 1,
    "p_thresh": 0.05,
    "test_prop": 0.5,
    "train_beta": 5,
    "test_beta": 2,
    "report_name": "programmatically_worstcase_example3_report",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
    "attack_metric_success_name": "P_HIGHER_AUC",
    "attack_metric_success_thresh": 0.05,
    "attack_metric_success_comp_type": "lte",
    "attack_metric_success_count_thresh": 2,
    "attack_fail_fast": True,
}
configfile_obj.add_config(config, "worst_case")

# Example 4: Adding a configuration dictionary to the JSON file
config = {
    "n_shadow_models": 100,
    "report_name": "programmatically_lira_example1_report",
    "training_data_filename": "train_data.csv",
    "test_data_filename": "test_data.csv",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
    "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
    "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
}
configfile_obj.add_config(config, "lira")

# Example 5: Adding a configuration dictionary to the JSON file
config = {
    "n_shadow_models": 150,
    "report_name": "programmatically_lira_example2_report",
    "shadow_models_fail_fast": True,
    "n_shadow_rows_confidences_min": 10,
    "training_data_filename": "train_data.csv",
    "test_data_filename": "test_data.csv",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
    "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
    "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
}
configfile_obj.add_config(config, "lira")

# Example 5: Adding an existing configuration file to a single JSON configuration file
config = {
    "n_shadow_models": 120,
    "report_name": "programmatically_lira_example3_report",
    "shadow_models_fail_fast": True,
    "n_shadow_rows_confidences_min": 10,
    "training_data_filename": "train_data.csv",
    "test_data_filename": "test_data.csv",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
    "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
    "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
}
with open("lira_config.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config))
configfile_obj.add_config("lira_config.json", "lira")
# attack method not only runs attacks given the configurations
# specified but also generates a single JSON output file
# in case if output_filename is specified
attack_obj = MultipleAttacks(
    config_filename="single_config.json",
    output_filename="single_output.json",
)
attack_obj.attack(target)

# [Researcher] Dump the target model and target data
target.save(path="target")

# [TRE] Runs the attack. This would be done on the command line, here we do that with os.system
# [TRE] First they access the help to work out which parameters they need to set
os.system(
    f"{sys.executable} -m aisdc.attacks.multiple_attacks run-attack-from-configfile --help"
)

# [TRE] Then they run the attack
os.system(
    f"{sys.executable} -m aisdc.attacks.multiple_attacks run-attack-from-configfile "
    "--attack-config-json-file-name single_config.json "
    "--attack-target-folder-path target "
    "--attack-output-json-file-name single_output2.json "
)
