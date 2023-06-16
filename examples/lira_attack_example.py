"""Examples for using the likelihood ratio attack code.

This code simulates a MIA attack providing the attacker with as much information as possible.
i.e. they have a subset of rows that they _know_ were used for training. And a subset that they
know were not. They also have query access to the target model.

The attack proceeds as described in this paper:
https://arxiv.org/pdf/2112.03570.pdf

The steps are as follows:

1. The researcher partitions their data into training and testing subsets
2. The researcher trains their model
3. The TRE runs the attacks
   *Programmatically*
     1. The TRE calls the attack code.
     2. The TRE computes and inspects attack metrics.
   *Command line
     3. The researcher writes out their training and testing data, as well as the predictions
        that their target model makes on this data.
     4. The TRE create a config file for the attack, specifying the file names for the files created
        in the previous two steps, as well as specifications for the shadow models.
     5. The attack is run with a command line command, creating a report.


Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.lira_attack_example

"""
# pylint: disable = duplicate-code

import json
import os
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks.attack_report_formatter import GenerateJSONModule
from aisdc.attacks.likelihood_attack import LIRAAttack  # pylint: disable = import-error
from aisdc.attacks.target import Target  # pylint: disable = import-error

# [Researcher] Access a dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=False)

# [Researcher] Split into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

# [Researcher] Define the classifier
target_model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1)
# [Researcher] Train the classifier
target_model.fit(train_X, train_y)

# [Researcher] Provide the model and the train and test data to the TRE
target = Target(model=target_model)
target.add_processed_data(train_X, train_y, test_X, test_y)

# [TRE] Creates a config file for the likelihood attack
config = {
    "training_data_filename": "train_data.csv",
    "test_data_filename": "test_data.csv",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
    "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
    "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
}

with open("lira_config.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config))

# [TRE] Example 1: sets up the attack
attack_obj = LIRAAttack(
    n_shadow_models=100,
    report_name="lira_example1_report",
    attack_config_json_file_name="lira_config.json",
)

# [TRE] runs the attack
attack_obj.attack(target)

# [TRE] Get the output
output = attack_obj.make_report(
    GenerateJSONModule("lira_attack.json")
)  # also makes .pdf and .json files

# [TRE] Accesses attack metrics and metadata
attack_metrics = output["attack_experiment_logger"]["attack_instance_logger"][
    "instance_0"
]
metadata = output["metadata"]

# [TRE] Looks at the metric values
print("Attack metrics:")
for key, value in attack_metrics.items():
    try:
        print(key, f"{value:.3f}")
    except TypeError:
        print(f"Cannot print {key}")

print("Global metrics")
for key, value in metadata["global_metrics"].items():
    print(key, value)


print("Programmatic example finished")
print("****************************")

# [TRE] Example 2: sets up the attack with fail-fast option
attack_obj = LIRAAttack(
    n_shadow_models=100,
    report_name="lira_example2_report",
    attack_config_json_file_name="lira_config.json",
    shadow_models_fail_fast=True,
    n_shadow_rows_confidences_min=10,
)

# [TRE] runs the attack
attack_obj.attack(target)

# [TRE] Get the output
output = attack_obj.make_report(
    GenerateJSONModule("lira_attack.json")
)  # also makes .pdf and .json files

# [TRE] Accesses attack metrics and metadata
attack_metrics = output["attack_experiment_logger"]["attack_instance_logger"][
    "instance_0"
]
metadata = output["metadata"]

# [TRE] Looks at the metric values
print("Attack metrics:")
for key, value in attack_metrics.items():
    try:
        print(key, f"{value:.3f}")
    except TypeError:
        print(f"Cannot print {key}")

print("Global metrics")
for key, value in metadata["global_metrics"].items():
    print(key, value)


print("Programmatic example with fail-fast option finished")
print("****************************")

print()
print()
print("Command line example starting")
print("*****************************")
# Command line version. The same functionality as above, but the attack is run from
# the command line rather than programmatically

# [Researcher] Dump the training and test predictions to .csv files
np.savetxt("train_preds.csv", target_model.predict_proba(train_X), delimiter=",")
np.savetxt("test_preds.csv", target_model.predict_proba(test_X), delimiter=",")

# [Researcher] Dump the training and test data to a .csv file
np.savetxt("train_data.csv", np.hstack((train_X, train_y[:, None])), delimiter=",")
np.savetxt("test_data.csv", np.hstack((test_X, test_y[:, None])), delimiter=",")


# [TRE] Runs the attack. This would be done on the command line, here we do that with os.system
# [TRE] First they access the help to work out which parameters they need to set
os.system(f"{sys.executable} -m aisdc.attacks.likelihood_attack run-attack --help")

# [TRE] Then they run the attack
# Example 1 to demonstrate all given shadow models trained
os.system(
    f"{sys.executable} -m aisdc.attacks.likelihood_attack run-attack "
    "--attack-config-json-file-name lira_config.json "
    "--report-name commandline_lira_example1_report "
    "--n-shadow-models 100 "
)

# Example 2 to demonstrate fail fast of shadow models trained
os.system(
    f"{sys.executable} -m aisdc.attacks.likelihood_attack run-attack "
    "--attack-config-json-file-name lira_config.json "
    "--report-name commandline_lira_example2_report "
    "--n-shadow-models 100 "
    "--shadow-models-fail-fast "
    "--n-shadow-rows-confidences-min 10 "
)

# Example 3 to demonstrate running attack from configuration file only
config = {
    "n_shadow_models": 150,
    "report_name": "commandline_lira_example3_report",
    "training_data_filename": "train_data.csv",
    "test_data_filename": "test_data.csv",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
    "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
    "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
}

with open("config_lira_cmd1.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config))

os.system(
    f"{sys.executable} -m aisdc.attacks.likelihood_attack run-attack-from-configfile "
    "--attack-config-json-file-name config_lira_cmd1.json "
)

# Example 4 to demonstrate running attack from configuration file only with fail fail fast option
config = {
    "n_shadow_models": 150,
    "report_name": "commandline_lira_example4_report",
    "shadow_models_fail_fast": True,
    "n_shadow_rows_confidences_min": 10,
    "training_data_filename": "train_data.csv",
    "test_data_filename": "test_data.csv",
    "training_preds_filename": "train_preds.csv",
    "test_preds_filename": "test_preds.csv",
    "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
    "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
}

with open("config_lira_cmd2.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config))

os.system(
    f"{sys.executable} -m aisdc.attacks.likelihood_attack run-attack-from-configfile "
    "--attack-config-json-file-name config_lira_cmd2.json "
)


# [TRE] The code produces a .pdf report (example_lira_report.pdf)
