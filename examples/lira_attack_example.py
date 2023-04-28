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

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks.dataset import Data  # pylint: disable = import-error
from aisdc.attacks.likelihood_attack import (  # pylint: disable = import-error
    LIRAAttack,
    LIRAAttackArgs,
)

# [Researcher] Access a dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=False)

# [Researcher] Split into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

# [Researcher] Define the classifier
target_model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1)
# [Researcher] Train the classifier
target_model.fit(train_X, train_y)

# [Researcher] Provide the model and the train and test data to the TRE
dataset = Data()
dataset.add_processed_data(train_X, train_y, test_X, test_y)

# [TRE] Creates a config file for the likelihood attack
config = {
    "training_data_file": "train_data.csv",
    "testing_data_file": "test_data.csv",
    "training_preds_file": "train_preds.csv",
    "testing_preds_file": "test_preds.csv",
    "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
    "target_hyppars": {"min_samples_split": 2, "min_samples_leaf": 1},
}

with open("config.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config))

# [TRE] sets up the attack
args = LIRAAttackArgs(n_shadow_models=100, report_name="lira_example_report")
attack_obj = LIRAAttack(args)

# [TRE] runs the attack
attack_obj.attack(dataset, target_model)

# [TRE] Get the output
output = attack_obj.make_report()  # also makes .pdf and .json files

# [TRE] Accesses attack metrics and metadata
attack_metrics = output["attack_metrics"][0]
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
os.system("python -m aisdc.attacks.likelihood_attack run-attack --help")

# [TRE] Then they run the attack
os.system(
    "python -m aisdc.attacks.likelihood_attack run-attack "
    "--json-file config.json "
    "--report-name example_lira_report "
    "--n-shadow-models 100"
)

# [TRE] The code produces a .pdf report (example_lira_report.pdf)
