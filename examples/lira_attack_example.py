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
   *Programatically*
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
import os
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

from attacks import likelihood_attack # pylint: disable = import-error
from attacks import metrics # pylint: disable = import-error

# [Researcher] Access a dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=False)

# [Researcher] Split into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

# [Researcher] Define the classifier
target_model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1)
# [Researcher] Train the classifier
target_model.fit(train_X, train_y)

# [Researcher] Provide the model and the train and test data to the TRE

# [TRE] Compute the predictions on the training and test sets
train_preds = target_model.predict_proba(train_X)
test_preds = target_model.predict_proba(test_X)

# [TRE] Create a shadow model
shadow_clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1)

# [TRE] Call attack code
attack_scores, attack_labels, attack_classifier = likelihood_attack.likelihood_scenario(
    shadow_clf,
    train_X,
    train_y,
    train_preds,
    test_X,
    test_y,
    test_preds,
    n_shadow_models=50
)


# [TRE] Computes attack metrics
attack_metrics = metrics.get_metrics(
    attack_classifier,
    attack_scores,
    attack_labels
)

# [TRE] Looks at the metric values
print(f"AUC: {attack_metrics['AUC']:.3f}")
auc_p, _ = metrics.auc_p_val(
    attack_metrics['AUC'],
    attack_labels.sum(),
    len(attack_labels) - attack_labels.sum()
)
print(f"AUC p value: {auc_p:.3f}")
print(f"FDIF01: {attack_metrics['FDIF01']:.3f}")
print(f"PDIF01: {np.exp(-attack_metrics['PDIF01']):.3f}")

print("Programmatic example finished")
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

# [Researcher] Dump the training and test data to a .csv file
np.savetxt(
    "train_data.csv",
    np.hstack((train_X, train_y[:, None])),
    delimiter=","
)
np.savetxt(
    "test_data.csv",
    np.hstack((test_X, test_y[:, None])),
    delimiter=","
)

# [TRE] Creates a config file for the likelihood attack
config = {
    "training_data_file": "train_data.csv",
    "testing_data_file": "test_data.csv",
    "training_preds_file": "train_preds.csv",
    "testing_preds_file": "test_preds.csv",
    "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
    "target_hyppars": {
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
}

with open('config.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(config))


# [TRE] Runs the attack. This would be done on the command line, here we do that with os.system
# [TRE] First they access the help to work out which parameters they need to set
os.system("python -m attacks.likelihood_attack run-attack --help")

# [TRE] Then they run the attack
os.system(
    (
        "python -m attacks.likelihood_attack run-attack "
        "--json-file config.json "
        "--report-name example_lira_report.pdf "
        "--n-shadow-models 50"
    )
)

# [TRE] The code produces a .pdf report (example_lira_report.pdf)
