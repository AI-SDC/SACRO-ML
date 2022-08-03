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
"""
import os
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

from attacks import worst_case_attack # pylint: disable = import-error

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

# [TRE] Define some attack parameters
args_dict = {
    # How many attacks to run -- in each the attack model is trained on a different
    # subset of the data
    'n_reps': 2,
    # Threshold to determine significance of things
    'p_thresh': 0.05,
    # Filename arguments needed by the code, meaningless if run programatically
    'in_sample_filename': None,
    'out_sample_filename': None,
    # Proportion of data to use as a test set for the attack model;
    'test_prop': 0.5
}

# Convert into a namespace so that the code can access attributes with dot ('.') syntax
args = SimpleNamespace(**args_dict)

# [TRE] Call attack code
metrics, metadata = worst_case_attack.attack(args, train_preds, test_preds)

# [TRE] explore the metrics
# For how many of the reps is the AUC p-value significant, with and without FDR correction. A
# significant P-value means that the attack was statistically successful at predicting rows at
# belonging in the training set

print(
    "Number of significant AUC values (raw):",
    f"{metadata['global_metrics']['n_sig_auc_p_vals']}/{args.n_reps}"
)

print(
    "Number of significant AUC values (FDR corrected):",
    f"{metadata['global_metrics']['n_sig_auc_p_vals_corrected']}/{args.n_reps}"
)

# Or the number of repetitions in which the PDIF (0.1) was significant
print(
    "Number of significant PDIF values (proportion of 0.1), raw:",
    f"{metadata['global_metrics']['n_sig_pdif_vals']}/{args.n_reps}"
)

print(
    "Number of significant PDIF values (proportion of 0.1), FDR corrected:",
    f"{metadata['global_metrics']['n_sig_pdif_vals_corrected']}/{args.n_reps}"
)

# [TRE] to compare the results obtained with those expected by chance, TRE staff can
# also run dummy attacks
# [TRE] Generate some fake model predictions in which there is no difference between the
# distribution of probabilities over the training and testing data. The number of rows to
# generate are set to match the real example
dummy_train, dummy_test = worst_case_attack.generate_arrays(
    len(train_preds),
    len(test_preds)
)

# [TRE] runs the attacks on the dummy predictions
dummy_metrics, dummy_metadata = worst_case_attack.attack(args, dummy_train, dummy_test)
 
# [TRE] looks at the metric values to compare with those for the model
print(
    "(dummy) Number of significant AUC values (raw):",
    f"{dummy_metadata['global_metrics']['n_sig_auc_p_vals']}/{args.n_reps}"
)

print(
    "(dummy) Number of significant AUC values (FDR corrected):",
    f"{dummy_metadata['global_metrics']['n_sig_auc_p_vals_corrected']}/{args.n_reps}"
)

# Or the number of repetitions in which the PDIF (0.1) was significant
print(
    "(dummy) Number of significant PDIF values (proportion of 0.1), raw:",
    f"{dummy_metadata['global_metrics']['n_sig_pdif_vals']}/{args.n_reps}"
)

print(
    "(dummy) Number of significant PDIF values (proportion of 0.1), FDR corrected:",
    f"{dummy_metadata['global_metrics']['n_sig_pdif_vals_corrected']}/{args.n_reps}"
)

print("Programatic example finished")
print("****************************")

print()
print()
print("Command line example starting")
print("*****************************")
# Command line version. The same functionality as above, but the attack is run from
# the command line rather than programatically

# [Researcher] Dump the training and test predictions to .csv files
np.savetxt("train_preds.csv", train_preds, delimiter=",")
np.savetxt("test_preds.csv", test_preds, delimiter=",")

# [TRE] Runs the attack. This would be done on the command line, here we do that with os.system
# [TRE] First they access the help to work out which parameters they need to set
os.system("python -m attacks.worst_case_attack run-attack --help")

# [TRE] Then they run the attack
os.system(
    (
        "python -m attacks.worst_case_attack run-attack "
        "--in-sample-preds train_preds.csv "
        "--out-of-sample-preds test_preds.csv "
        "--n-reps 10 "
        "--report-name example_report "
        "--dummy-reps 1 "
        "--test-prop 0.1"
    )
)

# [TRE] The code produces a .pdf report (example_report.pdf) and a .json file that can be
# injesetd by the shiny app
