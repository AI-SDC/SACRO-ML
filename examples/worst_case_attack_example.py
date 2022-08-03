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
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

from attacks import worst_case_attack

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
    'n_reps': 10,
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



# Command line version