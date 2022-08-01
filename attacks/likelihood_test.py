'''
Likelihood testing scenario from https://arxiv.org/pdf/2112.03570.pdf
'''
# pylint: disable = invalid-name
import argparse
import json
import logging
import importlib
from typing import Iterable, Any, Dict
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from  sklearn.datasets import load_breast_cancer

from attacks import report
from attacks import metrics

logging.basicConfig(level=logging.INFO)

N_SHADOW_MODELS = 256 # Number of shadow models that should be trained
EPS = 1e-16 # Used to avoid numerical issues in logit function
P_THRESH = 0.05 # default significance threshold

class DummyClassifier:
    '''A Dummy Classifier to allow this code to work with get_metrics'''
    def predict(self, test_X):
        '''Return an array of 1/0 depending on value in second column'''
        return 1 * (test_X[:, 1] > 0.5)

    def predict_proba(self, test_X):
        '''Simply return the test_X'''
        return test_X

def _logit(p: float) -> float:
    '''Standard logit function'''
    if p > 1 - EPS:
        p = 1 - EPS
    p = max(p, EPS)
    li = np.log(p / (1 - p))
    return li


def likelihood_scenario( # pylint: disable = too-many-locals, too-many-arguments
    shadow_clf,
    X_target_train: Iterable[float],
    y_target_train: Iterable[float],
    target_train_preds: Iterable[float],
    X_shadow_train: Iterable[float],
    y_shadow_train: Iterable[float],
    shadow_train_preds: Iterable[float],
    n_shadow_models: int=N_SHADOW_MODELS
):
    '''
    Implements the likelihood test, using the "offline" version
    See p.6 (top of second column) for details
    '''
    logger = logging.getLogger("lr-scenario")
    n_train_rows, _ = X_target_train.shape
    n_shadow_rows, _ = X_shadow_train.shape
    indices = np.arange(0, n_train_rows + n_shadow_rows, 1)

    # Combine taregt and shadow train, from which to sample datasets
    combined_X_train = np.vstack((
        X_target_train,
        X_shadow_train
    ))
    combined_y_train = np.hstack((
        y_target_train,
        y_shadow_train
    ))

    train_row_to_confidence = {i: [] for i in range(n_train_rows)}
    shadow_row_to_confidence = {i: [] for i in range(n_shadow_rows)}

    # Train N_SHADOW_MODELS shadow models
    logger.info("Training shadow models")
    for model_idx in range(n_shadow_models):
        if model_idx % 10 == 0:
            logger.info("Trained %d models", model_idx)
        # Pick the indices to use for training this one
        np.random.seed(model_idx) # Reproducibility
        these_idx = np.random.choice(indices, n_train_rows, replace=False)
        temp_X_train = combined_X_train[these_idx, :]
        temp_y_train = combined_y_train[these_idx]

        # Fit the shadow model
        shadow_clf.set_params(random_state=model_idx)
        shadow_clf.fit(temp_X_train, temp_y_train)

        # Get the predicted probabilities on the training data
        confidences = shadow_clf.predict_proba(X_target_train)
        these_idx = set(these_idx)
        for i in range(n_train_rows):
            if i not in these_idx:
                # If i was _not_ used for training, incorporate the logit of its confidence of
                # being correct - TODO: should we just be taking max??
                train_row_to_confidence[i].append(
                    _logit(
                        confidences[i, y_target_train[i]]
                    )
                )
        # Same process for shadow data
        shadow_confidences = shadow_clf.predict_proba(X_shadow_train)
        for i in range(n_shadow_rows):
            if i + n_train_rows not in these_idx:
                shadow_row_to_confidence[i].append(
                    _logit(
                        shadow_confidences[i, y_shadow_train[i]]
                    )
                )

    # Do the test described in the paper in each case
    mia_scores = []
    mia_labels = []
    logger.info("Computing scores for train rows")
    for i in range(n_train_rows):
        true_score = _logit(target_train_preds[i, y_target_train[i]])
        null_scores = np.array(train_row_to_confidence[i])
        mean_null = null_scores.mean()
        var_null = max(null_scores.var(), EPS) # var can be zero in some cases
        prob = norm.cdf(true_score, loc=mean_null, scale=np.sqrt(var_null))
        mia_scores.append([1 - prob, prob])
        mia_labels.append(1)

    logger.info("Computing scores for shadow rows")
    for i in range(n_shadow_rows):
        true_score = _logit(shadow_train_preds[i, y_shadow_train[i]])
        null_scores = np.array(shadow_row_to_confidence[i])
        mean_null = null_scores.mean()
        var_null = max(null_scores.var(), EPS) # var can be zeros in some cases
        prob = norm.cdf(true_score, loc=mean_null, scale=np.sqrt(var_null))
        mia_scores.append([1 - prob, prob])
        mia_labels.append(0)

    mia_clf = DummyClassifier()
    logger.info("Finished scenario")
    return np.array(mia_scores), np.array(mia_labels), mia_clf

def _run_attack(args: dict) -> None: # pylint: disable = too-many-locals
    '''Run attack'''
    logger = logging.getLogger("run-attack")
    logger.info("Reading config from %s", args.json_file)
    with open(args.json_file, 'r', encoding='utf-8') as f:
        config = json.loads(f.read())

    logger.info("Loading training data csv from %s", config['training_data_file'])
    training_data = np.loadtxt(config['training_data_file'], delimiter=",")
    train_X = training_data[:, :-1]
    train_y = training_data[:, -1].flatten().astype(int)
    logger.info("Loaded %d rows", len(train_X))

    logger.info("Loading test data csv from %s", config['testing_data_file'])
    test_data = np.loadtxt(config['testing_data_file'], delimiter=",")
    test_X = test_data[:, :-1]
    test_y = test_data[:, -1].flatten().astype(int)
    logger.info("Loaded %d rows", len(test_X))

    logger.info("Loading train predictions form %s", config['training_preds_file'])
    train_preds = np.loadtxt(config['training_preds_file'], delimiter=",")
    assert len(train_preds) == len(train_X)

    logger.info("Loading test predictions form %s", config['testing_preds_file'])
    test_preds = np.loadtxt(config['testing_preds_file'], delimiter=",")
    assert len(test_preds) == len(test_X)

    clf_module_name, clf_class_name = config['target_model']
    module = importlib.import_module(clf_module_name)
    clf_class = getattr(module, clf_class_name)
    clf_params = config['target_hyppars']
    clf = clf_class(**clf_params)
    logger.info("Created model: %s", str(clf))
    mia_test_probs, mia_test_labels, mia_clf = likelihood_scenario(
        RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, max_depth=10),
        train_X,
        train_y,
        train_preds,
        test_X,
        test_y,
        test_preds,
        n_shadow_models=args.n_shadow_models
    )
    metadata = {
        'input-config': args.json_file,
        'n_shadow_models': args.n_shadow_models
    }
    logger.info("Computing metrics")
    mia_metrics = metrics.get_metrics(
        mia_clf,
        mia_test_probs,
        mia_test_labels
    )
    # Check for significance of AUC and PDIF
    logger.info("Creating report")
    pdif = np.exp(-mia_metrics['PDIF'])
    metadata['PDIF_sig'] = f"Significant at p={args.p_thresh}" if pdif <= args.p_thresh \
        else f"Not significant at p={args.p_thresh}"
    auc_std = np.sqrt(
        metrics.expected_auc_var(
            0.5,
            mia_test_labels.sum(),
            len(mia_test_labels) - mia_test_labels.sum()
        )
    )
    auc_p = 1 - norm.cdf(mia_metrics['AUC'], loc=0.5, scale=auc_std)
    metadata['AUC_sig'] = f"Significant at p={args.p_thresh}" if auc_p <= args.p_thresh \
        else f"Not significant at p={args.p_thresh}"
    metadata['AUC NULL 3sd range'] = f"{0.5 - 3 * auc_std} -> {0.5 + 3 * auc_std}"
    pdf = report.create_lr_report(metadata, mia_metrics)
    pdf.output(args.pdf_name, 'F')


def _example(args: Dict) -> None: # pylint: disable = too-many-locals
    '''
    Runs an example attack using data from sklearn
    '''
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.5, stratify=y
    )
    rf = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2)
    rf.fit(train_X, train_y)
    mia_test_probs, mia_test_labels, mia_clf = likelihood_scenario(
        RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, max_depth=10),
        train_X,
        train_y,
        rf.predict_proba(train_X),
        test_X,
        test_y,
        rf.predict_proba(test_X),
        n_shadow_models=args.n_shadow_models
    )
    metadata = {
        'input-data': "Breast Cancer",
        'n_shadow_models': args.n_shadow_models
    }
    mia_metrics = metrics.get_metrics(
        mia_clf,
        mia_test_probs,
        mia_test_labels
    )
    # Check for significance of AUC and PDIF
    pdif = np.exp(-mia_metrics['PDIF'])
    metadata['PDIF_sig'] = f"Significant at p={args.p_thresh}" if pdif <= args.p_thresh \
        else f"Not significant at p={args.p_thresh}"
    auc_std = np.sqrt(
        metrics.expected_auc_var(
            0.5,
            mia_test_labels.sum(),
            len(mia_test_labels) - mia_test_labels.sum()
        )
    )
    auc_p = 1 - norm.cdf(mia_metrics['AUC'], loc=0.5, scale=auc_std)
    metadata['AUC_sig'] = f"Significant at p={args.p_thresh}" if auc_p <= args.p_thresh \
        else f"Not significant at p={args.p_thresh}"
    metadata['AUC NULL 3sd range'] = f"{0.5 - 3 * auc_std} -> {0.5 + 3 * auc_std}"
    pdf = report.create_lr_report(metadata, mia_metrics)
    pdf.output(args.pdf_name, 'F')


def _setup_example_data(_: Any):
    '''
    Method to create example data and save (including config). Intended to allow users
    to see how they would need to setup their own data.
    '''
    X, y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.5, stratify=y
    )
    rf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1)
    rf.fit(train_X, train_y)
    train_data = np.hstack((train_X, train_y[:, None]))
    np.savetxt("train_data.csv", train_data, delimiter=",")

    test_data = np.hstack((test_X, test_y[:, None]))
    np.savetxt("test_data.csv", test_data, delimiter=",")

    train_preds = rf.predict_proba(train_X)
    test_preds = rf.predict_proba(test_X)
    np.savetxt("train_preds.csv", train_preds, delimiter=",")
    np.savetxt("test_preds.csv", test_preds, delimiter=",")


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

def main():
    '''Main method to parse args and invoke relevant code'''
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-s', '--n-shadow-models',
        type=int,
        required=False,
        default=N_SHADOW_MODELS,
        action='store',
        dest='n_shadow_models',
        help=(
            'The number of shadow models to train (default = %(default)d)'
        )
    )
    parser.add_argument('--report-name',
        type=str,
        action="store",
        dest="pdf_name",
        required=False,
        default="lr_report.pdf",
        help=(
            'Output name for the .pdf report. Default = %(default)s'
        )
    )
    parser.add_argument('-p', '--p-thresh',
        type=float,
        action='store',
        dest='p_thresh',
        required=False,
        default=P_THRESH,
        help=(
            'Significance threshold for p-value comparisons. Default = %(default)f'
        )
    )
    subparsers = parser.add_subparsers()
    example_parser = subparsers.add_parser('run-example', parents=[parser])
    example_parser.set_defaults(func=_example)

    attack_parser = subparsers.add_parser('run-attack', parents=[parser])
    attack_parser.add_argument('-j', '--json-file',
        action="store",
        required=True,
        dest="json_file",
        type=str,
        help=(
            'Name of the .json file containing details for the run. Default = %(default)s'
        )
    )
    attack_parser.set_defaults(func=_run_attack)

    example_data_parser = subparsers.add_parser('setup-example-data')
    example_data_parser.set_defaults(func=_setup_example_data)

    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        print("Invalid command. Try --help to get more details")


if __name__ == '__main__':
    main()
