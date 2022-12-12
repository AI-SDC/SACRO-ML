"""
Likelihood testing scenario from https://arxiv.org/pdf/2112.03570.pdf
"""
# pylint: disable = invalid-name
# pylint: disable = too-many-branches

from __future__ import annotations

import argparse
import importlib
import json
import logging
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks import metrics, report
from aisdc.attacks.attack import Attack
from aisdc.attacks.dataset import Data

logging.basicConfig(level=logging.INFO)

N_SHADOW_MODELS = 100  # Number of shadow models that should be trained
EPS = 1e-16  # Used to avoid numerical issues in logit function
P_THRESH = 0.05  # default significance threshold


class DummyClassifier:
    """A Dummy Classifier to allow this code to work with get_metrics"""

    def predict(self, test_X):
        """Return an array of 1/0 depending on value in second column"""
        return 1 * (test_X[:, 1] > 0.5)

    def predict_proba(self, test_X):
        """Simply return the test_X"""
        return test_X


def _logit(p: float) -> float:
    """Standard logit function

    Parameters
    ----------
    p: float
        value to evaluate logit at

    Returns
    -------
    li: float
        logit(p)

    Notes
    -----
    If p is close to 0 or 1, evaluating the log will result in numerical instabilities.
    This code thresholds p at EPS and 1 - EPS where EPS defaults at 1e-16.
    """
    if p > 1 - EPS:
        p = 1 - EPS
    p = max(p, EPS)
    li = np.log(p / (1 - p))
    return li


class LIRAAttackArgs:
    """LIRA Attack arguments"""

    def __init__(self, **kwargs):
        self.__dict__["n_shadow_models"] = N_SHADOW_MODELS
        self.__dict__["p_thresh"] = 0.05
        self.__dict__["report_name"] = None
        self.__dict__["json_file"] = "config.json"
        self.__dict__.update(kwargs)

    def __str__(self):
        return ",".join(
            [f"{str(key)}: {str(value)}" for key, value in self.__dict__.items()]
        )

    def set_param(self, key: Hashable, value: Any) -> None:
        """Set a parameter"""
        self.__dict__[key] = value

    def get_args(self) -> dict:
        """Return arguments"""
        return self.__dict__


class LIRAAttack(Attack):
    """The main LIRA Attack class"""

    def __init__(self, args: LIRAAttackArgs = LIRAAttackArgs()) -> None:
        self.attack_metrics = None
        self.dummy_attack_metrics = None
        self.metadata = None
        self.args = args

    def __str__(self):
        return "LIRA Attack"

    def attack(self, dataset: Data, target_model: sklearn.base.BaseEstimator) -> None:
        """Programmatic attack running
        Runs a LIRA attack from a dataset object and a target model

        Parameters
        ----------
        dataset: attacks.dataset.Data
            Dataset as an instance of the Data class. Needs to have x_train, x_test, y_train
            and y_test set.
        target_mode: sklearn.base.BaseEstimator
            Trained target model. Any class that implements the sklearn.base.BaseEstimator
            interface (i.e. has fit, predict and predict_proba methods)
        """

        shadow_clf = sklearn.base.clone(target_model)

        dataset = self._check_and_update_dataset(dataset, target_model)

        self.run_scenario_from_preds(
            shadow_clf,
            dataset.x_train,
            dataset.y_train,
            target_model.predict_proba(dataset.x_train),
            dataset.x_test,
            dataset.y_test,
            target_model.predict_proba(dataset.x_test),
        )

    def _check_and_update_dataset(
        self, dataset: Data, target_model: sklearn.base.BaseEstimator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Makes sure that it is ok to use the class variables to index the prediction
        arrays. This has two steps:
        1. Replacing the values in y_train with their position in target_model.classes (will
           normally result in no change)
        2. Removing from the test set any rows corresponding to classes that are not in the
           training set.
        """

        y_train_new = []
        classes = list(target_model.classes_)  # pylint: disable = protected-access
        for y in dataset.y_train:
            y_train_new.append(classes.index(y))

        dataset.y_train = np.array(y_train_new, int)
        print(
            f"new ytrain has values and counts: {np.unique(dataset.y_train,return_counts=True)}"
        )
        ok_pos = []
        y_test_new = []
        for i, y in enumerate(dataset.y_test):
            if y in classes:
                ok_pos.append(i)
                y_test_new.append(classes.index(y))

        if len(y_test_new) != len(dataset.x_test):
            dataset.x_test = dataset.x_test[ok_pos, :]
        dataset.y_test = np.array(y_test_new, int)
        print(
            f"new ytest has values and counts: {np.unique(dataset.y_test,return_counts=True)}"
        )

        return dataset

    def run_scenario_from_preds(  # pylint: disable = too-many-statements
        self,
        shadow_clf: sklearn.base.BaseEstimator,
        X_target_train: Iterable[float],
        y_target_train: Iterable[float],
        target_train_preds: Iterable[float],
        X_shadow_train: Iterable[float],
        y_shadow_train: Iterable[float],
        shadow_train_preds: Iterable[float],
    ) -> tuple[np.ndarray, np.ndarray, sklearn.base.BaseEstimator]:

        # def run_scenario_from_preds( # pylint: disable = too-many-locals, too-many-arguments
        #     shadow_clf,
        #     X_target_train: Iterable[float],
        #     y_target_train: Iterable[float],
        #     target_train_preds: Iterable[float],
        #     X_shadow_train: Iterable[float],
        #     y_shadow_train: Iterable[float],
        #     shadow_train_preds: Iterable[float],

        """Implements the likelihood test, using the "offline" version
        See p.6 (top of second column) for details

        Parameters
        ----------
        shadow_clf: sklearn.Model
            An sklearn classifier that will be trained to form the shadow model.
            All hyper-parameters should have been set.
        X_target_train: np.ndarray
            Data that was used to train the target model
        y_target_train: np.ndarray
            Labels that were used to train the target model
        target_train_preds: np.ndarray
            Array of predictions produced by the target model on the training data
        X_shadow_train: np.ndarray
            Data that will be used to train the shadow models
        y_shadow_train: np.ndarray
            Labels that will be used to train the shadow model
        shadow_train_preds: np.ndarray
            Array of predictions produced by the target model on the shadow data


        Returns
        -------
        mia_scores: np.ndarray
            Attack probabilities of belonging to the training set or not
        mia_labels: np.ndarray
            True labels of belonging to the training set or not
        mia_cls: DummyClassifier
            A DummyClassifier that directly returns the scores for compatibility with code
            in metrics.py

        Examples
        --------

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=False)
        >>> train_X, test_X, train_y, test_y = train_test_split(
        >>>   X, y, test_size=0.5, stratify=y
        >>> )
        >>> rf = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2)
        >>> rf.fit(train_X, train_y)
        >>> mia_test_probs, mia_test_labels, mia_clf = likelihood_scenario(
        >>>     RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, max_depth=10),
        >>>     train_X,
        >>>     train_y,
        >>>     rf.predict_proba(train_X),
        >>>     test_X,
        >>>     test_y,
        >>>     rf.predict_proba(test_X),
        >>>     n_shadow_models=100
        >>> )
        """

        logger = logging.getLogger("lr-scenario")

        n_train_rows, _ = X_target_train.shape
        n_shadow_rows, _ = X_shadow_train.shape
        indices = np.arange(0, n_train_rows + n_shadow_rows, 1)

        # Combine taregt and shadow train, from which to sample datasets
        combined_X_train = np.vstack((X_target_train, X_shadow_train))
        combined_y_train = np.hstack((y_target_train, y_shadow_train))

        train_row_to_confidence = {i: [] for i in range(n_train_rows)}
        shadow_row_to_confidence = {i: [] for i in range(n_shadow_rows)}

        # Train N_SHADOW_MODELS shadow models
        logger.info("Training shadow models")
        for model_idx in range(self.args.n_shadow_models):
            if model_idx % 10 == 0:
                logger.info("Trained %d models", model_idx)
            # Pick the indices to use for training this one
            np.random.seed(model_idx)  # Reproducibility
            these_idx = np.random.choice(indices, n_train_rows, replace=False)
            temp_X_train = combined_X_train[these_idx, :]
            temp_y_train = combined_y_train[these_idx]

            # Fit the shadow model
            shadow_clf.set_params(random_state=model_idx)
            shadow_clf.fit(temp_X_train, temp_y_train)

            # map a class to a column
            class_map = {c: i for i, c in enumerate(shadow_clf.classes_)}

            # Get the predicted probabilities on the training data
            confidences = shadow_clf.predict_proba(X_target_train)
            # print(f'shadow clf returned confidences with shape {confidences.shape}')

            these_idx = set(these_idx)
            for i in range(n_train_rows):
                if i not in these_idx:
                    # If i was _not_ used for training, incorporate the logit of its confidence of
                    # being correct - TODO: should we just be taking max??
                    cl_pos = class_map.get(y_target_train[i], -1)
                    # Occasionally, the random data split will result in classes being
                    # absent from the training set. In these cases cl_pos will be -1 and
                    # we include logit(0) instead of discarding (also for the shadow data below)
                    if cl_pos >= 0:
                        train_row_to_confidence[i].append(
                            _logit(confidences[i, cl_pos])
                        )
                    else:  # pragma: no cover
                        # catch-all
                        train_row_to_confidence[i].append(_logit(0))
            # Same process for shadow data
            shadow_confidences = shadow_clf.predict_proba(X_shadow_train)
            for i in range(n_shadow_rows):
                if i + n_train_rows not in these_idx:
                    cl_pos = class_map.get(y_shadow_train[i], -1)
                    if cl_pos >= 0:
                        shadow_row_to_confidence[i].append(
                            _logit(shadow_confidences[i, cl_pos])
                        )
                    else:  # pragma: no cover
                        # catch-all
                        shadow_row_to_confidence[i].append(_logit(0))

        # Do the test described in the paper in each case
        mia_scores = []
        mia_labels = []
        logger.info("Computing scores for train rows")
        for i in range(n_train_rows):
            true_score = _logit(target_train_preds[i, y_target_train[i]])
            null_scores = np.array(train_row_to_confidence[i])
            mean_null = 0.0
            var_null = 0.0
            if not np.isnan(null_scores).all():
                mean_null = np.nanmean(null_scores)  # null_scores.mean()
                var_null = np.nanvar(null_scores)  #
            var_null = max(var_null, EPS)  # var can be zero in some cases
            prob = norm.cdf(true_score, loc=mean_null, scale=np.sqrt(var_null))
            mia_scores.append([1 - prob, prob])
            mia_labels.append(1)

        logger.info("Computing scores for shadow rows")
        for i in range(n_shadow_rows):
            true_score = _logit(shadow_train_preds[i, y_shadow_train[i]])
            null_scores = np.array(shadow_row_to_confidence[i])
            mean_null = null_scores.mean()
            var_null = max(null_scores.var(), EPS)  # var can be zeros in some cases
            prob = norm.cdf(true_score, loc=mean_null, scale=np.sqrt(var_null))
            mia_scores.append([1 - prob, prob])
            mia_labels.append(0)

        mia_clf = DummyClassifier()
        logger.info("Finished scenario")
        self.attack_metrics = [
            metrics.get_metrics(mia_clf, np.array(mia_scores), np.array(mia_labels))
        ]

    def example(self) -> None:  # pylint: disable = too-many-locals
        """Runs an example attack using data from sklearn

        Generates example data, trains a classifier and tuns the attack
        """
        X, y = load_breast_cancer(return_X_y=True, as_frame=False)
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.5, stratify=y
        )
        rf = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2)
        rf.fit(train_X, train_y)
        self.run_scenario_from_preds(
            sklearn.base.clone(rf),
            train_X,
            train_y,
            rf.predict_proba(train_X),
            test_X,
            test_y,
            rf.predict_proba(test_X),
        )

    def _construct_metadata(self) -> None:
        """Constructs the metadata object. Called by the reporting method"""
        self.metadata = {}
        self.metadata["experiment_details"] = {}
        self.metadata["experiment_details"].update(self.args.__dict__)
        if "func" in self.metadata["experiment_details"]:
            del self.metadata["experiment_details"]["func"]

        self.metadata["global_metrics"] = {}

        pdif = np.exp(-self.attack_metrics[0]["PDIF01"])

        self.metadata["global_metrics"]["PDIF_sig"] = (
            f"Significant at p={self.args.p_thresh}"
            if pdif <= self.args.p_thresh
            else f"Not significant at p={self.args.p_thresh}"
        )

        auc_p, auc_std = metrics.auc_p_val(
            self.attack_metrics[0]["AUC"],
            self.attack_metrics[0]["n_pos_test_examples"],
            self.attack_metrics[0]["n_neg_test_examples"],
        )
        self.metadata["global_metrics"]["AUC_sig"] = (
            f"Significant at p={self.args.p_thresh}"
            if auc_p <= self.args.p_thresh
            else f"Not significant at p={self.args.p_thresh}"
        )
        self.metadata["global_metrics"][
            "AUC NULL 3sd range"
        ] = f"{0.5 - 3 * auc_std} -> {0.5 + 3 * auc_std}"

        self.metadata["attack"] = str(self)

    def make_report(self) -> dict:
        """Create the report

        Creates the output report. If self.args.report_name is not None, it will also save the
        information in json and pdf formats

        Returns
        -------

        output: Dict
            Dictionary containing all attack output
        """
        logger = logging.getLogger("reporting")
        output = {}
        logger.info("Starting report, report_name = %s", self.args.report_name)
        output["attack_metrics"] = self.attack_metrics
        self._construct_metadata()
        output["metadata"] = self.metadata
        if self.args.report_name is not None:
            json_report = report.create_json_report(output)
            with open(f"{self.args.report_name}.json", "w", encoding="utf-8") as f:
                f.write(json_report)
            logger.info("Wrote report to %s", f"{self.args.report_name}.json")

            pdf = report.create_lr_report(output)
            pdf.output(f"{self.args.report_name}.pdf", "F")
            logger.info("Wrote pdf report to %s", f"{self.args.report_name}.pdf")
        return output

    def setup_example_data(self) -> None:
        """Method to create example data and save (including config). Intended to allow users
        to see how they would need to setup their own data.

        Generates train and test data .csv files, train and test predictions .csv files and
        a config.json file that can be used to run the attack from the command line.
        """
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
            "target_hyppars": {"min_samples_split": 2, "min_samples_leaf": 1},
        }

        with open("config.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(config))

    def attack_from_config(self) -> None:  # pylint: disable = too-many-locals
        """Runs an attack based on the args parsed from the command line"""
        logger = logging.getLogger("run-attack")
        logger.info("Reading config from %s", self.args.json_file)
        with open(self.args.json_file, encoding="utf-8") as f:
            config = json.loads(f.read())

        logger.info("Loading training data csv from %s", config["training_data_file"])
        training_data = np.loadtxt(config["training_data_file"], delimiter=",")
        train_X = training_data[:, :-1]
        train_y = training_data[:, -1].flatten().astype(int)
        logger.info("Loaded %d rows", len(train_X))

        logger.info("Loading test data csv from %s", config["testing_data_file"])
        test_data = np.loadtxt(config["testing_data_file"], delimiter=",")
        test_X = test_data[:, :-1]
        test_y = test_data[:, -1].flatten().astype(int)
        logger.info("Loaded %d rows", len(test_X))

        logger.info("Loading train predictions form %s", config["training_preds_file"])
        train_preds = np.loadtxt(config["training_preds_file"], delimiter=",")
        assert len(train_preds) == len(train_X)

        logger.info("Loading test predictions form %s", config["testing_preds_file"])
        test_preds = np.loadtxt(config["testing_preds_file"], delimiter=",")
        assert len(test_preds) == len(test_X)

        clf_module_name, clf_class_name = config["target_model"]
        module = importlib.import_module(clf_module_name)
        clf_class = getattr(module, clf_class_name)
        clf_params = config["target_hyppars"]
        clf = clf_class(**clf_params)
        logger.info("Created model: %s", str(clf))
        self.run_scenario_from_preds(
            clf, train_X, train_y, train_preds, test_X, test_y, test_preds
        )
        logger.info("Computing metrics")


# Methods invoked by command line script
def _setup_example_data(args):
    """Call the methods to setup some example data"""
    lira_args = LIRAAttackArgs(**args.__dict__)
    attack_obj = LIRAAttack(lira_args)
    attack_obj.setup_example_data()


def _example(args):
    """Call the methods to run an example"""
    lira_args = LIRAAttackArgs(**args.__dict__)
    attack_obj = LIRAAttack(lira_args)
    attack_obj.example()
    attack_obj.make_report()


def _run_attack(args):
    """Run a command line attack based on saved files described in .json file"""
    lira_args = LIRAAttackArgs(**args.__dict__)
    attack_obj = LIRAAttack(lira_args)
    attack_obj.attack_from_config()
    attack_obj.make_report()


def main():
    """Main method to parse args and invoke relevant code"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-s",
        "--n-shadow-models",
        type=int,
        required=False,
        default=N_SHADOW_MODELS,
        action="store",
        dest="n_shadow_models",
        help=("The number of shadow models to train (default = %(default)d)"),
    )
    parser.add_argument(
        "--report-name",
        type=str,
        action="store",
        dest="report_name",
        required=False,
        default="lr_report",
        help=("Output name for the report. Default = %(default)s"),
    )
    parser.add_argument(
        "-p",
        "--p-thresh",
        type=float,
        action="store",
        dest="p_thresh",
        required=False,
        default=P_THRESH,
        help=("Significance threshold for p-value comparisons. Default = %(default)f"),
    )
    subparsers = parser.add_subparsers()
    example_parser = subparsers.add_parser("run-example", parents=[parser])
    example_parser.set_defaults(func=_example)

    attack_parser = subparsers.add_parser("run-attack", parents=[parser])
    attack_parser.add_argument(
        "-j",
        "--json-file",
        action="store",
        required=True,
        dest="json_file",
        type=str,
        help=(
            "Name of the .json file containing details for the run. Default = %(default)s"
        ),
    )
    attack_parser.set_defaults(func=_run_attack)

    example_data_parser = subparsers.add_parser("setup-example-data")
    example_data_parser.set_defaults(func=_setup_example_data)

    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError as e:  # pragma:no cover
        print(e)
        print("Invalid command. Try --help to get more details")


if __name__ == "__main__":  # pragma:no cover
    main()
