"""Likelihood testing scenario from https://arxiv.org/pdf/2112.03570.pdf."""

# pylint: disable = invalid-name
# pylint: disable = too-many-branches

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import uuid
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc import metrics
from aisdc.attacks import report
from aisdc.attacks.attack import Attack
from aisdc.attacks.attack_report_formatter import GenerateJSONModule
from aisdc.attacks.target import Target

logging.basicConfig(level=logging.INFO)

N_SHADOW_MODELS = 100  # Number of shadow models that should be trained
EPS = 1e-16  # Used to avoid numerical issues in logit function
P_THRESH = 0.05  # default significance threshold


class DummyClassifier:
    """A Dummy Classifier to allow this code to work with get_metrics."""

    def predict(self, test_X):
        """Return an array of 1/0 depending on value in second column."""
        return 1 * (test_X[:, 1] > 0.5)

    def predict_proba(self, test_X):
        """Simply return the test_X."""
        return test_X


def _logit(p: float) -> float:
    """Standard logit function.

    Parameters
    ----------
    p : float
        value to evaluate logit at

    Returns
    -------
    li : float
        logit(p)

    Notes
    -----
    If p is close to 0 or 1, evaluating the log will result in numerical instabilities.
    This code thresholds p at EPS and 1 - EPS where EPS defaults at 1e-16.
    """
    if p > 1 - EPS:  # pylint:disable=consider-using-min-builtin
        p = 1 - EPS
    p = max(p, EPS)
    li = np.log(p / (1 - p))
    return li


class LIRAAttack(Attack):
    """The main LIRA Attack class."""

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable = too-many-arguments
        self,
        n_shadow_models: int = 100,
        p_thresh: float = 0.05,
        output_dir: str = "outputs_lira",
        report_name: str = "report_lira",
        training_data_filename: str = None,
        test_data_filename: str = None,
        training_preds_filename: str = None,
        test_preds_filename: str = None,
        target_model: list = None,
        target_model_hyp: dict = None,
        attack_config_json_file_name: str = None,
        n_shadow_rows_confidences_min: int = 10,
        shadow_models_fail_fast: bool = False,
        target_path: str = None,
    ) -> None:
        """Constructs an object to execute a LIRA attack.

        Parameters
        ----------
        n_shadow_models : int
            number of shadow models to be trained
        p_thresh : float
            threshold to determine significance of things. For instance auc_p_value and pdif_vals
        output_dir : str
            name of the directory where outputs are stored
        report_name : str
            name of the pdf and json output reports
        training_data_filename : str
            name of the data file for the training data (in-sample)
        test_data_filename : str
            name of the file for the test data (out-of-sample)
        training_preds_filename : str
            name of the file to keep predictions of the training data (in-sample)
        test_preds_filename : str
            name of the file to keep predictions of the test data (out-of-sample)
        target_model : list
            name of the module (i.e. classification module name such as 'sklearn.ensemble') and
            attack model name (i.e. classification model name such as 'RandomForestClassifier')
        target_model_hyp : dict
            dictionary of hyper parameters for the target_model
            such as min_sample_split, min_samples_leaf etc
        attack_config_json_file_name : str
            name of the configuration file to load parameters
        n_shadow_rows_confidences_min : int
            number of minimum number of confidences calculated for
            each row in test data (out-of-sample)
        shadow_models_fail_fast : bool
            If true it stops repetitions earlier based on the given minimum
            number of confidences for each row in the test data
        target_path : str
            path to the saved trained target model and target data
        """
        super().__init__()
        self.n_shadow_models = n_shadow_models
        self.p_thresh = p_thresh
        self.output_dir = output_dir
        self.report_name = report_name
        self.training_data_filename = training_data_filename
        self.test_data_filename = test_data_filename
        self.training_preds_filename = training_preds_filename
        self.test_preds_filename = test_preds_filename
        self.target_model = target_model
        self.target_model_hyp = target_model_hyp
        self.attack_config_json_file_name = attack_config_json_file_name
        self.n_shadow_rows_confidences_min = n_shadow_rows_confidences_min
        self.shadow_models_fail_fast = shadow_models_fail_fast
        self.target_path = target_path
        if self.attack_config_json_file_name is not None:
            self._update_params_from_config_file()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.attack_metrics = None
        self.attack_failfast_shadow_models_trained = None
        self.metadata = None

    def __str__(self):
        return "LIRA Attack"

    def attack(self, target: Target) -> None:
        """Programmatic attack running
        Runs a LIRA attack from a Target object and a target model.

        Parameters
        ----------
        target : attacks.target.Target
            target as an instance of the Target class. Needs to have x_train,
            x_test, y_train and y_test set.
        """

        shadow_clf = sklearn.base.clone(target.model)

        target = self._check_and_update_dataset(target)

        self.run_scenario_from_preds(
            shadow_clf,
            target.x_train,
            target.y_train,
            target.model.predict_proba(target.x_train),
            target.x_test,
            target.y_test,
            target.model.predict_proba(target.x_test),
        )

    def _check_and_update_dataset(self, target: Target) -> Target:
        """
        Makes sure that it is ok to use the class variables to index the
        prediction arrays. This has two steps:
        1. Replacing the values in y_train with their position in
        target.model.classes (will normally result in no change)
        2. Removing from the test set any rows corresponding to classes that
        are not in the training set.
        """
        logger = logging.getLogger("_check_and_update_dataset")
        y_train_new = []
        classes = list(target.model.classes_)
        for y in target.y_train:
            y_train_new.append(classes.index(y))

        target.y_train = np.array(y_train_new, int)

        logger.info(
            "new ytrain has values and counts: %s",
            f"{np.unique(target.y_train,return_counts=True)}",
        )
        ok_pos = []
        y_test_new = []
        for i, y in enumerate(target.y_test):
            if y in classes:
                ok_pos.append(i)
                y_test_new.append(classes.index(y))

        if len(y_test_new) != len(target.x_test):
            target.x_test = target.x_test[ok_pos, :]
        target.y_test = np.array(y_test_new, int)
        logger.info(
            "new ytest has values and counts: %s",
            f"{np.unique(target.y_test,return_counts=True)}",
        )

        return target

    def run_scenario_from_preds(  # pylint: disable = too-many-statements, too-many-arguments, too-many-locals
        self,
        shadow_clf: sklearn.base.BaseEstimator,
        X_target_train: Iterable[float],
        y_target_train: Iterable[float],
        target_train_preds: Iterable[float],
        X_shadow_train: Iterable[float],
        y_shadow_train: Iterable[float],
        shadow_train_preds: Iterable[float],
    ) -> tuple[np.ndarray, np.ndarray, sklearn.base.BaseEstimator]:
        """Implements the likelihood test, using the "offline" version
        See p.6 (top of second column) for details.

        Parameters
        ----------
        shadow_clf : sklearn.Model
            An sklearn classifier that will be trained to form the shadow model.
            All hyper-parameters should have been set.
        X_target_train : np.ndarray
            Data that was used to train the target model
        y_target_train : np.ndarray
            Labels that were used to train the target model
        target_train_preds : np.ndarray
            Array of predictions produced by the target model on the training data
        X_shadow_train : np.ndarray
            Data that will be used to train the shadow models
        y_shadow_train : np.ndarray
            Labels that will be used to train the shadow model
        shadow_train_preds : np.ndarray
            Array of predictions produced by the target model on the shadow data

        Returns
        -------
        mia_scores : np.ndarray
            Attack probabilities of belonging to the training set or not
        mia_labels : np.ndarray
            True labels of belonging to the training set or not
        mia_cls : DummyClassifier
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
        for model_idx in range(self.n_shadow_models):
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

            # Compute number of confidences for each row
            lengths_shadow_row_to_confidence = {
                key: len(value) for key, value in shadow_row_to_confidence.items()
            }
            n_shadow_confidences = self.n_shadow_rows_confidences_min
            # Stop training of shadow models when shadow_model_fail_fast is True
            # and a minimum number of confidences specified by parameter
            # (n_shadow_rows_confidences_min) are computed for each row
            if (
                not any(
                    value < n_shadow_confidences
                    for value in lengths_shadow_row_to_confidence.values()
                )
                and self.shadow_models_fail_fast
            ):
                break
        self.attack_failfast_shadow_models_trained = model_idx + 1
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

        mia_scores = np.array(mia_scores)
        mia_labels = np.array(mia_labels)
        y_pred_proba, y_test = metrics.get_probabilities(
            mia_clf, mia_scores, mia_labels, permute_rows=True
        )
        self.attack_metrics = [metrics.get_metrics(y_pred_proba, y_test)]

    def example(self) -> None:
        """Runs an example attack using data from sklearn.

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
        """Constructs the metadata object. Called by the reporting method."""
        self.metadata = {}
        self.metadata["experiment_details"] = {}
        self.metadata["experiment_details"] = self.get_params()

        self.metadata["global_metrics"] = {}

        pdif = np.exp(-self.attack_metrics[0]["PDIF01"])

        self.metadata["global_metrics"]["PDIF_sig"] = (
            f"Significant at p={self.p_thresh}"
            if pdif <= self.p_thresh
            else f"Not significant at p={self.p_thresh}"
        )

        auc_p, auc_std = metrics.auc_p_val(
            self.attack_metrics[0]["AUC"],
            self.attack_metrics[0]["n_pos_test_examples"],
            self.attack_metrics[0]["n_neg_test_examples"],
        )
        self.metadata["global_metrics"]["AUC_sig"] = (
            f"Significant at p={self.p_thresh}"
            if auc_p <= self.p_thresh
            else f"Not significant at p={self.p_thresh}"
        )
        self.metadata["global_metrics"][
            "null_auc_3sd_range"
        ] = f"{0.5 - 3 * auc_std} -> {0.5 + 3 * auc_std}"

        self.metadata["attack"] = str(self)

    def make_report(self) -> dict:
        """Create the report.

        Creates the output report. If self.args.report_name is not None, it will also save the
        information in json and pdf formats

        Returns
        -------

        output : Dict
            Dictionary containing all attack output
        """
        logger = logging.getLogger("reporting")
        report_dest = os.path.join(self.output_dir, self.report_name)
        logger.info(
            "Starting reports, pdf report name = %s, json report name = %s",
            report_dest + ".pdf",
            report_dest + ".json",
        )
        output = {}
        output["log_id"] = str(uuid.uuid4())
        output["log_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self._construct_metadata()
        output["metadata"] = self.metadata
        output["attack_experiment_logger"] = self._get_attack_metrics_instances()

        json_attack_formatter = GenerateJSONModule(report_dest + ".json")
        json_report = report.create_json_report(output)
        json_attack_formatter.add_attack_output(json_report, "LikelihoodAttack")

        pdf_report = report.create_lr_report(output)
        report.add_output_to_pdf(report_dest, pdf_report, "LikelihoodAttack")
        logger.info(
            "Wrote pdf report to %s and json report to %s",
            report_dest + ".pdf",
            report_dest + ".json",
        )

        return output

    def _get_attack_metrics_instances(self) -> dict:
        """Constructs the metadata object, after attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}

        for rep, _ in enumerate(self.attack_metrics):
            self.attack_metrics[rep][
                "n_shadow_models_trained"
            ] = self.attack_failfast_shadow_models_trained
            attack_metrics_instances["instance_" + str(rep)] = self.attack_metrics[rep]

        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        return attack_metrics_experiment

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
            "training_data_filename": "train_data.csv",
            "test_data_filename": "test_data.csv",
            "training_preds_filename": "train_preds.csv",
            "test_preds_filename": "test_preds.csv",
            "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
            "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
        }

        with open("config.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(config))

    def attack_from_config(self) -> None:  # pylint: disable = too-many-locals
        """Runs an attack based on the args parsed from the command line."""
        logger = logging.getLogger("run-attack")
        logger.info("Loading training data csv from %s", self.training_data_filename)
        training_data = np.loadtxt(self.training_data_filename, delimiter=",")
        train_X = training_data[:, :-1]
        train_y = training_data[:, -1].flatten().astype(int)
        logger.info("Loaded %d rows", len(train_X))

        logger.info("Loading test data csv from %s", self.test_data_filename)
        test_data = np.loadtxt(self.test_data_filename, delimiter=",")
        test_X = test_data[:, :-1]
        test_y = test_data[:, -1].flatten().astype(int)
        logger.info("Loaded %d rows", len(test_X))

        logger.info("Loading train predictions form %s", self.training_preds_filename)
        train_preds = np.loadtxt(self.training_preds_filename, delimiter=",")
        assert len(train_preds) == len(train_X)

        logger.info("Loading test predictions form %s", self.test_preds_filename)
        test_preds = np.loadtxt(self.test_preds_filename, delimiter=",")
        assert len(test_preds) == len(test_X)
        if self.target_model is not None:
            clf_module_name, clf_class_name = self.target_model
            module = importlib.import_module(clf_module_name)
            clf_class = getattr(module, clf_class_name)
            if self.target_model_hyp is not None:
                clf_params = self.target_model_hyp
                clf = clf_class(**clf_params)
        logger.info("Created model: %s", str(clf))
        self.run_scenario_from_preds(
            clf, train_X, train_y, train_preds, test_X, test_y, test_preds
        )
        logger.info("Computing metrics")


# Methods invoked by command line script
def _setup_example_data(args):
    """Call the methods to setup some example data."""
    attack_obj = LIRAAttack(
        n_shadow_models=args.n_shadow_models,
        n_shadow_rows_confidences_min=args.n_shadow_rows_confidences_min,
        output_dir=args.output_dir,
        report_name=args.report_name,
        p_thresh=args.p_thresh,
        shadow_models_fail_fast=args.shadow_models_fail_fast,
    )
    attack_obj.setup_example_data()


def _example(args):
    """Call the methods to run an example."""
    attack_obj = LIRAAttack(
        n_shadow_models=args.n_shadow_models,
        n_shadow_rows_confidences_min=args.n_shadow_rows_confidences_min,
        output_dir=args.output_dir,
        report_name=args.report_name,
        p_thresh=args.p_thresh,
        shadow_models_fail_fast=args.shadow_models_fail_fast,
    )
    attack_obj.example()
    attack_obj.make_report()


def _run_attack(args):
    """Run a command line attack based on saved files described in .json file."""
    # attack_obj = LIRAAttack(**args.__dict__)
    attack_obj = LIRAAttack(
        n_shadow_models=args.n_shadow_models,
        n_shadow_rows_confidences_min=args.n_shadow_rows_confidences_min,
        p_thresh=args.p_thresh,
        output_dir=args.output_dir,
        report_name=args.report_name,
        shadow_models_fail_fast=args.shadow_models_fail_fast,
        attack_config_json_file_name=args.attack_config_json_file_name,
    )
    attack_obj.attack_from_config()
    attack_obj.make_report()


def _run_attack_from_configfile(args):
    """Run a command line attack based on saved files described in .json file."""
    attack_obj = LIRAAttack(
        attack_config_json_file_name=args.attack_config_json_file_name,
        target_path=str(args.target_path),
    )
    print(args.attack_config_json_file_name)
    target = Target()
    target.load(attack_obj.target_path)
    attack_obj.attack(target)
    attack_obj.make_report()


def main():
    """Main method to parse args and invoke relevant code."""
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
        "--n-shadow-rows-confidences-min",
        type=int,
        action="store",
        dest="n_shadow_rows_confidences_min",
        default=10,
        required=False,
        help=(
            """Number of confidences against rows in shadow data from the shadow models
            and works when --shadow-models-fail-fast = True. Default = %(default)d"""
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        action="store",
        dest="output_dir",
        default="output_lira",
        required=False,
        help=("Directory name where output files are stored. Default = %(default)s."),
    )

    parser.add_argument(
        "--report-name",
        type=str,
        action="store",
        dest="report_name",
        default="report_lira",
        required=False,
        help=(
            """Filename for the pdf and json output reports. Default = %(default)s.
            Code will append .pdf and .json"""
        ),
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

    parser.add_argument(
        "--shadow-models-fail-fast",
        action="store_true",
        required=False,
        dest="shadow_models_fail_fast",
        help=(
            """To stop training shadow models early based on minimum number of
            confidences across all rows (--n-shadow-rows-confidences-min)
            in the shadow data. Default = %(default)s"""
        ),
    )

    subparsers = parser.add_subparsers()
    example_parser = subparsers.add_parser("run-example", parents=[parser])
    example_parser.set_defaults(func=_example)

    attack_parser = subparsers.add_parser("run-attack", parents=[parser])
    attack_parser.add_argument(
        "-j",
        "--attack-config-json-file-name",
        action="store",
        required=True,
        dest="attack_config_json_file_name",
        type=str,
        help=(
            "Name of the .json file containing details for the run. Default = %(default)s"
        ),
    )
    attack_parser.set_defaults(func=_run_attack)

    attack_parser_config = subparsers.add_parser("run-attack-from-configfile")
    attack_parser_config.add_argument(
        "-j",
        "--attack-config-json-file-name",
        action="store",
        required=True,
        dest="attack_config_json_file_name",
        type=str,
        default="config_lira_cmd.json",
        help=(
            "Name of the .json file containing details for the run. Default = %(default)s"
        ),
    )

    attack_parser_config.add_argument(
        "-t",
        "--attack-target-folder-path",
        action="store",
        required=True,
        dest="target_path",
        type=str,
        default="lira_target",
        help=(
            """Name of the target directory to load the trained target model and the target data.
            Default = %(default)s"""
        ),
    )

    attack_parser_config.set_defaults(func=_run_attack_from_configfile)

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
