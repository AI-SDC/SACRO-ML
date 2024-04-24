"""
Worst_case_attack.py.

Runs a worst case attack based upon predictive probabilities stored in two .csv files
"""  # pylint: disable = too-many-lines

from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from aisdc import metrics
from aisdc.attacks import report
from aisdc.attacks.attack import Attack
from aisdc.attacks.attack_report_formatter import GenerateJSONModule
from aisdc.attacks.failfast import FailFast
from aisdc.attacks.target import Target

logging.basicConfig(level=logging.INFO)

P_THRESH = 0.05


class WorstCaseAttack(Attack):
    """Class to wrap the worst case attack code."""

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable = too-many-arguments, too-many-locals, too-many-statements
        self,
        n_reps: int = 10,
        reproduce_split: int | Iterable[int] | None = 5,
        p_thresh: float = 0.05,
        n_dummy_reps: int = 1,
        train_beta: int = 1,
        test_beta: int = 1,
        test_prop: float = 0.2,
        n_rows_in: int = 1000,
        n_rows_out: int = 1000,
        training_preds_filename: str = None,
        test_preds_filename: str = None,
        output_dir: str = "output_worstcase",
        report_name: str = "report_worstcase",
        include_model_correct_feature: bool = False,
        sort_probs: bool = True,
        mia_attack_model: Any = RandomForestClassifier,
        mia_attack_model_hyp: dict = None,
        attack_metric_success_name: str = "P_HIGHER_AUC",
        attack_metric_success_thresh: float = 0.05,
        attack_metric_success_comp_type: str = "lte",
        attack_metric_success_count_thresh: int = 5,
        attack_fail_fast: bool = False,
        attack_config_json_file_name: str = None,
        target_path: str = None,
    ) -> None:
        """Constructs an object to execute a worst case attack.

        Parameters
        ----------
        n_reps : int
            number of attacks to run -- in each iteration an attack model
            is trained on a different subset of the data
        reproduce_split : int
            variable that controls the reproducibility of the data split.
            It can be an integer or a list of integers of length n_reps. Default : 5.
        p_thresh : float
            threshold to determine significance of things. For instance auc_p_value and pdif_vals
        n_dummy_reps : int
            number of baseline (dummy) experiments to do
        train_beta : int
            value of b for beta distribution used to sample the in-sample (training) probabilities
        test_beta : int
            value of b for beta distribution used to sample the out-of-sample (test) probabilities
        test_prop : float
            proportion of data to use as a test set for the attack model
        n_rows_in : int
            number of rows for in-sample (training data)
        n_rows_out : int
            number of rows for out-of-sample (test data)
        training_preds_filename : str
            name of the file to keep predictions of the training data (in-sample)
        test_preds_filename : str
            name of the file to keep predictions of the test data (out-of-sample)
        output_dir : str
            name of the directory where outputs are stored
        report_name : str
            name of the pdf and json output reports
        include_model_correct_feature : bool
            inclusion of additional feature to hold whether or not the target model
            made a correct prediction for each example
        sort_probs : bool
            true in case require to sort combine preds (from training and test)
            to have highest probabilities in the first column
        mia_attack_model : Any
            name of the attack model such as RandomForestClassifier
        mia_attack_model_hyp : dict
            dictionary of hyper parameters for the mia_attack_model
            such as min_sample_split, min_samples_leaf etc
        attack_metric_success_name : str
            name of metric to compute for the attack being successful
        attack_metric_success_thresh : float
            threshold for a given metric to measure attack being successful or not
        attack_metric_success_comp_type : str
            threshold comparison operator (i.e., gte: greater than or equal to, gt:
            greater than, lte: less than or equal to, lt: less than,
            eq: equal to and not_eq: not equal to)
        attack_metric_success_count_thresh : int
            a counter to record how many times an attack was successful
            given that the threshold has fulfilled criteria for a given comparison type
        attack_fail_fast : bool
            If true it stops repetitions earlier based on the given attack metric
            (i.e., attack_metric_success_name) considering the comparison type
            (attack_metric_success_comp_type) satisfying a threshold
            (i.e., attack_metric_success_thresh) for n
            (attack_metric_success_count_thresh) number of times
        attack_config_json_file_name : str
            name of the configuration file to load parameters
        target_path : str
            path to the saved trained target model and target data
        """

        super().__init__()
        self.n_reps = n_reps
        self.reproduce_split = reproduce_split
        if isinstance(reproduce_split, int):
            reproduce_split = [reproduce_split] + [
                x**2 for x in range(reproduce_split, reproduce_split + n_reps - 1)
            ]
        else:
            reproduce_split = list(
                dict.fromkeys(reproduce_split)
            )  # remove potential duplicates
            if len(reproduce_split) == n_reps:
                pass
            elif len(reproduce_split) > n_reps:
                print("split", reproduce_split, "nreps", n_reps)
                reproduce_split = list(reproduce_split)[0:n_reps]
                print(
                    "WARNING: the length of the parameter 'reproduce_split'\
                     is longer than n_reps. Values have been removed."
                )
            else:
                # assign values to match length of n_reps
                reproduce_split += [
                    reproduce_split[-1] * x
                    for x in range(2, (n_reps - len(reproduce_split) + 2))
                ]
                print(
                    "WARNING: the length of the parameter 'reproduce_split'\
                     is shorter than n_reps. Vales have been added."
                )
            print("reproduce split now", reproduce_split)
        self.reproduce_split = reproduce_split
        self.p_thresh = p_thresh
        self.n_dummy_reps = n_dummy_reps
        self.train_beta = train_beta
        self.test_beta = test_beta
        self.test_prop = test_prop
        self.n_rows_in = n_rows_in
        self.n_rows_out = n_rows_out
        self.training_preds_filename = training_preds_filename
        self.test_preds_filename = test_preds_filename
        self.output_dir = output_dir
        self.report_name = report_name
        self.include_model_correct_feature = include_model_correct_feature
        self.sort_probs = sort_probs
        self.mia_attack_model = mia_attack_model
        if mia_attack_model_hyp is None:
            self.mia_attack_model_hyp = {
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_depth": 5,
            }
        else:
            self.mia_attack_model_hyp = mia_attack_model_hyp
        self.attack_metric_success_name = attack_metric_success_name
        self.attack_metric_success_thresh = attack_metric_success_thresh
        self.attack_metric_success_comp_type = attack_metric_success_comp_type
        self.attack_metric_success_count_thresh = attack_metric_success_count_thresh
        self.attack_fail_fast = attack_fail_fast
        self.attack_config_json_file_name = attack_config_json_file_name
        self.target_path = target_path
        # Updating parameters from a configuration json file
        if self.attack_config_json_file_name is not None:
            self._update_params_from_config_file()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.attack_metrics = None
        self.attack_metric_failfast_summary = None
        self.dummy_attack_metrics = None
        self.dummy_attack_metric_failfast_summary = None
        self.metadata = None

    def __str__(self):
        return "WorstCase attack"

    def attack(self, target: Target) -> None:
        """Programmatic attack entry point.

        To be used when code has access to Target class and trained target model

        Parameters
        ----------
        target : attacks.target.Target
            target as a Target class object
        """
        train_preds = target.model.predict_proba(target.x_train)
        test_preds = target.model.predict_proba(target.x_test)
        train_correct = None
        test_correct = None
        if self.include_model_correct_feature:
            train_correct = 1 * (target.y_train == target.model.predict(target.x_train))
            test_correct = 1 * (target.y_test == target.model.predict(target.x_test))

        self.attack_from_preds(
            train_preds,
            test_preds,
            train_correct=train_correct,
            test_correct=test_correct,
        )

    def attack_from_prediction_files(self):
        """Start an attack from saved prediction files.

        To be used when only saved predictions are available.

        Filenames for the saved prediction files to be specified in the arguments provided
        in the constructor
        """
        train_preds = np.loadtxt(self.training_preds_filename, delimiter=",")
        test_preds = np.loadtxt(self.test_preds_filename, delimiter=",")
        self.attack_from_preds(train_preds, test_preds)

    def attack_from_preds(
        self,
        train_preds: np.ndarray,
        test_preds: np.ndarray,
        train_correct: np.ndarray = None,
        test_correct: np.ndarray = None,
    ) -> None:
        """
        Runs the attack based upon the predictions in train_preds and test_preds, and the params
        stored in self.args.

        Parameters
        ----------
        train_preds : np.ndarray
            Array of train predictions. One row per example, one column per class (i.e. 2)
        test_preds : np.ndarray
            Array of test predictions. One row per example, one column per class (i.e. 2)
        """
        logger = logging.getLogger("attack-from-preds")
        logger.info("Running main attack repetitions")
        attack_metric_dict = self.run_attack_reps(
            train_preds,
            test_preds,
            train_correct=train_correct,
            test_correct=test_correct,
        )
        self.attack_metrics = attack_metric_dict["mia_metrics"]
        self.attack_metric_failfast_summary = attack_metric_dict[
            "failfast_metric_summary"
        ]

        self.dummy_attack_metrics = []
        self.dummy_attack_metric_failfast_summary = []
        if self.n_dummy_reps > 0:
            logger.info("Running dummy attack reps")
            n_train_rows = len(train_preds)
            n_test_rows = len(test_preds)
            for _ in range(self.n_dummy_reps):
                d_train_preds, d_test_preds = self.generate_arrays(
                    n_train_rows,
                    n_test_rows,
                    self.train_beta,
                    self.test_beta,
                )
                temp_attack_metric_dict = self.run_attack_reps(
                    d_train_preds, d_test_preds
                )
                temp_metrics = temp_attack_metric_dict["mia_metrics"]
                temp_metric_failfast_summary = temp_attack_metric_dict[
                    "failfast_metric_summary"
                ]

                self.dummy_attack_metrics.append(temp_metrics)
                self.dummy_attack_metric_failfast_summary.append(
                    temp_metric_failfast_summary
                )

        logger.info("Finished running attacks")

    def _prepare_attack_data(
        self,
        train_preds: np.ndarray,
        test_preds: np.ndarray,
        train_correct: np.ndarray = None,
        test_correct: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data and labels for attack model
        Combines the train and test preds into a single numpy array (optionally) sorting each
        row to have the highest probabilities in the first column. Constructs a label array that
        has ones corresponding to training rows and zeros to testing rows.
        """
        logger = logging.getLogger("prep-attack-data")
        if self.sort_probs:
            logger.info("Sorting probabilities to leave highest value in first column")
            train_preds = -np.sort(-train_preds, axis=1)
            test_preds = -np.sort(-test_preds, axis=1)

        logger.info("Creating MIA data")

        if self.include_model_correct_feature and train_correct is not None:
            train_preds = np.hstack((train_preds, train_correct[:, None]))
            test_preds = np.hstack((test_preds, test_correct[:, None]))

        mi_x = np.vstack((train_preds, test_preds))
        mi_y = np.hstack((np.ones(len(train_preds)), np.zeros(len(test_preds))))

        return (mi_x, mi_y)

    def run_attack_reps(  # pylint: disable = too-many-locals
        self,
        train_preds: np.ndarray,
        test_preds: np.ndarray,
        train_correct: np.ndarray = None,
        test_correct: np.ndarray = None,
    ) -> dict:
        """
        Run actual attack reps from train and test predictions.

        Parameters
        ----------
        train_preds : np.ndarray
            predictions from the model on training (in-sample) data
        test_preds : np.ndarray
            predictions from the model on testing (out-of-sample) data

        Returns
        -------
        mia_metrics_dict : dict
            a dictionary with two items including mia_metrics
            (a list of metric across repetitions) and failfast_metric_summary object
            (an object of FailFast class) to maintain summary of
            fail/success of attacks for a given metric of failfast option
        """
        self.n_rows_in = len(train_preds)
        self.n_rows_out = len(test_preds)
        logger = logging.getLogger("attack-reps")
        mi_x, mi_y = self._prepare_attack_data(
            train_preds, test_preds, train_correct, test_correct
        )

        mia_metrics = []

        failfast_metric_summary = FailFast(self)

        for rep in range(self.n_reps):
            logger.info(
                "Rep %d of %d split %d", rep + 1, self.n_reps, self.reproduce_split[rep]
            )
            mi_train_x, mi_test_x, mi_train_y, mi_test_y = train_test_split(
                mi_x,
                mi_y,
                test_size=self.test_prop,
                stratify=mi_y,
                random_state=self.reproduce_split[rep],
                shuffle=True,
            )
            attack_classifier = self.mia_attack_model(**self.mia_attack_model_hyp)
            attack_classifier.fit(mi_train_x, mi_train_y)
            y_pred_proba, y_test = metrics.get_probabilities(
                attack_classifier, mi_test_x, mi_test_y, permute_rows=True
            )

            mia_metrics.append(metrics.get_metrics(y_pred_proba, y_test))

            if self.include_model_correct_feature and train_correct is not None:
                # Compute the Yeom TPR and FPR
                yeom_preds = mi_test_x[:, -1]
                tn, fp, fn, tp = confusion_matrix(mi_test_y, yeom_preds).ravel()
                mia_metrics[-1]["yeom_tpr"] = tp / (tp + fn)
                mia_metrics[-1]["yeom_fpr"] = fp / (fp + tn)
                mia_metrics[-1]["yeom_advantage"] = (
                    mia_metrics[-1]["yeom_tpr"] - mia_metrics[-1]["yeom_fpr"]
                )

            failfast_metric_summary.check_attack_success(mia_metrics[rep])

            if (
                failfast_metric_summary.check_overall_attack_success(self)
                and self.attack_fail_fast
            ):
                break

        logger.info("Finished simulating attacks")

        mia_metrics_dict = {}
        mia_metrics_dict["mia_metrics"] = mia_metrics
        mia_metrics_dict["failfast_metric_summary"] = failfast_metric_summary

        return mia_metrics_dict

    def _get_global_metrics(self, attack_metrics: list) -> dict:
        """Summarise metrics from a metric list.

        Returns
        -------
        global_metrics : Dict
            Dictionary of summary metrics

        Arguments
        ---------
        attack_metrics: List
            list of attack metrics dictionaries
        """
        global_metrics = {}
        if attack_metrics is not None and len(attack_metrics) != 0:
            auc_p_vals = [
                metrics.auc_p_val(
                    m["AUC"], m["n_pos_test_examples"], m["n_neg_test_examples"]
                )[0]
                for m in attack_metrics
            ]

            m = attack_metrics[0]
            _, auc_std = metrics.auc_p_val(
                0.5, m["n_pos_test_examples"], m["n_neg_test_examples"]
            )

            global_metrics["null_auc_3sd_range"] = (
                f"{0.5 - 3*auc_std:.4f} -> {0.5 + 3*auc_std:.4f}"
            )
            global_metrics["n_sig_auc_p_vals"] = self._get_n_significant(
                auc_p_vals, self.p_thresh
            )
            global_metrics["n_sig_auc_p_vals_corrected"] = self._get_n_significant(
                auc_p_vals, self.p_thresh, bh_fdr_correction=True
            )

            pdif_vals = [np.exp(-m["PDIF01"]) for m in attack_metrics]
            global_metrics["n_sig_pdif_vals"] = self._get_n_significant(
                pdif_vals, self.p_thresh
            )
            global_metrics["n_sig_pdif_vals_corrected"] = self._get_n_significant(
                pdif_vals, self.p_thresh, bh_fdr_correction=True
            )

        return global_metrics

    def _get_n_significant(self, p_val_list, p_thresh, bh_fdr_correction=False):
        """
        Helper method to determine if values within a list of p-values are significant at
        p_thresh. Can perform multiple testing correction.
        """
        if not bh_fdr_correction:
            return sum(1 for p in p_val_list if p <= p_thresh)
        p_val_list = np.asarray(sorted(p_val_list))
        n_vals = len(p_val_list)
        hoch_vals = np.array([(k / n_vals) * P_THRESH for k in range(1, n_vals + 1)])
        bh_sig_list = p_val_list <= hoch_vals
        if any(bh_sig_list):
            n_sig_bh = (np.where(bh_sig_list)[0]).max() + 1
        else:
            n_sig_bh = 0
        return n_sig_bh

    def _generate_array(self, n_rows: int, beta: float) -> np.ndarray:
        """Generate a single array of predictions, used when doing baseline experiments.

        Parameters
        ----------
        n_rows : int
            the number of rows worth of data to generate
        beta : float
            the beta parameter for sampling probabilities

        Returns
        -------
        preds : np.ndarray
            Array of predictions. Two columns, n_rows rows

        Notes
        -----

        Examples
        --------
        """

        preds = np.zeros((n_rows, 2), float)
        for row_idx in range(n_rows):
            train_class = np.random.choice(2)
            train_prob = np.random.beta(1, beta)
            preds[row_idx, train_class] = train_prob
            preds[row_idx, 1 - train_class] = 1 - train_prob
        return preds

    def generate_arrays(
        self,
        n_rows_in: int,
        n_rows_out: int,
        train_beta: float = 2,
        test_beta: float = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate train and test prediction arrays, used when computing baseline.

        Parameters
        ----------
        n_rows_in : int
            number of rows of in-sample (training) probabilities
        n_rows_out : int
            number of rows of out-of-sample (testing) probabilities
        train_beta : float
            beta value for generating train probabilities
        test_beta : float:
            beta_value for generating test probabilities

        Returns
        -------
        train_preds : np.ndarray
            Array of train predictions (n_rows x 2 columns)
        test_preds : np.ndarray
            Array of test predictions (n_rows x 2 columns)
        """
        train_preds = self._generate_array(n_rows_in, train_beta)
        test_preds = self._generate_array(n_rows_out, test_beta)
        return train_preds, test_preds

    def make_dummy_data(self) -> None:
        """Makes dummy data for testing functionality.

        Parameters
        ----------
        args : dict
            Command line arguments

        Returns
        -------

        Notes
        -----
        Returns nothing but saves two .csv files
        """
        logger = logging.getLogger("dummy-data")
        logger.info(
            "Making dummy data with %d rows in and %d out",
            self.n_rows_in,
            self.n_rows_out,
        )
        logger.info("Generating rows")
        train_preds, test_preds = self.generate_arrays(
            self.n_rows_in,
            self.n_rows_out,
            train_beta=self.train_beta,
            test_beta=self.test_beta,
        )
        logger.info("Saving files")
        np.savetxt(self.training_preds_filename, train_preds, delimiter=",")
        np.savetxt(self.test_preds_filename, test_preds, delimiter=",")

    def _construct_metadata(self):
        """Constructs the metadata object, after attacks."""
        self.metadata = {}
        # Store all args
        self.metadata["experiment_details"] = {}
        self.metadata["experiment_details"] = self.get_params()

        self.metadata["attack"] = str(self)

        # Global metrics
        self.metadata["global_metrics"] = self._get_global_metrics(self.attack_metrics)
        self.metadata["baseline_global_metrics"] = self._get_global_metrics(
            self._unpack_dummy_attack_metrics_experiments_instances()
        )

    def _unpack_dummy_attack_metrics_experiments_instances(self) -> list:
        """Constructs the metadata object, after attacks."""
        dummy_attack_metrics_instances = []

        for exp_rep, _ in enumerate(self.dummy_attack_metrics):
            temp_dummy_attack_metrics = self.dummy_attack_metrics[exp_rep]
            dummy_attack_metrics_instances += temp_dummy_attack_metrics

        return dummy_attack_metrics_instances

    def _get_attack_metrics_instances(self) -> dict:
        """Constructs the metadata object, after attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}

        for rep, _ in enumerate(self.attack_metrics):
            attack_metrics_instances["instance_" + str(rep)] = self.attack_metrics[rep]

        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        attack_metrics_experiment["attack_metric_failfast_summary"] = (
            self.attack_metric_failfast_summary.get_attack_summary()
        )

        return attack_metrics_experiment

    def _get_dummy_attack_metrics_experiments_instances(self) -> dict:
        """Constructs the metadata object, after attacks."""
        dummy_attack_metrics_experiments = {}

        for exp_rep, _ in enumerate(self.dummy_attack_metrics):
            temp_dummy_attack_metrics = self.dummy_attack_metrics[exp_rep]
            dummy_attack_metric_instances = {}
            for rep, _ in enumerate(temp_dummy_attack_metrics):
                dummy_attack_metric_instances["instance_" + str(rep)] = (
                    temp_dummy_attack_metrics[rep]
                )
            temp = {}
            temp["attack_instance_logger"] = dummy_attack_metric_instances
            temp["attack_metric_failfast_summary"] = (
                self.dummy_attack_metric_failfast_summary[exp_rep].get_attack_summary()
            )
            dummy_attack_metrics_experiments[
                "dummy_attack_metrics_experiment_" + str(exp_rep)
            ] = temp

        return dummy_attack_metrics_experiments

    def make_report(self) -> dict:
        """Creates output dictionary structure and generates
        pdf and json outputs if filenames are given.
        """
        output = {}
        output["log_id"] = str(uuid.uuid4())
        output["log_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        self._construct_metadata()
        output["metadata"] = self.metadata

        output["attack_experiment_logger"] = self._get_attack_metrics_instances()
        output["dummy_attack_experiments_logger"] = (
            self._get_dummy_attack_metrics_experiments_instances()
        )

        report_dest = os.path.join(self.output_dir, self.report_name)
        json_attack_formatter = GenerateJSONModule(report_dest + ".json")
        json_report = report.create_json_report(output)
        json_attack_formatter.add_attack_output(json_report, "WorstCaseAttack")

        pdf_report = report.create_mia_report(output)
        report.add_output_to_pdf(report_dest, pdf_report, "WorstCaseAttack")
        return output


def _make_dummy_data(args):
    """Initialise class and run dummy data creation."""
    args.__dict__["training_preds_filename"] = "train_preds.csv"
    args.__dict__["test_preds_filename"] = "test_preds.csv"
    attack_obj = WorstCaseAttack(
        train_beta=args.train_beta,
        test_beta=args.test_beta,
        n_rows_in=args.n_rows_in,
        n_rows_out=args.n_rows_out,
        training_preds_filename=args.training_preds_filename,
        test_preds_filename=args.test_preds_filename,
    )
    attack_obj.make_dummy_data()


def _run_attack(args):
    """Initialise class and run attack from prediction files."""
    attack_obj = WorstCaseAttack(
        n_reps=args.n_reps,
        p_thresh=args.p_thresh,
        n_dummy_reps=args.n_dummy_reps,
        train_beta=args.train_beta,
        test_beta=args.test_beta,
        test_prop=args.test_prop,
        training_preds_filename=args.training_preds_filename,
        test_preds_filename=args.test_preds_filename,
        output_dir=args.output_dir,
        report_name=args.report_name,
        sort_probs=args.sort_probs,
        attack_metric_success_name=args.attack_metric_success_name,
        attack_metric_success_thresh=args.attack_metric_success_thresh,
        attack_metric_success_comp_type=args.attack_metric_success_comp_type,
        attack_metric_success_count_thresh=args.attack_metric_success_count_thresh,
        attack_fail_fast=args.attack_fail_fast,
    )
    print(attack_obj.training_preds_filename)
    attack_obj.attack_from_prediction_files()
    _ = attack_obj.make_report()


def _run_attack_from_configfile(args):
    """Initialise class and run attack from prediction files using config file."""
    attack_obj = WorstCaseAttack(
        attack_config_json_file_name=str(args.attack_config_json_file_name),
        target_path=str(args.target_path),
    )
    target = Target()
    target.load(attack_obj.target_path)
    attack_obj.attack(target)
    _ = attack_obj.make_report()


def main():
    """Main method to parse arguments and invoke relevant method."""
    logger = logging.getLogger("main")
    parser = argparse.ArgumentParser(
        description=("Perform a worst case attack from saved model predictions")
    )

    subparsers = parser.add_subparsers()
    dummy_parser = subparsers.add_parser("make-dummy-data")
    dummy_parser.add_argument(
        "--num-rows-in",
        action="store",
        dest="n_rows_in",
        type=int,
        required=False,
        default=1000,
        help=("How many rows to generate in the in-sample file. Default = %(default)d"),
    )

    dummy_parser.add_argument(
        "--num-rows-out",
        action="store",
        dest="n_rows_out",
        type=int,
        required=False,
        default=1000,
        help=(
            "How many rows to generate in the out-of-sample file. Default = %(default)d"
        ),
    )

    dummy_parser.add_argument(
        "--train-beta",
        action="store",
        type=float,
        required=False,
        default=5,
        dest="train_beta",
        help=(
            """Value of b parameter for beta distribution used to sample the in-sample
            probabilities. High values will give more extreme probabilities. Set this
            value higher than --test-beta to see successful attacks. Default = %(default)f"""
        ),
    )

    dummy_parser.add_argument(
        "--test-beta",
        action="store",
        type=float,
        required=False,
        default=2,
        dest="test_beta",
        help=(
            "Value of b parameter for beta distribution used to sample the out-of-sample "
            "probabilities. High values will give more extreme probabilities. Set this value "
            "lower than --train-beta to see successful attacks. Default = %(default)f"
        ),
    )

    dummy_parser.set_defaults(func=_make_dummy_data)

    attack_parser = subparsers.add_parser("run-attack")
    attack_parser.add_argument(
        "-i",
        "--training-preds-filename",
        action="store",
        dest="training_preds_filename",
        required=False,
        type=str,
        default="train_preds.csv",
        help=(
            "csv file containing the predictive probabilities (one column per class) for the "
            "training data (one row per training example). Default = %(default)s"
        ),
    )

    attack_parser.add_argument(
        "-o",
        "--test-preds-filename",
        action="store",
        dest="test_preds_filename",
        required=False,
        type=str,
        default="test_preds.csv",
        help=(
            "csv file containing the predictive probabilities (one column per class) for the "
            "non-training data (one row per training example). Default = %(default)s"
        ),
    )

    attack_parser.add_argument(
        "-r",
        "--n-reps",
        type=int,
        required=False,
        default=5,
        action="store",
        dest="n_reps",
        help=(
            "Number of repetitions (splitting data into attack model training and testing "
            "partitions to perform. Default = %(default)d"
        ),
    )

    attack_parser.add_argument(
        "-t",
        "--test-prop",
        type=float,
        required=False,
        default=0.3,
        action="store",
        dest="test_prop",
        help=(
            "Proportion of examples to be used for testing when fiting the attack model. "
            "Default = %(default)f"
        ),
    )

    attack_parser.add_argument(
        "--output-dir",
        type=str,
        action="store",
        dest="output_dir",
        default="output_worstcase",
        required=False,
        help=("Directory name where output files are stored. Default = %(default)s."),
    )

    attack_parser.add_argument(
        "--report-name",
        type=str,
        action="store",
        dest="report_name",
        default="report_worstcase",
        required=False,
        help=(
            """Filename for the pdf and json report outputs. Default = %(default)s.
            Code will append .pdf and .json"""
        ),
    )

    attack_parser.add_argument(
        "--n-dummy-reps",
        type=int,
        action="store",
        dest="n_dummy_reps",
        default=1,
        required=False,
        help=(
            "Number of dummy datasets to sample. Each will be assessed with --n-reps train and "
            "test splits. Set to 0 to do no baseline calculations. Default = %(default)d"
        ),
    )

    attack_parser.add_argument(
        "--p-thresh",
        action="store",
        type=float,
        default=P_THRESH,
        required=False,
        dest="p_thresh",
        help=("P-value threshold for significance testing. Default = %(default)f"),
    )

    attack_parser.add_argument(
        "--train-beta",
        action="store",
        type=float,
        required=False,
        default=5,
        dest="train_beta",
        help=(
            "Value of b parameter for beta distribution used to sample the in-sample probabilities."
            "High values will give more extreme probabilities. Set this value higher than "
            "--test-beta to see successful attacks. Default = %(default)f"
        ),
    )

    attack_parser.add_argument(
        "--test-beta",
        action="store",
        type=float,
        required=False,
        default=2,
        dest="test_beta",
        help=(
            "Value of b parameter for beta distribution used to sample the out-of-sample "
            "probabilities. High values will give more extreme probabilities. Set this value "
            "lower than --train-beta to see successful attacks. Default = %(default)f"
        ),
    )

    # --include-correct feature not supported as not currently possible from the command line
    # as we cannot compute the correctness of predictions.

    attack_parser.add_argument(
        "--sort-probs",
        action="store",
        type=bool,
        default=True,
        required=False,
        dest="sort_probs",
        help=(
            "Whether or not to sort the output probabilities (per row) before "
            "using them to train the attack model. Default = %(default)f"
        ),
    )

    attack_parser.add_argument(
        "--attack-metric-success-name",
        action="store",
        type=str,
        default="P_HIGHER_AUC",
        required=False,
        dest="attack_metric_success_name",
        help=(
            """for computing attack success/failure based on
            --attack-metric-success-thresh option. Default = %(default)s"""
        ),
    )

    attack_parser.add_argument(
        "--attack-metric-success-thresh",
        action="store",
        type=float,
        default=0.05,
        required=False,
        dest="attack_metric_success_thresh",
        help=(
            """for defining threshold value to measure attack success
            for the metric defined by argument --fail-metric-name option. Default = %(default)f"""
        ),
    )

    attack_parser.add_argument(
        "--attack-metric-success-comp-type",
        action="store",
        type=str,
        default="lte",
        required=False,
        dest="attack_metric_success_comp_type",
        help=(
            """for computing attack success/failure based on
            --attack-metric-success-thresh option. Default = %(default)s"""
        ),
    )

    attack_parser.add_argument(
        "--attack-metric-success-count-thresh",
        action="store",
        type=int,
        default=2,
        required=False,
        dest="attack_metric_success_count_thresh",
        help=(
            """for setting counter limit to stop further repetitions given the attack is
             successful and the --attack-fail-fast is true. Default = %(default)d"""
        ),
    )

    attack_parser.add_argument(
        "--attack-fail-fast",
        action="store_true",
        required=False,
        dest="attack_fail_fast",
        help=(
            """to stop further repetitions when the given metric has fulfilled
            a criteria for a specified number of times (--attack-metric-success-count-thresh)
            and this has a true status. Default = %(default)s"""
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
        default="config_worstcase_cmd.json",
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
        default="worstcase_target",
        help=(
            """Name of the target directory to load the trained target model and the target data.
            Default = %(default)s"""
        ),
    )

    attack_parser_config.set_defaults(func=_run_attack_from_configfile)

    args = parser.parse_args()

    try:
        args.func(args)
    except AttributeError as e:  # pragma:no cover
        logger.error("Invalid command. Try --help to get more details")
        logger.error(e)


if __name__ == "__main__":  # pragma:no cover
    main()
