"""
worst_case_attack.py

Runs a worst case attack based upon predictive probabilities stored in two .csv files
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Hashable
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks import metrics, report
from aisdc.attacks.attack import Attack
from aisdc.attacks.dataset import Data

logging.basicConfig(level=logging.INFO)

P_THRESH = 0.05


class WorstCaseAttackArgs:
    """Arguments for worst case"""

    def __init__(self, **kwargs):
        self.__dict__["n_reps"] = 10
        self.__dict__["p_thresh"] = 0.05
        self.__dict__["n_dummy_reps"] = 1
        self.__dict__["train_beta"] = 2
        self.__dict__["test_beta"] = 2
        self.__dict__["test_prop"] = 0.3
        self.__dict__["n_rows_in"] = 1000
        self.__dict__["n_rows_out"] = 1000
        self.__dict__["in_sample_filename"] = None
        self.__dict__["out_sample_filename"] = None
        self.__dict__["report_name"] = None
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


class WorstCaseAttack(Attack):
    """Class to wrap the worst case attack code"""

    def __init__(self, args: WorstCaseAttackArgs = WorstCaseAttackArgs()):
        self.attack_metrics = None
        self.dummy_attack_metrics = None
        self.metadata = None
        self.args = args

    def __str__(self):
        return "WorstCase attack"

    def attack(self, dataset: Data, target_model: sklearn.base.BaseEstimator) -> None:
        """Programmatic attack entry point

        To be used when code has access to data class and trained target model

        Parameters
        ----------
        dataset: attacks.dataset.Data
            dataset as a Data class object
        target_model: sklearn.base.BaseEstimator
            target model that inherits from an sklearn BaseEstimator
        """
        train_preds = target_model.predict_proba(dataset.x_train)
        test_preds = target_model.predict_proba(dataset.x_test)
        self.attack_from_preds(train_preds, test_preds)

    def attack_from_prediction_files(self):
        """Start an attack from saved prediction files

        To be used when only saved predictions are available.

        Filenames for the saved prediction files to be specified in the arguments provided
        in the constructor
        """
        train_preds = np.loadtxt(self.args.in_sample_filename, delimiter=",")
        test_preds = np.loadtxt(self.args.out_sample_filename, delimiter=",")
        self.attack_from_preds(train_preds, test_preds)

    def attack_from_preds(  # pylint: disable=too-many-locals
        self, train_preds: np.ndarray, test_preds: np.ndarray
    ) -> None:
        """
        Runs the attack based upon the predictions in train_preds and test_preds, and the params
        stored in self.args

        Parameters
        ----------

        train_preds: np.ndarray
            Array of train predictions. One row per example, one column per class (i.e. 2)
        test_preds:  np.ndarray
            Array of test predictions. One row per example, one column per class (i.e. 2)

        """
        logger = logging.getLogger("attack-from-preds")
        logger.info("Running main attack repetitions")
        self.attack_metrics = self.run_attack_reps(train_preds, test_preds)
        if self.args.n_dummy_reps > 0:
            logger.info("Running dummy attack reps")
            self.dummy_attack_metrics = []
            n_train_rows = len(train_preds)
            n_test_rows = len(test_preds)
            for _ in range(self.args.n_dummy_reps):
                d_train_preds, d_test_preds = self.generate_arrays(
                    n_train_rows, n_test_rows, self.args.train_beta, self.args.test_beta
                )
                temp_metrics = self.run_attack_reps(d_train_preds, d_test_preds)
                self.dummy_attack_metrics += temp_metrics
        logger.info("Finished running attacks")

    def run_attack_reps(self, train_preds: np.ndarray, test_preds: np.ndarray) -> list:
        """
        Run actual attack reps from train and test predictions

        Parameters
        ----------
        train_preds: np.ndarray
            predictions from the model on training (in-sample) data
        test_preds: np.ndarray
            predictions from the model on testing (out-of-sample) data

        Returns
        -------
        mia_metrics: List
            List of attack metrics dictionaries, one for each repetition
        """
        self.args.set_param("n_rows_in", len(train_preds))
        self.args.set_param("n_rows_out", len(test_preds))
        logger = logging.getLogger("attack-reps")
        logger.info("Sorting probabilities to leave highest value in first column")
        train_preds = -np.sort(-train_preds, axis=1)
        test_preds = -np.sort(-test_preds, axis=1)

        logger.info("Creating MIA data")
        mi_x = np.vstack((train_preds, test_preds))
        mi_y = np.hstack((np.ones(len(train_preds)), np.zeros(len(test_preds))))

        mia_metrics = []
        for rep in range(self.args.n_reps):
            logger.info("Rep %d of %d", rep, self.args.n_reps)
            mi_train_x, mi_test_x, mi_train_y, mi_test_y = train_test_split(
                mi_x, mi_y, test_size=self.args.test_prop, stratify=mi_y
            )
            attack_classifier = RandomForestClassifier()
            attack_classifier.fit(mi_train_x, mi_train_y)

            mia_metrics.append(
                metrics.get_metrics(attack_classifier, mi_test_x, mi_test_y)
            )

        logger.info("Finished simulating attacks")

        return mia_metrics

    def _get_global_metrics(self, attack_metrics: list) -> dict:
        """Summarise metrics from a metric list

        Arguments
        ---------
        attack_metrics: List
            list of attack metrics dictionaries

        Returns
        -------
        global_metrics: Dict
            Dictionary of summary metrics

        """
        global_metrics = {}
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

        global_metrics[
            "null_auc_3sd_range"
        ] = f"{0.5 - 3*auc_std:.4f} -> {0.5 + 3*auc_std:.4f}"
        global_metrics["n_sig_auc_p_vals"] = self._get_n_significant(
            auc_p_vals, self.args.p_thresh
        )
        global_metrics["n_sig_auc_p_vals_corrected"] = self._get_n_significant(
            auc_p_vals, self.args.p_thresh, bh_fdr_correction=True
        )

        pdif_vals = [np.exp(-m["PDIF01"]) for m in attack_metrics]
        global_metrics["n_sig_pdif_vals"] = self._get_n_significant(
            pdif_vals, self.args.p_thresh
        )
        global_metrics["n_sig_pdif_vals_corrected"] = self._get_n_significant(
            pdif_vals, self.args.p_thresh, bh_fdr_correction=True
        )

        return global_metrics

    def _get_n_significant(self, p_val_list, p_thresh, bh_fdr_correction=False):
        """
        Helper method to determine if values within a list of p-values are significant at
        p_thresh. Can perform multiple testing correction.
        """
        if not bh_fdr_correction:
            return sum(
                1 for p in p_val_list if p <= p_thresh
            )  # pylint: disable = consider-using-generator
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
        """Generate a single array of predictions, used when doing baseline experiments

        Parameters
        ----------
        n_rows: int
            the number of rows worth of data to generate
        beta: float
            the beta parameter for sampling probabilities

        Returns
        -------
        preds: np.ndarray
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
        """Generate train and test prediction arrays, used when computing baseline

        Parameters
        ----------
        n_rows_in: int
            number of rows of in-sample (training) probabilities
        n_rows_out: int
            number of rows of out-of-sample (testing) probabilities
        train_beta: float
            beta value for generating train probabilities
        test_beta: float:
            beta_value for generating test probabilities

        Returns
        -------
        train_preds: np.ndarray
            Array of train predictions (n_rows x 2 columns)
        test_preds: np.ndarray
            Array of test predictions (n_rows x 2 columns)
        """
        train_preds = self._generate_array(n_rows_in, train_beta)
        test_preds = self._generate_array(n_rows_out, test_beta)
        return train_preds, test_preds

    def make_dummy_data(self) -> None:
        """Makes dummy data for testing functionality

        Parameters
        ----------
        args: dict
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
            self.args.n_rows_in,
            self.args.n_rows_out,
        )
        logger.info("Generating rows")
        train_preds, test_preds = self.generate_arrays(
            self.args.n_rows_in,
            self.args.n_rows_out,
            train_beta=self.args.train_beta,
            test_beta=self.args.test_beta,
        )
        logger.info("Saving files")
        np.savetxt(self.args.in_sample_filename, train_preds, delimiter=",")
        np.savetxt(self.args.out_sample_filename, test_preds, delimiter=",")

    def _construct_metadata(self):
        """Constructs the metadata object, after attacks"""
        self.metadata = {}
        # Store all args
        self.metadata["experiment_details"] = {}
        self.metadata["experiment_details"].update(self.args.__dict__)
        if "func" in self.metadata["experiment_details"]:
            del self.metadata["experiment_details"]["func"]

        self.metadata["attack"] = str(self)

        # Global metrics
        self.metadata["global_metrics"] = self._get_global_metrics(self.attack_metrics)
        self.metadata["baseline_global_metrics"] = self._get_global_metrics(
            self.dummy_attack_metrics
        )

    def make_report(self) -> dict:
        """Creates output dictionary structure"""
        output = {}
        output["attack_metrics"] = self.attack_metrics
        output["dummy_attack_metrics"] = self.dummy_attack_metrics
        self._construct_metadata()
        output["metadata"] = self.metadata
        if self.args.report_name is not None:
            json_report = report.create_json_report(output)
            with open(f"{self.args.report_name}.json", "w", encoding="utf-8") as f:
                f.write(json_report)

            pdf = report.create_mia_report(output)
            pdf.output(f"{self.args.report_name}.pdf", "F")

        return output


def _make_dummy_data(args):
    """Initialise class and run dummy data creation"""
    wc_args = WorstCaseAttackArgs(**args.__dict__)
    wc_args.set_param("in_sample_filename", "train_preds.csv")
    wc_args.set_param("out_sample_filename", "test_preds.csv")
    attack_obj = WorstCaseAttack(wc_args)
    attack_obj.make_dummy_data()


def _run_attack(args):
    """Initialise class and run attack from prediction files"""
    wc_args = WorstCaseAttackArgs(**args.__dict__)
    attack_obj = WorstCaseAttack(wc_args)
    attack_obj.attack_from_prediction_files()
    _ = attack_obj.make_report()


def main():
    """main method to parse arguments and invoke relevant method"""
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
            "Value of b parameter for beta distribution used to sample the in-sample "
            "probabilities. "
            "High values will give more extreme probabilities. Set this value higher than "
            "--test-beta to see successful attacks. Default = %(default)f"
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
        "--in-sample-preds",
        action="store",
        dest="in_sample_filename",
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
        "--out-of-sample-preds",
        action="store",
        dest="out_sample_filename",
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
        "--report-name",
        type=str,
        action="store",
        dest="report_name",
        default="worstcase_report",
        required=False,
        help=(
            "Filename for the report output. Default = %(default)s. Code will append .pdf and .json"
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

    attack_parser.set_defaults(func=_run_attack)

    args = parser.parse_args()

    try:
        args.func(args)
    except AttributeError as e:  # pragma:no cover
        logger.error("Invalid command. Try --help to get more details")
        logger.error(e)


if __name__ == "__main__":  # pragma:no cover
    main()
