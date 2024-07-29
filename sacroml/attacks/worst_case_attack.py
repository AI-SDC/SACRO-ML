"""Run a worst case attack based upon predictive probabilities."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from fpdf import FPDF
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sacroml import metrics
from sacroml.attacks import report
from sacroml.attacks.attack import Attack, get_class_by_name
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

P_THRESH = 0.05


class WorstCaseAttack(Attack):  # pylint: disable=too-many-instance-attributes
    """Worst case attack."""

    def __init__(  # pylint: disable = too-many-arguments
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        n_reps: int = 10,
        reproduce_split: int | Iterable[int] | None = 5,
        p_thresh: float = 0.05,
        n_dummy_reps: int = 1,
        train_beta: int = 1,
        test_beta: int = 1,
        test_prop: float = 0.2,
        include_model_correct_feature: bool = False,
        sort_probs: bool = True,
        attack_model: str = "sklearn.ensemble.RandomForestClassifier",
        attack_model_params: dict | None = None,
    ) -> None:
        """Construct an object to execute a worst case attack.

        Parameters
        ----------
        output_dir : str
            Name of the directory where outputs are stored.
        write_report : bool
            Whether to generate a JSON and PDF report.
        n_reps : int
            Number of attacks to run -- in each iteration an attack model
            is trained on a different subset of the data.
        reproduce_split : int or Iterable[int] or None
            Variable that controls the reproducibility of the data split.
            It can be an integer or a list of integers of length `n_reps`.
            Default : 5.
        p_thresh : float
            Threshold to determine significance of things. For instance
            `auc_p_value` and `pdif_vals`.
        n_dummy_reps : int
            Number of baseline (dummy) experiments to do.
        train_beta : int
            Value of b for beta distribution used to sample the in-sample
            (training) probabilities.
        test_beta : int
            Value of b for beta distribution used to sample the out-of-sample
            (test) probabilities.
        test_prop : float
            Proportion of data to use as a test set for the attack model.
        include_model_correct_feature : bool
            Inclusion of additional feature to hold whether or not the target model
            made a correct prediction for each example.
        sort_probs : bool
            Whether to sort combined preds (from training and test)
            to have highest probabilities in the first column.
        attack_model : str
            Class name of the attack model.
        attack_model_params : dict or None
            Dictionary of hyperparameters for the `attack_model`
            such as `min_sample_split`, `min_samples_leaf`, etc.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.n_reps: int = n_reps
        self.reproduce_split: int | Iterable[int] | None = reproduce_split
        self.p_thresh: float = p_thresh
        self.n_dummy_reps: int = n_dummy_reps
        self.train_beta: int = train_beta
        self.test_beta: int = test_beta
        self.test_prop: float = test_prop
        self.include_model_correct_feature: bool = include_model_correct_feature
        self.sort_probs: bool = sort_probs
        self.attack_model: str = attack_model
        self.attack_model_params: dict | None = attack_model_params
        self.dummy_attack_metrics: list = []

    def __str__(self) -> str:
        """Return name of attack."""
        return "WorstCase attack"

    def attack(self, target: Target) -> dict:
        """Run worst case attack.

        Parameters
        ----------
        target : attacks.target.Target
            target as a Target class object

        Returns
        -------
        dict
            Attack report.
        """
        train_c = None
        test_c = None
        # compute target model probas if possible
        if (
            target.model is not None
            and target.X_train is not None
            and target.y_train is not None
        ):
            proba_train = target.model.predict_proba(target.X_train)
            proba_test = target.model.predict_proba(target.X_test)
            if self.include_model_correct_feature:
                train_c = 1 * (target.y_train == target.model.predict(target.X_train))
                test_c = 1 * (target.y_test == target.model.predict(target.X_test))
        # use supplied target model probas if unable to compute
        elif target.proba_train is not None and target.proba_test is not None:
            proba_train = target.proba_train
            proba_test = target.proba_test
        # cannot proceed
        else:
            logger.info("Insufficient Target details to run worst case attack.")
            return {}
        # execute attack
        self.attack_from_preds(
            proba_train,
            proba_test,
            train_correct=train_c,
            test_correct=test_c,
        )
        # create the report
        output = self._make_report(target)
        # write the report
        self._write_report(output)
        # return the report
        return output

    def _make_report(self, target: Target) -> dict:
        """Create attack report."""
        output = super()._make_report(target)
        output["dummy_attack_experiments_logger"] = (
            self._get_dummy_attack_metrics_experiments_instances()
        )
        return output

    def attack_from_preds(
        self,
        proba_train: np.ndarray,
        proba_test: np.ndarray,
        train_correct: np.ndarray | None = None,
        test_correct: np.ndarray | None = None,
    ) -> None:
        """Run attack based upon the predictions in proba_train and proba_test.

        Parameters
        ----------
        proba_train : np.ndarray
            Array of train predictions. One row per example, one column per class.
        proba_test : np.ndarray
            Array of test predictions. One row per example, one column per class.
        """
        logger.info("Running main attack repetitions")
        attack_metric_dict = self.run_attack_reps(
            proba_train,
            proba_test,
            train_correct=train_correct,
            test_correct=test_correct,
        )
        self.attack_metrics = attack_metric_dict["mia_metrics"]

        self.dummy_attack_metrics = []
        if self.n_dummy_reps > 0:
            logger.info("Running dummy attack reps")
            n_train_rows = len(proba_train)
            n_test_rows = len(proba_test)
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
                self.dummy_attack_metrics.append(temp_metrics)

        logger.info("Finished running attacks")

    def _prepare_attack_data(
        self,
        proba_train: np.ndarray,
        proba_test: np.ndarray,
        train_correct: np.ndarray = None,
        test_correct: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data and labels for attack model.

        Combines the train and test preds into a single numpy array
        (optionally) sorting each row to have the highest probabilities in the
        first column. Constructs a label array that has ones corresponding to
        training rows and zeros to testing rows.
        """
        if self.sort_probs:
            logger.info("Sorting probabilities to leave highest value in first column")
            proba_train = -np.sort(-proba_train, axis=1)
            proba_test = -np.sort(-proba_test, axis=1)

        logger.info("Creating MIA data")

        if self.include_model_correct_feature and train_correct is not None:
            proba_train = np.hstack((proba_train, train_correct[:, None]))
            proba_test = np.hstack((proba_test, test_correct[:, None]))

        mi_x = np.vstack((proba_train, proba_test))
        mi_y = np.hstack((np.ones(len(proba_train)), np.zeros(len(proba_test))))
        return (mi_x, mi_y)

    def _get_attack_model(self):
        """Return an instantiated attack model."""
        # load attack model module and get class
        model = get_class_by_name(self.attack_model)
        params = self.attack_model_params
        if (  # set custom default parameters for RF attack model
            self.attack_model == "sklearn.ensemble.RandomForestClassifier"
            and self.attack_model_params is None
        ):
            params = {
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_depth": 5,
            }
        # instantiate attack model
        return model(**params) if params is not None else model()

    def _get_reproducible_split(self) -> list:
        """Return a list of splits."""
        split = self.reproduce_split
        n_reps = self.n_reps
        if isinstance(split, int):
            split = [split] + [x**2 for x in range(split, split + n_reps - 1)]
        else:
            # remove potential duplicates
            split = list(dict.fromkeys(split))
            if len(split) == n_reps:
                pass
            elif len(split) > n_reps:
                print("split", split, "nreps", n_reps)
                split = list(split)[0:n_reps]
                print(
                    "WARNING: the length of the parameter 'reproduce_split' "
                    "is longer than n_reps. Values have been removed."
                )
            else:
                # assign values to match length of n_reps
                split += [split[-1] * x for x in range(2, (n_reps - len(split) + 2))]
                print(
                    "WARNING: the length of the parameter 'reproduce_split' "
                    "is shorter than n_reps. Values have been added."
                )
            print("reproduce split now", split)
        return split

    def run_attack_reps(  # pylint: disable = too-many-locals
        self,
        proba_train: np.ndarray,
        proba_test: np.ndarray,
        train_correct: np.ndarray = None,
        test_correct: np.ndarray = None,
    ) -> dict:
        """Run actual attack reps from train and test predictions.

        Parameters
        ----------
        proba_train : np.ndarray
            Predictions from the model on training (in-sample) data.
        proba_test : np.ndarray
            Predictions from the model on testing (out-of-sample) data.

        Returns
        -------
        dict
            Dictionary of mia_metrics (a list of metric across repetitions).
        """
        mi_x, mi_y = self._prepare_attack_data(
            proba_train, proba_test, train_correct, test_correct
        )

        mia_metrics = []
        split = self._get_reproducible_split()

        for rep in range(self.n_reps):
            logger.info("Rep %d of %d split %d", rep + 1, self.n_reps, split[rep])

            mi_train_x, mi_test_x, mi_train_y, mi_test_y = train_test_split(
                mi_x,
                mi_y,
                test_size=self.test_prop,
                stratify=mi_y,
                random_state=split[rep],
                shuffle=True,
            )

            attack_classifier = self._get_attack_model()
            attack_classifier.fit(mi_train_x, mi_train_y)

            y_pred_proba = attack_classifier.predict_proba(mi_test_x)
            mia_metrics.append(metrics.get_metrics(y_pred_proba, mi_test_y))

            if self.include_model_correct_feature and train_correct is not None:
                # Compute the Yeom TPR and FPR
                yeom_preds = mi_test_x[:, -1]
                tn, fp, fn, tp = confusion_matrix(mi_test_y, yeom_preds).ravel()
                mia_metrics[-1]["yeom_tpr"] = tp / (tp + fn)
                mia_metrics[-1]["yeom_fpr"] = fp / (fp + tn)
                mia_metrics[-1]["yeom_advantage"] = (
                    mia_metrics[-1]["yeom_tpr"] - mia_metrics[-1]["yeom_fpr"]
                )

        logger.info("Finished simulating attacks")
        return {"mia_metrics": mia_metrics}

    def _get_global_metrics(self, attack_metrics: list) -> dict:
        """Summarise metrics from a metric list.

        Parameters
        ----------
        attack_metrics : List
            list of attack metrics dictionaries

        Returns
        -------
        global_metrics : Dict
            Dictionary of summary metrics
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

    def _get_n_significant(
        self, p_val_list: list[float], p_thresh: float, bh_fdr_correction: bool = False
    ) -> int:
        """Return number of p-values significant at `p_thresh`.

        Can perform multiple testing correction.
        """
        if not bh_fdr_correction:
            return sum(1 for p in p_val_list if p <= p_thresh)
        p_val_list = np.asarray(sorted(p_val_list))
        n_vals = len(p_val_list)
        hoch_vals = np.array([(k / n_vals) * P_THRESH for k in range(1, n_vals + 1)])
        bh_sig_list = p_val_list <= hoch_vals
        return np.where(bh_sig_list)[0].max() + 1 if any(bh_sig_list) else 0

    def _generate_array(self, n_rows: int, beta: float) -> np.ndarray:
        """Generate array of predictions, used when doing baseline experiments.

        Parameters
        ----------
        n_rows : int
            The number of rows worth of data to generate.
        beta : float
            The beta parameter for sampling probabilities.

        Returns
        -------
        preds : np.ndarray
            Array of predictions. Two columns, `n_rows` rows.
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
            Number of rows of in-sample (training) probabilities.
        n_rows_out : int
            Number of rows of out-of-sample (testing) probabilities.
        train_beta : float
            Beta value for generating train probabilities.
        test_beta : float:
            Beta value for generating test probabilities.

        Returns
        -------
        proba_train : np.ndarray
            Array of train predictions (n_rows x 2 columns).
        proba_test : np.ndarray
            Array of test predictions (n_rows x 2 columns).
        """
        proba_train = self._generate_array(n_rows_in, train_beta)
        proba_test = self._generate_array(n_rows_out, test_beta)
        return proba_train, proba_test

    def _construct_metadata(self) -> None:
        """Construct the metadata object after attacks."""
        super()._construct_metadata()

        self.metadata["global_metrics"] = self._get_global_metrics(self.attack_metrics)
        self.metadata["baseline_global_metrics"] = self._get_global_metrics(
            self._unpack_dummy_attack_metrics_experiments_instances()
        )

    def _unpack_dummy_attack_metrics_experiments_instances(self) -> list:
        """Construct the metadata object after attacks."""
        dummy_attack_metrics_instances = []
        for exp_rep, _ in enumerate(self.dummy_attack_metrics):
            temp_dummy_attack_metrics = self.dummy_attack_metrics[exp_rep]
            dummy_attack_metrics_instances += temp_dummy_attack_metrics
        return dummy_attack_metrics_instances

    def _get_attack_metrics_instances(self) -> dict:
        """Construct the metadata object after attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}
        for rep, _ in enumerate(self.attack_metrics):
            attack_metrics_instances["instance_" + str(rep)] = self.attack_metrics[rep]
        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        return attack_metrics_experiment

    def _get_dummy_attack_metrics_experiments_instances(self) -> dict:
        """Construct the metadata object after attacks."""
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
            dummy_attack_metrics_experiments[
                "dummy_attack_metrics_experiment_" + str(exp_rep)
            ] = temp
        return dummy_attack_metrics_experiments

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report."""
        return report.create_mia_report(output)
