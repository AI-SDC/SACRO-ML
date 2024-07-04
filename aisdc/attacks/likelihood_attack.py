"""Likelihood testing scenario from https://arxiv.org/pdf/2112.03570.pdf."""

# pylint: disable = too-many-branches

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import sklearn
from fpdf import FPDF
from scipy.stats import norm

from aisdc import metrics
from aisdc.attacks import report
from aisdc.attacks.attack import Attack
from aisdc.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-16  # Used to avoid numerical issues in logit function


class DummyClassifier:
    """A Dummy Classifier to allow this code to work with get_metrics."""

    def predict(self, X_test):
        """Return an array of 1/0 depending on value in second column."""
        return 1 * (X_test[:, 1] > 0.5)

    def predict_proba(self, X_test):
        """Simply return the X_test."""
        return X_test


def _logit(p: float) -> float:
    """Return standard logit.

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
    If p is close to 0 or 1, evaluating the log will result in numerical
    instabilities.  This code thresholds p at EPS and 1 - EPS where EPS
    defaults at 1e-16.
    """
    if p > 1 - EPS:  # pylint:disable=consider-using-min-builtin
        p = 1 - EPS
    p = max(p, EPS)
    return np.log(p / (1 - p))


class LIRAAttack(Attack):
    """The main LiRA Attack class."""

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        n_shadow_models: int = 100,
        p_thresh: float = 0.05,
    ) -> None:
        """Construct an object to execute a LiRA attack.

        Parameters
        ----------
        output_dir : str
            Name of the directory where outputs are stored.
        write_report : bool
            Whether to generate a JSON and PDF report.
        n_shadow_models : int
            Number of shadow models to be trained.
        p_thresh : float
            Threshold to determine significance of things. For instance
            auc_p_value and pdif_vals.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.n_shadow_models = n_shadow_models
        self.p_thresh = p_thresh

    def __str__(self):
        """Return the name of the attack."""
        return "LiRA Attack"

    def attack(self, target: Target) -> dict:
        """Run a LiRA attack from a Target object and a target model.

        Needs to have X_train, X_test, y_train and y_test set.

        Parameters
        ----------
        target : attacks.target.Target
            target as an instance of the Target class.

        Returns
        -------
        dict
            Attack report.
        """
        shadow_clf = sklearn.base.clone(target.model)
        target = self._check_and_update_dataset(target)
        # execute attack
        self.run_scenario_from_preds(
            shadow_clf,
            target.X_train,
            target.y_train,
            target.model.predict_proba(target.X_train),
            target.X_test,
            target.y_test,
            target.model.predict_proba(target.X_test),
        )
        # create the report
        output = self._make_report(target)
        # write the report
        self._write_report(output)
        # return the report
        return output

    def _check_and_update_dataset(self, target: Target) -> Target:
        """Check that it is safe to use class variables to index prediction arrays.

        This has two steps:
        1. Replacing the values in y_train with their position in
        target.model.classes (will normally result in no change)
        2. Removing from the test set any rows corresponding to classes that
        are not in the training set.
        """
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

        if len(y_test_new) != len(target.X_test):
            target.X_test = target.X_test[ok_pos, :]
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
    ) -> None:
        """Run the likelihood test, using the "offline" version.

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
        """
        n_train_rows, _ = X_target_train.shape
        n_shadow_rows, _ = X_shadow_train.shape
        indices = np.arange(0, n_train_rows + n_shadow_rows, 1)

        # Combine taregt and shadow train, from which to sample datasets
        combined_x_train = np.vstack((X_target_train, X_shadow_train))
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
            temp_x_train = combined_x_train[these_idx, :]
            temp_y_train = combined_y_train[these_idx]

            # Fit the shadow model
            shadow_clf.set_params(random_state=model_idx)
            shadow_clf.fit(temp_x_train, temp_y_train)

            # map a class to a column
            class_map = {c: i for i, c in enumerate(shadow_clf.classes_)}

            # Get the predicted probabilities on the training data
            confidences = shadow_clf.predict_proba(X_target_train)

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

        mia_scores = np.array(mia_scores)
        mia_labels = np.array(mia_labels)
        y_pred_proba = mia_clf.predict_proba(mia_scores)
        self.attack_metrics = [metrics.get_metrics(y_pred_proba, mia_labels)]

    def _construct_metadata(self) -> None:
        """Construct the metadata object."""
        super()._construct_metadata()

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
        self.metadata["global_metrics"]["null_auc_3sd_range"] = (
            f"{0.5 - 3 * auc_std} -> {0.5 + 3 * auc_std}"
        )

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report."""
        return report.create_lr_report(output)

    def _get_attack_metrics_instances(self) -> dict:
        """Construct the metadata object after attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}

        for rep, _ in enumerate(self.attack_metrics):
            attack_metrics_instances["instance_" + str(rep)] = self.attack_metrics[rep]

        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        return attack_metrics_experiment
