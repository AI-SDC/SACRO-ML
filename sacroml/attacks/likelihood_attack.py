"""Likelihood testing scenario from https://arxiv.org/pdf/2112.03570.pdf."""

from __future__ import annotations

import logging
import os

import numpy as np
from fpdf import FPDF
from scipy.stats import norm
from sklearn.base import BaseEstimator

from sacroml import metrics
from sacroml.attacks import report
from sacroml.attacks.attack import Attack
from sacroml.attacks.model import Model
from sacroml.attacks.target import Target
from sacroml.attacks.utils import get_p_normal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS: float = 1e-16  # Used to avoid numerical issues
MODES: list[str] = ["offline", "offline-carlini", "online-carlini"]


class LIRAAttack(Attack):
    """The main LiRA Attack class."""

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        n_shadow_models: int = 100,
        p_thresh: float = 0.05,
        mode: str = "offline",
        fix_variance: bool = False,
        report_individual: bool = False,
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
        mode : str
            Attack mode: {"offline", "offline-carlini", "online-carlini"}
        fix_variance : bool
            Whether to use the global standard deviation or per record.
        report_individual : bool
            Whether to report metrics for each individual record.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.n_shadow_models: int = n_shadow_models
        self.p_thresh: float = p_thresh
        self.mode: str = mode
        self.fix_variance: bool = fix_variance
        self.report_individual: bool = report_individual

        self.result: dict = {}  # individual record results
        if self.report_individual:
            self.result["score"] = []
            self.result["label"] = []
            self.result["target_logit"] = []
            self.result["out_p_norm"] = []
            self.result["out_prob"] = []
            self.result["out_mean"] = []
            self.result["out_std"] = []
            if self.mode == "online-carlini":
                self.result["in_prob"] = []
                self.result["in_mean"] = []
                self.result["in_std"] = []

        # Create folder for saving trained shadow models
        shadow_path: str = os.path.normpath(f"{self.output_dir}/shadow_models")
        os.makedirs(shadow_path, exist_ok=True)

    def __str__(self):
        """Return the name of the attack."""
        return "LiRA Attack"

    @classmethod
    def attackable(cls, target: Target) -> bool:  # pragma: no cover
        """Return whether a target can be assessed with LIRAAttack."""
        required_methods = [
            "clone",
            "predict_proba",
            "predict",
            "get_classes",
            "set_params",
        ]

        if (
            target.has_model()
            and target.has_data()
            and all(hasattr(target.model, method) for method in required_methods)
        ):
            return True

        logger.info("WARNING: LiRA requires a loadable model.")
        return False

    def _attack(self, target: Target) -> dict:
        """Run a LiRA attack from a Target object and a target model.

        Parameters
        ----------
        target : attacks.target.Target
            target as an instance of the Target class.

        Returns
        -------
        dict
            Attack report.
        """
        # prepare
        shadow_clf = target.model.clone()
        target = self._check_and_update_dataset(target)
        # execute attack
        self._run(
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
        if not isinstance(target.model.model, BaseEstimator):
            return target

        y_train_new = []
        classes = list(target.model.get_classes())
        for y in target.y_train:
            y_train_new.append(classes.index(y))
        target.y_train = np.array(y_train_new, int)
        logger.info(
            "new y_train has values and counts: %s",
            np.unique(target.y_train, return_counts=True),
        )
        ok_pos = []
        y_test_new = []
        for i, y in enumerate(target.y_test):
            if y in classes:
                ok_pos.append(i)
                y_test_new.append(classes.index(y))
        if len(y_test_new) != len(target.X_test):  # pragma: no cover
            target.X_test = target.X_test[ok_pos, :]
        target.y_test = np.array(y_test_new, int)
        logger.info(
            "new y_test has values and counts: %s",
            np.unique(target.y_test, return_counts=True),
        )
        return target

    def _run(
        self,
        shadow_clf: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        proba_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        proba_test: np.ndarray,
    ) -> None:
        """Run the likelihood test.

        See p.6 (top of second column) for details.

        With mode "offline", we measure the probability of observing a
        confidence as high as the target model's under the null-hypothesis that
        the target point is a non-member. That is we, use the norm CDF.

        With mode "offline-carlini", we measure the probability that a target point
        did not come from the non-member distribution. That is, we use Carlini's
        implementation with a single norm (log) PDF.

        With mode "online-carlini", we use Carlini's implementation of the standard
        likelihood ratio test, measuring the ratio of probabilities the sample came
        from the two distributions. That is, the (log) PDF of pr_in minus pr_out.

        Parameters
        ----------
        shadow_clf : Model
            A classifier that will be trained to form the shadow models.
            All hyperparameters should have been set.
        X_train : np.ndarray
            Data that was used to train the target model.
        y_train : np.ndarray
            Labels that were used to train the target model.
        proba_train : np.ndarray
            Array of predictions produced by the target model on the training data.
        X_test : np.ndarray
            Data that will be used to train the shadow models.
        y_test : np.ndarray
            Labels that will be used to train the shadow models.
        proba_test : np.ndarray
            Array of predictions produced by the target model on the shadow data.
        """
        logger.info("Running %s LiRA, fix_variance=%s", self.mode, self.fix_variance)

        n_train_rows = X_train.shape[0]
        n_shadow_rows = X_test.shape[0]

        combined_data = self._combine_data(
            X_train, y_train, proba_train, X_test, y_test, proba_test
        )

        self._train_shadow_models(
            shadow_clf,
            combined_data["features"],
            combined_data["labels"],
            n_train_rows,
        )

        out_conf, in_conf = self._get_shadow_signals(
            combined_data["features"],
            combined_data["labels"],
        )

        mia_scores, n_normal = self._compute_scores(
            combined_data["labels"], combined_data["predictions"], out_conf, in_conf
        )

        self._save_attack_metrics(mia_scores, n_train_rows, n_shadow_rows, n_normal)
        logger.info("Finished scenario")

    def _combine_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        proba_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        proba_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Combine training and test data.

        Assumes label encoding.
        """
        combined_features = np.vstack((X_train, X_test))
        combined_labels = np.hstack((y_train, y_test))
        combined_predictions = np.vstack((proba_train, proba_test))

        return {
            "features": combined_features,
            "labels": combined_labels,
            "predictions": combined_predictions,
        }

    def _train_shadow_models(
        self,
        shadow_clf: Model,
        combined_x_train: np.ndarray,
        combined_y_train: np.ndarray,
        n_train_rows: int,
    ) -> None:
        """Train and save shadow models.

        Parameters
        ----------
        shadow_clf : Model
            A classifier that will be trained to form the shadow models.
        combined_x_train : np.ndarray
            Array of combined train and test features.
        combined_y_train : np.ndarray
            Array of combined train and test labels.
        n_train_rows : int
            Number of samples in the training set.
        """
        logger.info("Training shadow models")

        n_combined: int = combined_x_train.shape[0]
        indices: np.ndarray = np.arange(0, n_combined, 1)

        for idx in range(self.n_shadow_models):
            if idx % 10 == 0:
                logger.info("Trained %d models", idx)

            # Pick the indices to use for training this one
            np.random.seed(idx)
            indices_train = np.random.choice(indices, n_train_rows, replace=False)
            indices_test = np.setdiff1d(indices, indices_train)

            # Fit the shadow model
            shadow_clf.set_params(random_state=idx)
            shadow_clf.fit(
                combined_x_train[indices_train, :],
                combined_y_train[indices_train],
            )

            # Save model and indices
            self.save_shadow_model(idx, shadow_clf, indices_train, indices_test)

    def _get_shadow_signals(
        self,
        combined_x_train: np.ndarray,
        combined_y_train: np.ndarray,
    ) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
        """Return confidence scores from saved shadow models.

        Parameters
        ----------
        combined_x_train : np.ndarray
            Array of combined train and test features.
        combined_y_train : np.ndarray
            Array of combined train and test labels.

        Returns
        -------
        tuple[dict, dict]
            Dictionary of confidences when not in the training set.
            Dictionary of confidences when in the training set.
        """
        logger.info("Getting shadow model signals")

        n_shadow_models: int = self.get_n_shadow_models()
        n_combined: int = combined_x_train.shape[0]
        out_conf: dict[int, list[float]] = {i: [] for i in range(n_combined)}
        in_conf: dict[int, list[float]] = {i: [] for i in range(n_combined)}

        for model_idx in range(n_shadow_models):
            # load shadow model
            shadow_clf, indices_train, _ = self.get_shadow_model(model_idx)
            # map a class to a column
            class_map = {c: i for i, c in enumerate(shadow_clf.get_classes())}
            # generate shadow confidences
            shadow_confidences = shadow_clf.predict_proba(combined_x_train)
            indices_train = set(indices_train)
            for i, conf in enumerate(shadow_confidences):
                # logit of the correct class
                label = class_map.get(combined_y_train[i], -1)
                # Occasionally, the random data split will result in classes being
                # absent from the training set. In these cases label will be -1 and
                # we include logit(0) instead of discarding
                logit = _logit(0) if label < 0 else _logit(conf[label])
                if i not in indices_train:
                    out_conf[i].append(logit)
                else:
                    in_conf[i].append(logit)
        return out_conf, in_conf

    def _compute_scores(
        self,
        combined_y_train: np.ndarray,
        combined_target_preds: np.ndarray,
        out_conf: dict[int, list[float]],
        in_conf: dict[int, list[float]],
    ) -> tuple[list[list[float]], int]:
        """Compute LiRA scores for each record."""
        logger.info("Computing scores")

        mia_scores: list[list[float]] = []
        n_normal: int = 0
        global_in_std: float = self._get_global_std(in_conf)
        global_out_std: float = self._get_global_std(out_conf)

        for i, label in enumerate(combined_y_train):
            logit = _logit(combined_target_preds[i, label])

            out_mean, out_std = self._get_mean_std(out_conf[i], global_out_std)
            in_mean, in_std = self._get_mean_std(in_conf[i], global_in_std)

            pr_in, pr_out = self._get_probabilities(
                logit=logit,
                out_mean=out_mean,
                out_std=out_std,
                in_mean=in_mean,
                in_std=in_std,
                mode=self.mode,
            )

            mia_scores.append([pr_in, pr_out])

            if get_p_normal(np.array(out_conf[i])) <= 0.05:
                n_normal += 1

            if self.report_individual:
                self._save_individual_result(
                    label,
                    logit,
                    out_conf[i],
                    pr_out,
                    out_mean,
                    out_std,
                    pr_in,
                    in_mean,
                    in_std,
                )

        return mia_scores, n_normal

    def _get_probabilities(
        self,
        logit: float,
        out_mean: float,
        out_std: float,
        in_mean: float,
        in_std: float,
        mode: str,
    ) -> tuple[float, float]:
        """Calculate probabilities based on the selected mode."""
        if mode == "offline":
            pr_out = norm.cdf(logit, loc=out_mean, scale=out_std + EPS)
            pr_in = 1 - pr_out
        elif mode == "online-carlini":
            pr_out = -norm.logpdf(logit, out_mean, out_std + EPS)
            pr_in = -norm.logpdf(logit, in_mean, in_std + EPS)
            pr_in = pr_in - pr_out
            pr_out = -pr_in
        elif mode == "offline-carlini":
            pr_out = -norm.logpdf(logit, out_mean, out_std + EPS)
            pr_in = -pr_out
        else:
            raise ValueError(f"Unsupported LiRA mode: {self.mode}")

        return float(pr_in), float(pr_out)

    def _save_attack_metrics(
        self,
        mia_scores: list[list[float]],
        n_train_rows: int,
        n_shadow_rows: int,
        n_normal: int,
    ) -> None:
        """Save attack metrics and individual results."""
        mia_clf = self._DummyClassifier()
        mia_scores_array = np.array(mia_scores)
        mia_labels = np.array([1] * n_train_rows + [0] * n_shadow_rows)
        y_pred_proba = mia_clf.predict_proba(mia_scores_array)

        self.attack_metrics = [metrics.get_metrics(y_pred_proba, mia_labels)]
        self.attack_metrics[-1]["n_normal"] = n_normal / (n_train_rows + n_shadow_rows)

        if self.report_individual:
            self.result["score"] = [score[1] for score in mia_scores]
            self.result["member"] = mia_labels
            self.attack_metrics[-1]["individual"] = self.result

    def _save_individual_result(
        self,
        label: int,
        target_logit: float,
        out_conf_sample: list[float],
        pr_out: float,
        out_mean: float,
        out_std: float,
        pr_in: float,
        in_mean: float,
        in_std: float,
    ) -> None:
        """Save individual record result."""
        out_p_norm = get_p_normal(np.array(out_conf_sample))

        self.result["label"].append(label)
        self.result["target_logit"].append(target_logit)
        self.result["out_p_norm"].append(out_p_norm)
        self.result["out_prob"].append(pr_out)
        self.result["out_mean"].append(out_mean)
        self.result["out_std"].append(out_std + EPS)

        if self.mode == "online-carlini":
            self.result["in_prob"].append(pr_in)
            self.result["in_mean"].append(in_mean)
            self.result["in_std"].append(in_std + EPS)

    def _get_mean_std(
        self, confidences: list[float], global_std: float
    ) -> tuple[float, float]:
        """Return the mean and standard deviation of a list of confidences."""
        scores: np.ndarray = np.array(confidences)
        mean: float = 0
        std: float = 0
        if not np.isnan(scores).all():
            mean = float(np.nanmean(scores))
            std = float(np.nanstd(scores))
        if self.fix_variance:
            std = global_std
        return mean, std

    def _get_global_std(self, confidences: dict[int, list[float]]) -> float:
        """Return the global standard deviation."""
        global_std: float = 0
        if self.fix_variance:
            # requires conversion from a dict of diff size proba lists
            arrays = list(confidences.values())
            combined = np.concatenate(arrays)
            if not np.isnan(combined).all():
                global_std = float(np.nanstd(combined))
        return global_std

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

    class _DummyClassifier:
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
        value to evaluate logit at.

    Returns
    -------
    float
        logit(p)

    Notes
    -----
    If `p` is close to 0 or 1, evaluating the log will result in numerical
    instabilities. This code thresholds `p` at `EPS` and `1 - EPS` where `EPS`
    defaults at 1e-16.
    """
    p = min(p, 1 - EPS)
    p = max(p, EPS)
    return np.log(p / (1 - p))
