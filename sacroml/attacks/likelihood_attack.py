"""Likelihood testing scenario from https://arxiv.org/pdf/2112.03570.pdf."""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import sklearn
from fpdf import FPDF
from scipy.stats import norm, shapiro

from sacroml import metrics
from sacroml.attacks import report
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-16  # Used to avoid numerical issues


class LIRAAttack(Attack):
    """The main LiRA Attack class."""

    def __init__(  # pylint: disable=too-many-arguments
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

    def __str__(self):
        """Return the name of the attack."""
        return "LiRA Attack"

    def attack(self, target: Target) -> dict:
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
        shadow_clf = sklearn.base.clone(target.model)
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
        y_train_new = []
        classes = list(target.model.classes_)
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
        if len(y_test_new) != len(target.X_test):
            target.X_test = target.X_test[ok_pos, :]
        target.y_test = np.array(y_test_new, int)
        logger.info(
            "new y_test has values and counts: %s",
            np.unique(target.y_test, return_counts=True),
        )
        return target

    def _run(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        shadow_clf: sklearn.base.BaseEstimator,
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
        shadow_clf : sklearn.Model
            An sklearn classifier that will be trained to form the shadow models.
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

        # Combine target and shadow train, from which to sample datasets
        n_train_rows, _ = X_train.shape
        n_shadow_rows, _ = X_test.shape
        combined_x_train = np.vstack((X_train, X_test))
        combined_y_train = np.hstack((y_train, y_test))
        combined_target_preds = np.vstack((proba_train, proba_test))

        # Get the confidences of samples when in and not in the training set
        out_conf, in_conf = self._train_shadow_models(
            shadow_clf,
            combined_x_train,
            combined_y_train,
            n_train_rows,
        )

        # Get the LiRA scores, and how many confidences were normally distributed
        mia_scores, n_normal = self._compute_scores(
            combined_y_train, combined_target_preds, out_conf, in_conf
        )

        # Save metrics
        mia_clf = self._DummyClassifier()
        mia_scores = np.array(mia_scores)
        mia_labels = np.array([1] * n_train_rows + [0] * n_shadow_rows)
        y_pred_proba = mia_clf.predict_proba(mia_scores)
        self.attack_metrics = [metrics.get_metrics(y_pred_proba, mia_labels)]
        self.attack_metrics[-1]["n_normal"] = n_normal / (n_train_rows + n_shadow_rows)
        if self.report_individual:
            self.result["score"] = [score[1] for score in mia_scores]
            self.result["member"] = mia_labels
            self.attack_metrics[-1]["individual"] = self.result

        logger.info("Finished scenario")

    def _compute_scores(  # pylint: disable=too-many-locals
        self,
        combined_y_train: np.ndarray,
        combined_target_preds: np.ndarray,
        out_conf: dict[list[float]],
        in_conf: dict[list[float]],
    ) -> tuple[list[list[float]], int]:
        """Compute LiRA scores for each record."""
        logger.info("Computing scores")

        mia_scores: list[list[float]] = []
        n_normal: int = 0
        global_in_std: float = self._get_global_std(in_conf)
        global_out_std: float = self._get_global_std(out_conf)

        # score each record in the member and non-member sets
        for i, label in enumerate(combined_y_train):
            # get the target model behaviour on the record
            target_logit: float = _logit(combined_target_preds[i, label])
            # get behaviour observed with the record as a non-member
            out_mean, out_std = self._describe_conf(out_conf[i], global_out_std)
            # get behaviour observed with the record as a member
            in_mean, in_std = self._describe_conf(in_conf[i], global_in_std)
            # compare behaviour
            if self.mode == "offline":
                pr_out = norm.cdf(target_logit, loc=out_mean, scale=out_std + EPS)
                pr_in = 1 - pr_out
            elif self.mode == "online-carlini":
                pr_out = -norm.logpdf(target_logit, out_mean, out_std + EPS)
                pr_in = -norm.logpdf(target_logit, in_mean, in_std + EPS)
                # ratio
                pr_in = pr_in - pr_out
                pr_out = -pr_in
            elif self.mode == "offline-carlini":
                pr_out = -norm.logpdf(target_logit, out_mean, out_std + EPS)
                pr_in = -pr_out
            else:
                raise ValueError(f"Unsupported LiRA mode: {self.mode}")
            mia_scores.append([pr_in, pr_out])
            # test the non-member samples for normality
            out_p_norm = self._get_p_normal(np.array(out_conf[i]))
            if out_p_norm <= 0.05:
                n_normal += 1
            # save individual record result
            if self.report_individual:
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
        return mia_scores, n_normal

    def _describe_conf(
        self, confidences: list[float], global_std: float
    ) -> tuple[float, float]:
        """Return the mean and standard deviation of a list of confidences."""
        scores: np.ndarray = np.array(confidences)
        mean: float = 0
        std: float = 0
        if not np.isnan(scores).all():
            mean = np.nanmean(scores)
            std = np.nanstd(scores)
        if self.fix_variance:
            std = global_std
        return mean, std

    def _get_global_std(self, confidences: dict[str, list[float]]) -> float:
        """Return the global standard deviation."""
        global_std: float = 0
        if self.fix_variance:
            # requires conversion from a dict of diff size proba lists
            arrays = list(confidences.values())
            combined = np.concatenate(arrays)
            if not np.isnan(combined).all():
                global_std = np.nanstd(combined)
        return global_std

    def _get_p_normal(self, samples: np.ndarray) -> float:
        """Test whether a set of samples is normally distributed."""
        p_normal: float = np.NaN
        if np.nanvar(samples) > EPS:
            with contextlib.suppress(ValueError):
                _, p_normal = shapiro(samples)
        return p_normal

    def _train_shadow_models(  # pylint: disable=too-many-locals
        self,
        shadow_clf: sklearn.base.BaseEstimator,
        combined_x_train: np.ndarray,
        combined_y_train: np.ndarray,
        n_train_rows: int,
    ) -> tuple[dict, dict]:
        """Train shadow models and return confidence scores.

        Parameters
        ----------
        shadow_clf : sklearn.base.BaseEstimator
            An sklearn classifier that will be trained to form the shadow models.
        combined_x_train : np.ndarray
            Array of combined train and test features.
        combined_y_train : np.ndarray
            Array of combined train and test labels.
        n_train_rows : int
            Number of samples in the training set.

        Returns
        -------
        tuple[dict, dict]
            Dictionary of confidences when not in the training set.
            Dictionary of confidences when in the training set.
        """
        logger.info("Training shadow models")

        n_combined, _ = combined_x_train.shape
        out_conf: dict = {i: [] for i in range(n_combined)}
        in_conf: dict = {i: [] for i in range(n_combined)}
        indices: np.ndarray = np.arange(0, n_combined, 1)

        for model_idx in range(self.n_shadow_models):
            if model_idx % 10 == 0:
                logger.info("Trained %d models", model_idx)
            # Pick the indices to use for training this one
            np.random.seed(model_idx)  # Reproducibility
            these_idx = np.random.choice(indices, n_train_rows, replace=False)
            # Fit the shadow model
            shadow_clf.set_params(random_state=model_idx)
            shadow_clf.fit(
                combined_x_train[these_idx, :],
                combined_y_train[these_idx],
            )
            # map a class to a column
            class_map = {c: i for i, c in enumerate(shadow_clf.classes_)}
            # generate shadow confidences
            shadow_confidences = shadow_clf.predict_proba(combined_x_train)
            these_idx = set(these_idx)
            for i, conf in enumerate(shadow_confidences):
                # logit of the correct class
                label = class_map.get(combined_y_train[i], -1)
                # Occasionally, the random data split will result in classes being
                # absent from the training set. In these cases label will be -1 and
                # we include logit(0) instead of discarding
                logit = _logit(0) if label < 0 else _logit(conf[label])
                if i not in these_idx:
                    out_conf[i].append(logit)
                else:
                    in_conf[i].append(logit)
        return out_conf, in_conf

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
    if p > 1 - EPS:
        p = 1 - EPS
    p = max(p, EPS)
    return np.log(p / (1 - p))
