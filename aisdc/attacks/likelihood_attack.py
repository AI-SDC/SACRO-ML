"""Likelihood testing scenario from https://arxiv.org/pdf/2112.03570.pdf."""

# pylint: disable = too-many-branches

from __future__ import annotations

import logging

import numpy as np
import sklearn
from fpdf import FPDF
from scipy.stats import norm, shapiro

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
        self.run(
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
            f"{np.unique(target.y_train, return_counts=True)}",
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
            f"{np.unique(target.y_test, return_counts=True)}",
        )

        return target

    def run(  # pylint: disable=too-many-statements,too-many-arguments,too-many-locals
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
        n_train_rows, _ = X_train.shape
        n_shadow_rows, _ = X_test.shape
        n_combined = n_train_rows + n_shadow_rows
        indices = np.arange(0, n_combined, 1)

        # Combine taregt and shadow train, from which to sample datasets
        combined_x_train = np.vstack((X_train, X_test))
        combined_y_train = np.hstack((y_train, y_test))
        combined_target_preds = np.vstack((proba_train, proba_test))

        out_confidences = {i: [] for i in range(n_combined)}
        in_confidences = {i: [] for i in range(n_combined)}

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
                    out_confidences[i].append(logit)
                else:
                    in_confidences[i].append(logit)

        logger.info("Computing scores")

        # Do the test described in the paper in each case
        mia_scores = []
        mia_labels = [1] * n_train_rows + [0] * n_shadow_rows
        n_normal = 0

        if self.report_individual:
            result = {}
            result["score"] = []
            result["label"] = []
            result["target_logit"] = []
            result["out_p_norm"] = []
            result["out_prob"] = []
            result["out_mean"] = []
            result["out_std"] = []
            if self.mode == "online_carlini":
                result["in_prob"] = []
                result["in_mean"] = []
                result["in_std"] = []

        if self.fix_variance:  # compute global standard deviations
            # requires conversion from a dict of diff size proba lists
            out_arrays = list(out_confidences.values())
            out_combined = np.concatenate(out_arrays)
            global_out_std = 0
            if not np.isnan(out_combined).all():
                global_out_std = np.nanstd(out_combined)
            in_arrays = list(in_confidences.values())
            in_combined = np.concatenate(in_arrays)
            global_in_std = 0
            if not np.isnan(in_combined).all():
                global_in_std = np.nanstd(in_combined)

        # scpre each record in the member and non-member sets
        for i in range(n_combined):
            # get the target model behaviour on the record
            label = combined_y_train[i]
            target_conf = combined_target_preds[i, label]
            target_logit = _logit(target_conf)

            # compare the target logit with behaviour observed as a non-member
            out_scores = np.array(out_confidences[i])
            out_mean = 0
            out_std = 0
            if not np.isnan(out_scores).all():
                out_mean = np.nanmean(out_scores)
                out_std = np.nanstd(out_scores)
            if self.fix_variance:
                out_std = global_out_std
            out_prob = -norm.logpdf(target_logit, out_mean, out_std + EPS)

            # test the non-member samples for normality
            out_p_norm = np.NaN
            if np.nanvar(out_scores) > EPS:
                try:
                    _, out_p_norm = shapiro(out_scores)
                    if out_p_norm <= 0.05:
                        n_normal += 1
                except ValueError:  # pragma: no cover
                    pass

            if self.mode == "offline":
                # probability of observing a confidence as high as the target model's
                # under the null-hypothesis that the target point is a non-member
                out_prob = norm.cdf(target_logit, loc=out_mean, scale=out_std + EPS)
                mia_scores.append([1 - out_prob, out_prob])
            elif self.mode == "online-carlini":
                # compare the target logit with behaviour observed as a member
                in_scores = np.array(in_confidences[i])
                in_mean = 0
                in_std = 0
                if not np.isnan(in_scores).all():
                    in_mean = np.nanmean(in_scores)
                    in_std = np.nanstd(in_scores)
                if self.fix_variance:
                    in_std = global_in_std
                in_prob = -norm.logpdf(target_logit, in_mean, in_std + EPS)
                # compute the likelihood ratio
                prob = in_prob - out_prob
                mia_scores.append([prob, -prob])
            elif self.mode == "offline-carlini":
                # probability the record is not a non-member
                prob = out_prob
                mia_scores.append([-prob, prob])
            else:
                raise ValueError(f"Unsupported LiRA mode: {self.mode}")

            if self.report_individual:
                result["label"].append(label)
                result["target_logit"].append(target_logit)
                result["out_p_norm"].append(out_p_norm)
                result["out_prob"].append(out_prob)
                result["out_mean"].append(out_mean)
                result["out_std"].append(out_std + EPS)
                if self.mode == "online_carlini":
                    result["in_prob"].append(in_prob)
                    result["in_mean"].append(in_mean)
                    result["in_std"].append(in_std + EPS)

        # save metrics
        mia_clf = DummyClassifier()
        mia_scores = np.array(mia_scores)
        mia_labels = np.array(mia_labels)
        y_pred_proba = mia_clf.predict_proba(mia_scores)
        self.attack_metrics = [metrics.get_metrics(y_pred_proba, mia_labels)]
        self.attack_metrics[-1]["n_normal"] = n_normal / n_combined
        if self.report_individual:
            result["score"] = [score[1] for score in mia_scores]
            result["member"] = mia_labels
            self.attack_metrics[-1]["individual"] = result

        logger.info("Finished scenario")

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
