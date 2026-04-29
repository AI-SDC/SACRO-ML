"""Quantile Membership Inference Attack (QMIA).

Scalable Membership Inference Attacks via Quantile Regression.
Bertran et al., NeurIPS 2023. https://arxiv.org/abs/2307.03694

Trains a histogram-based quantile regressor on non-member hinge scores to
learn per-sample membership thresholds.  A sample is predicted as a member
when its observed score exceeds the predicted threshold.

Uses ``HistGradientBoostingRegressor`` rather than ``GradientBoostingRegressor``
for its histogram-based splitting algorithm, which is faster on large datasets.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

import numpy as np
from fpdf import FPDF
from sklearn.ensemble import HistGradientBoostingRegressor

from sacroml import metrics
from sacroml.attacks import report, utils
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target
from sacroml.version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QMIAAttack(Attack):
    """Paper-faithful tabular QMIA attack.

    This implementation focuses on tabular classification. It fits a quantile
    regressor on public non-member examples (``X_test``, ``y_test``) to predict
    a sample-dependent threshold for the hinge score. Membership evidence is
    then the margin between the observed score and the predicted threshold.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        alpha: float = 0.01,
        p_thresh: float = 0.05,
        max_iter: int = 100,
        random_state: int = 0,
        report_individual: bool = False,
    ) -> None:
        """Construct a QMIA attack.

        Parameters
        ----------
        output_dir : str
            Name of the directory where outputs are stored.
        write_report : bool
            Whether to generate a JSON and PDF report.
        alpha : float
            Target false-positive rate for the public non-member distribution.
        p_thresh : float
            P-value threshold for AUC significance reporting.
        max_iter : int
            Maximum number of boosting iterations for the quantile regressor.
        random_state : int
            Random seed for the QMIA regressor.
        report_individual : bool
            Whether to include per-record QMIA outputs in the report.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.alpha: float = alpha
        self.p_thresh: float = p_thresh
        self.max_iter: int = max_iter
        self.random_state: int = random_state
        self.report_individual: bool = report_individual
        self.quantile_model: HistGradientBoostingRegressor | None = None

    def __str__(self) -> str:
        """Return the name of the attack."""
        return "QMIA Attack"

    @classmethod
    def attackable(cls, target: Target) -> bool:
        """Return whether a target can be assessed with QMIA.

        Parameters
        ----------
        target : Target
            The target to assess.

        Returns
        -------
        bool
            True if the target has a model and data.
        """
        if not (
            target.has_model()
            and target.has_data()
            and hasattr(target.model, "predict_proba")
        ):
            logger.warning(
                "QMIA requires a model with predict_proba and train/test data."
            )
            return False
        return True

    def _attack(self, target: Target) -> dict:
        """Run a QMIA attack."""
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1.")

        target = utils.check_and_update_dataset(target)

        proba_train = target.model.predict_proba(target.X_train)
        proba_test = target.model.predict_proba(target.X_test)
        if not (np.isfinite(proba_train).all() and np.isfinite(proba_test).all()):
            output = self._make_failed_output(
                target,
                "target.model.predict_proba returned non-finite values; "
                "QMIA cannot score rows with NaN/Inf probabilities.",
            )
            try:
                self._write_report(output)
            except OSError:
                logger.warning("Could not write failed report.")
            return output

        train_scores = utils.qmia_hinge_score(proba_train, target.y_train)
        test_scores = utils.qmia_hinge_score(proba_test, target.y_test)

        # Train quantile regressor on non-member scores; quantile = 1 - alpha
        # so that a fraction alpha of non-members exceed their own threshold.
        # Early stopping cuts fit time ~20-40% on large n; below 1000 the 10%
        # validation split is too noisy and can stop training too early.
        x_test_with_y = np.column_stack((target.X_test, target.y_test))
        use_early_stopping: bool = len(test_scores) >= 1000
        self.quantile_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=1.0 - self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=use_early_stopping,
        )
        self.quantile_model.fit(x_test_with_y, test_scores)

        combined_x = np.vstack((target.X_train, target.X_test))
        combined_y = np.hstack((target.y_train, target.y_test))
        combined_x_with_y = np.column_stack((combined_x, combined_y))
        combined_scores = np.hstack((train_scores, test_scores))
        thresholds = self.quantile_model.predict(combined_x_with_y)

        # HGBR silently returns a constant predictor on degenerate inputs;
        # catch that here so the attack cannot report a plausible-but-wrong AUC.
        threshold_spread: float = float(np.std(thresholds))
        score_spread: float = float(np.std(test_scores))
        if threshold_spread < max(1e-10, 1e-6 * score_spread):
            output = self._make_failed_output(
                target,
                "QMIA quantile regressor degenerated to a near-constant "
                f"predictor (threshold std={threshold_spread:.3e}, score "
                f"std={score_spread:.3e}). Likely causes: target model "
                "produces uniform hinge scores (e.g., DummyClassifier), "
                "non-member set too small, or target output lacks "
                "information. Attack cannot produce meaningful results.",
            )
            try:
                self._write_report(output)
            except OSError:
                logger.warning("Could not write failed report.")
            return output

        y_membership: np.ndarray = utils.membership_labels(
            len(train_scores), len(test_scores)
        )
        y_pred_proba: np.ndarray = self._compute_membership_probs(
            combined_scores, thresholds
        )

        self.attack_metrics = [metrics.get_metrics(y_pred_proba, y_membership)]
        # Non-member predictions from the public slice: "member" = margin > 0.
        obs_fpr: float = float(np.mean(y_pred_proba[len(train_scores) :, 1] > 0.0))
        self.attack_metrics[0]["observed_public_fpr"] = obs_fpr

        # QR-MIA's core calibration claim: obs_fpr should track alpha.
        fpr_tolerance: float = max(2.0 * self.alpha, 0.05)
        calibration_ok: bool = abs(obs_fpr - self.alpha) <= fpr_tolerance
        self.attack_metrics[0]["calibration_ok"] = calibration_ok
        if not calibration_ok:
            logger.warning(
                "QMIA calibration deviated from target: "
                "observed_public_fpr=%.4f vs alpha=%.4f (tolerance=%.4f). "
                "Attack results may be unreliable.",
                obs_fpr,
                self.alpha,
                fpr_tolerance,
            )

        if self.report_individual:
            margins = combined_scores - thresholds
            individual = {
                "score": combined_scores.tolist(),
                "threshold": thresholds.tolist(),
                "margin": margins.tolist(),
                "member_prob": y_pred_proba[:, 1].tolist(),
                "member": y_membership.tolist(),
            }
            self.attack_metrics[0]["individual"] = individual

        output = self._make_report(target)
        output["status"] = "success"
        self._write_report(output)
        return output

    def _compute_membership_probs(
        self, scores: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Convert QMIA margins into [p_non_member, p_member] rows."""
        margins = np.asarray(scores - thresholds, dtype=float)
        return utils.margins_to_two_column_probs(margins)

    def _make_failed_output(self, target: Target, fail_reason: str) -> dict:
        """Build output dict for an attack that could not produce results."""
        self.metadata = {
            "sacroml_version": __version__,
            "attack_name": str(self),
            "attack_params": self.get_params(),
            "global_metrics": {},
        }
        if target.model is not None:
            self.metadata["target_model"] = target.model.model_name
            self.metadata["target_model_params"] = target.model.model_params
            self.metadata["target_train_params"] = target.model.train_params
        return {
            "log_id": str(uuid.uuid4()),
            "log_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "metadata": self.metadata,
            "status": "failed",
            "fail_reason": fail_reason,
        }

    def _construct_metadata(self) -> None:
        """Construct the metadata object."""
        super()._construct_metadata()
        m = self.attack_metrics[0]
        n_pos = m["n_pos_test_examples"]
        n_neg = m["n_neg_test_examples"]
        p_val, std = metrics.auc_p_val(m["AUC"], n_pos, n_neg)
        self.metadata["global_metrics"]["alpha"] = self.alpha
        self.metadata["global_metrics"]["p_thresh"] = self.p_thresh
        self.metadata["global_metrics"]["AUC_sig"] = (
            f"AUC p-value: {p_val:.4f} (significant: {p_val < self.p_thresh})"
        )
        self.metadata["global_metrics"]["null_auc_3sd_range"] = [
            0.5 - 3 * std,
            0.5 + 3 * std,
        ]
        self.metadata["global_metrics"]["TPR"] = m["TPR"]
        self.metadata["global_metrics"]["FPR"] = m["FPR"]
        self.metadata["global_metrics"]["Advantage"] = m["Advantage"]

    def _get_attack_metrics_instances(self) -> dict:
        """Construct per-instance attack metrics."""
        attack_metrics_instances = {
            f"instance_{idx}": metric for idx, metric in enumerate(self.attack_metrics)
        }
        return {"attack_instance_logger": attack_metrics_instances}

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report."""
        if output.get("status") == "failed":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_xy(0, 0)
            report.title(pdf, "Quantile Regression Attack Report")
            report.subtitle(pdf, "Attack Status: Failed")
            report.line(pdf, output.get("fail_reason", "Unknown reason."))
            return pdf
        return report.create_qmia_report(output)
