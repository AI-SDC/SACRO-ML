"""Quantile Membership Inference Attack (CatBoost backend).

Scalable Membership Inference Attacks via Quantile Regression.
Bertran et al., NeurIPS 2023. https://arxiv.org/abs/2307.03694

Trains a CatBoost quantile regressor on non-member confidence scores
to learn per-sample membership thresholds. Supports Gaussian uncertainty
mode (RMSEWithUncertainty) and direct quantile mode.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fpdf import FPDF
from scipy.stats import norm

from sacroml import metrics
from sacroml.attacks import report, utils
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

try:  # pragma: no cover - exercised in integration tests
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - depends on environment
    CatBoostRegressor = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QMIAAttack(Attack):
    """Paper-faithful tabular QMIA attack.

    This first implementation focuses on binary tabular classification. It fits
    a regressor on public non-member examples (`X_test`, `y_test`) to predict a
    sample-dependent threshold for the true-label score. Membership evidence is
    then the margin between the observed score and the predicted threshold.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        alpha: float = 0.01,
        use_gaussian: bool = True,
        catboost_params: dict | None = None,
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
        use_gaussian : bool
            If true, fit CatBoost uncertainty regression and derive thresholds
            from a Gaussian quantile. Otherwise, fit a direct quantile regressor.
        catboost_params : dict or None
            Optional keyword arguments forwarded to ``CatBoostRegressor``.
        random_state : int
            Random seed for the QMIA regressor.
        report_individual : bool
            Whether to include per-record QMIA outputs in the report.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.alpha: float = alpha
        self.use_gaussian: bool = use_gaussian
        self.catboost_params: dict | None = catboost_params
        self.random_state: int = random_state
        self.report_individual: bool = report_individual
        self.result: dict = {}

    def __str__(self) -> str:
        """Return the name of the attack."""
        return "QMIA Attack"

    @classmethod
    def attackable(cls, target: Target) -> bool:  # pragma: no cover
        """Return whether a target can be assessed with QMIA."""
        if CatBoostRegressor is None:
            logger.info("WARNING: QMIA requires CatBoostRegressor to be installed.")
            return False

        if not (target.has_model() and target.has_data()):
            logger.info("WARNING: QMIA requires a loadable model and train/test data.")
            return False

        if not hasattr(target.model, "predict_proba"):
            logger.info("WARNING: QMIA requires predict_proba on the target model.")
            return False

        return True

    def _attack(self, target: Target) -> dict:
        """Run a QMIA attack."""
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1.")

        target = utils.check_and_update_dataset(target)

        proba_train = target.model.predict_proba(target.X_train)
        proba_test = target.model.predict_proba(target.X_test)

        train_scores = utils.qmia_hinge_score(proba_train, target.y_train)
        test_scores = utils.qmia_hinge_score(proba_test, target.y_test)

        x_test_with_y = np.column_stack((target.X_test, target.y_test))
        regressor = self._fit_regressor(x_test_with_y, test_scores)

        combined_x = np.vstack((target.X_train, target.X_test))
        combined_y = np.hstack((target.y_train, target.y_test))
        combined_x_with_y = np.column_stack((combined_x, combined_y))
        combined_scores = np.hstack((train_scores, test_scores))
        thresholds = self._predict_thresholds(regressor, combined_x_with_y)
        y_membership = utils.membership_labels(len(train_scores), len(test_scores))
        y_pred_proba = self._compute_membership_probs(combined_scores, thresholds)

        self.attack_metrics = [metrics.get_metrics(y_pred_proba, y_membership)]
        self.attack_metrics[0]["observed_public_fpr"] = float(
            np.mean(y_pred_proba[len(train_scores) :, 1] >= 0.5)
        )

        if self.report_individual:
            margins = combined_scores - thresholds
            self.result = {
                "score": combined_scores.tolist(),
                "threshold": thresholds.tolist(),
                "margin": margins.tolist(),
                "member_prob": y_pred_proba[:, 1].tolist(),
                "member": y_membership.tolist(),
            }
            self.attack_metrics[0]["individual"] = self.result

        output = self._make_report(target)
        self._write_report(output)
        return output

    def _default_catboost_params(self) -> dict[str, Any]:
        """Return stable default CatBoost parameters for QMIA."""
        base = {
            "depth": 4,
            "iterations": 50,
            "learning_rate": 0.05,
            "loss_function": "RMSEWithUncertainty",
            "random_seed": self.random_state,
            "verbose": False,
        }
        if not self.use_gaussian:
            base["loss_function"] = f"Quantile:alpha={1 - self.alpha}"
        return base

    def _fit_regressor(
        self, x_public: np.ndarray, public_scores: np.ndarray
    ) -> CatBoostRegressor:
        """Fit the tabular QMIA regressor."""
        if CatBoostRegressor is None:  # pragma: no cover
            raise ImportError("QMIAAttack requires the 'catboost' dependency.")

        params = self._default_catboost_params()
        if self.catboost_params is not None:
            params.update(self.catboost_params)

        regressor = CatBoostRegressor(**params)
        regressor.fit(x_public, public_scores)
        return regressor

    def _predict_thresholds(
        self, regressor: CatBoostRegressor, X: np.ndarray
    ) -> np.ndarray:
        """Predict per-row non-member thresholds."""
        if self.use_gaussian:
            raw_pred = np.asarray(regressor.predict(X, prediction_type="RawFormulaVal"))
            if raw_pred.ndim != 2 or raw_pred.shape[1] != 2:
                raise ValueError(
                    "Expected CatBoost uncertainty predictions with shape (n_rows, 2)."
                )
            mu = raw_pred[:, 0]
            # For RMSEWithUncertainty, RawFormulaVal returns the mean and the
            # log standard deviation for each row.
            sigma = np.exp(raw_pred[:, 1])
            sigma = np.maximum(sigma, utils.EPS)
            return norm.ppf(1 - self.alpha, loc=mu, scale=sigma)

        return np.asarray(regressor.predict(X), dtype=float)

    def _compute_membership_probs(
        self, scores: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Convert QMIA margins into [p_non_member, p_member] rows."""
        margins = np.asarray(scores - thresholds, dtype=float)
        return utils.margins_to_two_column_probs(margins)

    def _construct_metadata(self) -> None:
        """Construct the metadata object."""
        super()._construct_metadata()
        self.metadata["global_metrics"]["alpha"] = self.alpha
        self.metadata["global_metrics"]["use_gaussian"] = self.use_gaussian
        self.metadata["global_metrics"]["regressor_mode"] = (
            "gaussian_uncertainty" if self.use_gaussian else "direct_quantile"
        )
        self.metadata["global_metrics"]["qmia_score"] = "hinge_logit"
        self.metadata["global_metrics"]["public_slice"] = "target.X_test"
        self.metadata["global_metrics"]["membership_score_kind"] = (
            "sigmoid(score_minus_threshold)"
        )

    def _get_attack_metrics_instances(self) -> dict:
        """Construct per-instance attack metrics."""
        attack_metrics_instances = {
            f"instance_{idx}": metric for idx, metric in enumerate(self.attack_metrics)
        }
        return {"attack_instance_logger": attack_metrics_instances}

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report."""
        return report.create_mia_report(output)
