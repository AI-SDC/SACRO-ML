"""Quantile regression membership inference attack.

Scalable Membership Inference Attacks via Quantile Regression.
Bertran et al., NeurIPS 2023. https://arxiv.org/abs/2307.03694

Key idea: instead of training N shadow models to estimate the distribution
of confidence scores, train a single quantile regression model on the
non-member (reference) set. For each sample x, this model predicts the
threshold q_alpha(x) below which (1-alpha)% of non-member scores fall.
A sample is predicted a member if its score exceeds that threshold.

This gives a calibrated false positive rate equal to alpha by construction,
requires no knowledge of the target model architecture, and is truly
black-box (only predict_proba access is needed).
"""

from __future__ import annotations

import logging

import numpy as np
from fpdf import FPDF
from sklearn.ensemble import GradientBoostingRegressor

from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QMIAAttack(Attack):
    """Membership inference attack via per-sample quantile regression.

    Trains one quantile regression model on the reference (non-member) set
    to learn a per-sample decision threshold. A point is predicted as a
    member if its confidence score exceeds its predicted threshold.

    The false positive rate is calibrated to alpha by construction: since
    q_alpha(x) estimates the (1-alpha)-quantile of non-member scores
    conditioned on x, exactly alpha% of non-members will score above their
    own threshold.

    Parameters
    ----------
    alpha : float
        Target false positive rate. Must be in (0, 1). Default 0.1.
    n_estimators : int
        Number of boosting stages for the quantile regression model.
        Default 100.
    output_dir : str
        Directory where output files are written. Default "outputs".
    write_report : bool
        Whether to generate JSON and PDF reports. Default True.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        n_estimators: int = 100,
        output_dir: str = "outputs",
        write_report: bool = True,
    ) -> None:
        """Construct QMIAAttack Object."""
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.alpha = float(alpha)
        self.n_estimators: int = n_estimators
        self.quantile_model: GradientBoostingRegressor | None = None

    def __str__(self) -> str:
        """Return the name of the attack."""
        return """QMIA Attack"""

    @classmethod
    def attackable(cls, target: Target) -> bool:
        """Return whether a target can be assessed with QMIAAttack.

        Requires a target with a loaded model and all four data splits
        (X_train, y_train, X_test, y_test). No architecture information
        is needed - only black-box predict_proba access.

        Parameters
        ----------
        target : Target
            The target object to check.

        Returns
        -------
        bool
            True if the attack can proceed, False otherwise.
        """
        if target.has_model() and target.has_data():
            return True
        logging.warning(
            "QMIAAttack requires a target with a loaded model and all data splits."
        )
        return False

    def _get_confidence_scores(
        self,
        target: Target,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Return the model's confidence on the true label for each sample.

        This is the score used by the attack: predict_proba(X)[i, y[i]].

        Parameters
        ----------
        target : Target
            The target object containing the wrapped model.
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True labels.

        Returns
        -------
        np.ndarray
            1-D array of confidence scores, one per sample.
        """

    def _train_quantile_model(
        self,
        x_ref: np.ndarray,
        scores_ref: np.ndarray,
    ) -> None:
        """Fit the quantile regression model on the reference (non-member) set.

        Trains a GradientBoostingRegressor with quantile loss at level
        (1 - alpha), learning the per-sample threshold below which
        (1-alpha)% of non-member scores fall.

        Parameters
        ----------
        X_ref : np.ndarray
            Features of reference (non-member) samples.
        scores_ref : np.ndarray
            Confidence scores of reference samples.
        """

    def _attack(
        self,
        target: Target,
    ) -> dict:
        """Run the QMIA attack on the target model.

        Steps:
        1. Score the reference set (X_test), these are non-members.
        2. Train the quantile model on (X_test, scores_test).
        3. For every sample (train + test), predict its per-sample threshold.
        4. Predict member if score > threshold.
        5. Compute metrics and write the report.

        Parameters
        ----------
        target : Target
            The target object containing the model and data.

        Returns
        -------
        dict
            Attack report dictionary.
        """

    def _get_attack_metrics_instances(
        self,
    ) -> dict:
        """Return attack metrics in the standard attack_instance_logger format.

        Returns
        -------
        dict
            Metrics dictionary structured as expected by the report formatter.
        """

    def _construct_metadata(
        self,
    ) -> dict:
        """Extend base metadata with QMIA-specific global metrics."""

    def _make_pdf(self, output) -> FPDF:
        """Construct a PDF report for the attack results.

        Parameters
        ----------
        output : dict
            The output dictionary containing attack results and metadata.

        Returns
        -------
        FPDF
            The constructed PDF object.
        """
