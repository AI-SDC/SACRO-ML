"""Training data in model attack.

Detects when instance-based models (SVM, kNN) store training data by extracting
support vectors or stored neighbors, comparing them to training data, and
reporting matches with mitigation guidance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fpdf import FPDF
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sacroml.attacks import report
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number of example matches to report for visibility
N_EXAMPLE_MATCHES = 10

# Tolerance for float comparison when matching vectors
RTOL = 1e-5
ATOL = 1e-8

MITIGATIONS = [
    "Not relevant for this dataset (by agreement with TRE)",
    "Use DP-version (e.g., SafeSVC for SVM)",
    "Describe alternatives you've considered",
    "Relying on documentation and training",
]

# DP-related class names or step names that indicate embedding in DP-space
DP_INDICATORS = ("safesvc", "dpsvc", "dp_svc", "dp", "differential", "private")


def _get_final_estimator(model: BaseEstimator) -> BaseEstimator:
    """Get the final estimator, unwrapping Pipeline if needed."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _has_dp_embedding(model: BaseEstimator) -> bool:
    """Check if model uses DP embedding (Pipeline with DP step or DP estimator)."""
    # Check if final estimator is DP variant
    final = _get_final_estimator(model)
    name = type(final).__name__.lower()
    if any(ind in name for ind in DP_INDICATORS):
        return True

    # Check Pipeline steps for DP-related names
    if isinstance(model, Pipeline):
        for step_name, _ in model.steps[:-1]:
            if any(ind in str(step_name).lower() for ind in DP_INDICATORS):
                return True
    return False


def _is_attackable_model(model: BaseEstimator) -> bool:
    """Return True if model is SVC or KNeighborsClassifier (excluding DP variants)."""
    if _has_dp_embedding(model):
        return False
    final = _get_final_estimator(model)
    if isinstance(final, SVC):
        # kernel="precomputed" stores gram matrix, not input-space vectors
        return getattr(final, "kernel", "") != "precomputed"
    return isinstance(final, KNeighborsClassifier)


def _to_dense(arr: np.ndarray) -> np.ndarray:
    """Convert array to dense ndarray (handles sparse matrices)."""
    if hasattr(arr, "toarray"):
        return np.asarray(arr.toarray())
    return np.asarray(arr)


def _get_stored_vectors_and_train_data(
    model: BaseEstimator, X_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Extract stored vectors from model and get training data in same space.

    Returns
    -------
    stored : np.ndarray
        Vectors stored in the model (support_vectors_ or _fit_X).
    train_data : np.ndarray
        Training data in the same space as stored vectors.
    model_type : str
        "SVC" or "KNeighborsClassifier".
    """
    final = _get_final_estimator(model)

    if isinstance(model, Pipeline) and len(model.steps) > 1:
        # Transform X_train through preprocessing steps to match stored space
        train_data = model[:-1].transform(X_train)
    else:
        train_data = X_train

    if isinstance(final, SVC):
        stored = final.support_vectors_
        # SVC with kernel="precomputed" stores gram matrix rows, not input space
        if getattr(final, "kernel", "") == "precomputed":
            raise ValueError(
                "SVC with kernel='precomputed' stores kernel-space vectors; "
                "cannot compare to raw training data."
            )
        return _to_dense(stored), _to_dense(train_data), "SVC"
    if isinstance(final, KNeighborsClassifier):
        stored = final._fit_X
        return _to_dense(stored), _to_dense(train_data), "KNeighborsClassifier"

    raise ValueError(f"Unsupported model type: {type(final)}")


def _values_preview(arr: np.ndarray, idx: int, max_vals: int = 5) -> list:
    """Get preview of values for a row, handling sparse matrices."""
    row = arr[idx]
    if hasattr(row, "toarray"):
        row = row.toarray().flatten()
    else:
        row = np.asarray(row).flatten()
    return row.tolist()[:max_vals]


def _find_matches(stored: np.ndarray, train_data: np.ndarray) -> list[dict[str, Any]]:
    """
    Find rows in stored that match rows in train_data.

    Returns list of dicts with train_idx, stored_idx, values_preview.
    """
    matches = []
    n_train = train_data.shape[0]
    n_stored = stored.shape[0]

    for s_idx in range(n_stored):
        sv = stored[s_idx : s_idx + 1]
        for t_idx in range(n_train):
            if np.allclose(sv, train_data[t_idx : t_idx + 1], rtol=RTOL, atol=ATOL):
                matches.append(
                    {
                        "train_idx": int(t_idx),
                        "stored_idx": int(s_idx),
                        "values_preview": _values_preview(stored, s_idx),
                    }
                )
                break

    return matches


class TrainingDataInModelAttack(Attack):
    """Attack that detects when instance-based models contain training data.

    Extracts support vectors (SVM) or stored neighbors (kNN), compares them
    to training data, and reports matches with mitigation guidance.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        n_example_matches: int = N_EXAMPLE_MATCHES,
    ) -> None:
        """Construct an object to execute a training data in model attack.

        Parameters
        ----------
        output_dir : str
            Name of the directory where outputs are stored.
        write_report : bool
            Whether to generate a JSON and PDF report.
        n_example_matches : int
            Number of example matches to report for visibility.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.n_example_matches = n_example_matches

    def __str__(self) -> str:
        """Return the name of the attack."""
        return "Training Data in Model Attack"

    @classmethod
    def attackable(cls, target: Target) -> bool:
        """Return whether a target can be assessed with this attack."""
        if not target.has_model():
            logger.info("target.model is missing, cannot proceed")
            return False
        if not target.has_data():
            logger.info("target data (X_train, y_train, etc.) is missing")
            return False

        model = target.model.model
        if model is None:
            return False

        # Only sklearn BaseEstimator supported (no PyTorch)
        if not isinstance(model, BaseEstimator):
            logger.info("TrainingDataInModelAttack requires sklearn model")
            return False

        if not _is_attackable_model(model):
            logger.info(
                "Model type %s is not SVC or KNeighborsClassifier (or is DP variant)",
                type(_get_final_estimator(model)).__name__,
            )
            return False

        return True

    def _attack(self, target: Target) -> dict:
        """Run the training data in model attack."""
        model = target.model.model
        X_train = target.X_train

        dp_space_caveat = _has_dp_embedding(model)
        try:
            stored, train_data, model_type = _get_stored_vectors_and_train_data(
                model, X_train
            )
        except (AttributeError, ValueError) as e:
            logger.warning(
                "Cannot extract stored vectors (model may be unfitted or use "
                "unsupported config): %s",
                e,
            )
            return {}

        n_stored = stored.shape[0]
        n_training = train_data.shape[0]

        if model_type == "KNeighborsClassifier":
            # kNN stores full training set; all rows match
            matches = [
                {
                    "train_idx": i,
                    "stored_idx": i,
                    "values_preview": _values_preview(stored, i),
                }
                for i in range(min(n_stored, n_training))
            ]
            n_matches = n_stored
        else:
            matches = _find_matches(stored, train_data)
            n_matches = len(matches)

        contains_training_data = n_matches > 0
        example_matches = matches[: self.n_example_matches]

        self.attack_metrics = {
            "model_type": model_type,
            "contains_training_data": contains_training_data,
            "n_stored": n_stored,
            "n_training": n_training,
            "n_matches": n_matches,
            "dp_space_caveat": dp_space_caveat,
            "example_matches": example_matches,
            "mitigations": MITIGATIONS,
        }

        output = self._make_report(target)
        if self.write_report:
            self._write_report(output)

        return output

    def _construct_metadata(self) -> None:
        """Construct metadata for the report."""
        super()._construct_metadata()
        if self.attack_metrics:
            self.metadata["global_metrics"] = self.attack_metrics

    def _get_attack_metrics_instances(self) -> dict:
        """Return attack metrics in the format expected by the report."""
        if not self.attack_metrics:
            return {}
        return {
            "attack_instance_logger": {
                "instance_0": self.attack_metrics,
            }
        }

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report."""
        return report.create_training_data_report(output)
