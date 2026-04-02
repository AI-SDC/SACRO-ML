"""Instance-based model attack.

Detects when instance-based models (SVM, kNN) store training data as part
of their model parameters (support vectors or neighbors), confirming a
concrete data leakage pathway.

This module provides the `InstanceBasedAttack` class, which:
- Checks if a model is an instance-based type (SVM or kNN)
- Extracts the stored instances (support vectors or neighbors)
- Compares them to the training data to confirm data leakage
- Reports matching examples and available mitigations
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from fpdf import FPDF
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM

try:
    from sklearn.pipeline import Pipeline
except ImportError:  # pragma: no cover
    Pipeline = None

from sacroml.attacks import report
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SVM_TYPES = (SVC, NuSVC, SVR, NuSVR, OneClassSVM)
KNN_TYPES = (KNeighborsClassifier, KNeighborsRegressor)

_INTRODUCTION = (
    "This report provides the results of an instance-based model data "
    "leakage check. Some model types -- notably Support Vector Machines "
    "(SVM) and k-Nearest Neighbours (kNN) -- store training data points "
    "as part of their fitted model parameters. SVM models store 'support "
    "vectors' (a subset of training records that define the decision "
    "boundary), while kNN models store the entire training dataset. "
    "When such a model is released from a Trusted Research Environment "
    "(TRE), these stored data points can be directly extracted, "
    "constituting a concrete data leakage risk.\n This attack extracts "
    "any stored instances from the model, compares them against the "
    "original training data, and reports whether matches are found."
)

_GLOSSARY = {
    "Support Vectors": (
        "In SVM models, support vectors are the training data points that "
        "lie closest to the decision boundary. These are stored verbatim "
        "inside the fitted model and can be extracted directly."
    ),
    "kNN Storage": (
        "k-Nearest Neighbours models store the entire training dataset "
        "internally, as predictions are made by finding the k closest "
        "stored points to a new input."
    ),
    "DP Variant": (
        "A differentially private variant of a model adds calibrated "
        "noise to break the direct link between stored model parameters "
        "and the original training data, mitigating the leakage risk."
    ),
    "Storage Fraction": (
        "The proportion of training data points stored inside the model. "
        "For SVM this is typically a subset; for kNN this is 1.0 (all data)."
    ),
    "Match Fraction": (
        "The proportion of stored instances that exactly match a training "
        "data point. A high match fraction confirms data leakage."
    ),
}


@dataclass
class InstanceBasedAttackResults:
    """Results of an instance-based model attack."""

    model_type: str
    is_instance_based: bool
    is_dp_safe: bool
    n_stored_instances: int
    n_training_samples: int
    storage_fraction: float
    n_matched: int
    n_checked: int
    match_fraction: float
    example_matches: list[dict]
    data_leakage_confirmed: bool
    mitigations: list[str]
    details: dict | None = None


class InstanceBasedAttack(Attack):
    """Detect training data stored in instance-based model parameters.

    Instance-based models such as SVM and kNN store training data points
    (support vectors or all neighbors) inside the fitted model. This attack
    extracts those stored instances, compares them to the training data, and
    reports whether the model leaks training data.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        n_examples: int = 10,
        atol: float = 1e-8,
    ) -> None:
        """Construct an instance-based model attack.

        Parameters
        ----------
        output_dir : str
            Name of a directory to write outputs.
        write_report : bool
            Whether to generate a JSON and PDF report.
        n_examples : int
            Maximum number of matching examples to include in the report.
        atol : float
            Absolute tolerance for floating-point comparison when matching
            stored instances to training data.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.n_examples = n_examples
        self.atol = atol
        self.results: InstanceBasedAttackResults | None = None

    def __str__(self) -> str:
        """Return the name of the attack."""
        return "Instance-Based Model Attack"

    @classmethod
    def attackable(cls, target: Target) -> bool:
        """Return whether a target can be assessed with this attack.

        Requires a model and training data. Non-instance-based models are
        handled gracefully (reported as not applicable).
        """
        if not target.has_model():
            logger.info("target.model is missing, cannot proceed")
            return False
        if not target.has_data():
            logger.info("target data is missing, cannot proceed")
            return False
        return True

    @staticmethod
    def _unwrap_model(model):
        """Extract the final estimator and preprocessor from a Pipeline.

        Returns
        -------
        tuple
            (final_estimator, preprocessor_pipeline_or_None)
            If the model is a Pipeline with preprocessing steps, returns
            a Pipeline of just the preprocessing steps so X_train can be
            transformed to the same space as the stored instances.
        """
        if Pipeline is not None and isinstance(model, Pipeline):
            final_estimator = model.steps[-1][1]
            preprocessor = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
            return final_estimator, preprocessor
        return model, None

    def _compare_instances(
        self,
        stored_instances: np.ndarray,
        stored_indices: np.ndarray | None,
        X_train: np.ndarray,
    ) -> tuple[int, list[dict]]:
        """Compare stored model instances against training data.

        Parameters
        ----------
        stored_instances : np.ndarray
            Data points stored inside the model.
        stored_indices : np.ndarray or None
            Indices of stored instances into the original training data.
        X_train : np.ndarray
            The training data to compare against.

        Returns
        -------
        n_matched : int
            Number of stored instances that match training data.
        example_matches : list[dict]
            Details of the first n_examples matches.
        """
        n_matched = 0
        example_matches: list[dict] = []

        for i, stored_row in enumerate(stored_instances):
            matched = False
            match_index = None

            # Try index-based direct comparison first
            if stored_indices is not None and i < len(stored_indices):
                idx = int(stored_indices[i])
                if 0 <= idx < len(X_train) and np.allclose(
                    stored_row, X_train[idx], atol=self.atol
                ):
                    matched = True
                    match_index = idx

            # Fallback: search through training data
            if not matched:
                for j in range(len(X_train)):
                    if np.allclose(stored_row, X_train[j], atol=self.atol):
                        matched = True
                        match_index = j
                        break

            if matched:
                n_matched += 1
                if len(example_matches) < self.n_examples:
                    n_preview = min(5, stored_row.shape[0])
                    example_matches.append(
                        {
                            "stored_index": i,
                            "training_index": match_index,
                            "stored_values": stored_row[:n_preview].tolist(),
                            "training_values": (
                                X_train[match_index][:n_preview].tolist()
                                if match_index is not None
                                else None
                            ),
                        }
                    )

        return n_matched, example_matches

    def _build_mitigations(
        self, is_svm: bool, is_knn: bool, is_dp_safe: bool
    ) -> list[str]:
        """Build the list of available mitigations."""
        mitigations: list[str] = []

        if is_dp_safe:
            mitigations.append(
                "This model uses a DP-safe variant. The stored parameters are "
                "in a transformed/noisy space and do not directly correspond "
                "to training data points."
            )

        if is_svm:
            mitigations.append(
                "Use a differentially private SVM variant (e.g., DPSVC from "
                "sacroml.safemodel) which adds noise to the separating "
                "hyperplane in a transformed feature space, breaking the "
                "direct link between support vectors and training data."
            )

        if is_knn:
            mitigations.append(
                "kNN models inherently store all training data. Consider "
                "using a model type that does not require storing training "
                "instances (e.g., decision tree, random forest, or neural "
                "network)."
            )

        mitigations.append(
            "By agreement with the TRE, this risk may be deemed 'not "
            "relevant' for this particular dataset if the data is already "
            "public or low-sensitivity."
        )

        return mitigations

    def _attack(self, target: Target) -> dict:
        """Run the instance-based model attack.

        Parameters
        ----------
        target : Target
            The target object containing the model and data.

        Returns
        -------
        dict
            Attack report dictionary.
        """
        raw_model, preprocessor = self._unwrap_model(target.model.model)
        model_type = type(raw_model).__name__

        is_svm = isinstance(raw_model, SVM_TYPES)
        is_knn = isinstance(raw_model, KNN_TYPES)
        is_instance_based = is_svm or is_knn

        # Lazy import to avoid circular dependency
        from sacroml.safemodel.classifiers.dp_svc import DPSVC  # noqa: PLC0415

        is_dp_safe = isinstance(raw_model, DPSVC)

        X_train = target.X_train
        # If model was inside a Pipeline with preprocessing, transform
        # X_train to the same space as the stored instances
        if preprocessor is not None:
            X_train = preprocessor.transform(X_train)
        n_training = len(X_train)

        if not is_instance_based:
            logger.info(
                "Model type %s is not instance-based, no data leakage risk "
                "from stored instances.",
                model_type,
            )
            self.results = InstanceBasedAttackResults(
                model_type=model_type,
                is_instance_based=False,
                is_dp_safe=False,
                n_stored_instances=0,
                n_training_samples=n_training,
                storage_fraction=0.0,
                n_matched=0,
                n_checked=0,
                match_fraction=0.0,
                example_matches=[],
                data_leakage_confirmed=False,
                mitigations=[],
            )
            output = self._make_report(target)
            self._write_report(output)
            return output

        # Extract stored instances
        stored_instances = None
        stored_indices = None

        if is_svm:
            if hasattr(raw_model, "support_vectors_"):
                stored_instances = np.asarray(raw_model.support_vectors_)
                stored_indices = np.asarray(raw_model.support_)
            else:
                logger.warning(
                    "SVM model %s does not have support_vectors_ attribute. "
                    "It may not be fitted.",
                    model_type,
                )

        if is_knn:
            if hasattr(raw_model, "_fit_X"):
                stored_instances = np.asarray(raw_model._fit_X)
                stored_indices = np.arange(len(stored_instances))
            else:
                logger.warning(
                    "kNN model %s does not have _fit_X attribute. "
                    "It may not be fitted.",
                    model_type,
                )

        if stored_instances is None:
            self.results = InstanceBasedAttackResults(
                model_type=model_type,
                is_instance_based=True,
                is_dp_safe=is_dp_safe,
                n_stored_instances=0,
                n_training_samples=n_training,
                storage_fraction=0.0,
                n_matched=0,
                n_checked=0,
                match_fraction=0.0,
                example_matches=[],
                data_leakage_confirmed=False,
                mitigations=self._build_mitigations(is_svm, is_knn, is_dp_safe),
            )
            output = self._make_report(target)
            self._write_report(output)
            return output

        n_stored = len(stored_instances)

        # Check shape compatibility
        if stored_instances.shape[1] != X_train.shape[1]:
            logger.warning(
                "Feature dimension mismatch: stored instances have %d "
                "features, training data has %d. Cannot compare.",
                stored_instances.shape[1],
                X_train.shape[1],
            )
            self.results = InstanceBasedAttackResults(
                model_type=model_type,
                is_instance_based=True,
                is_dp_safe=is_dp_safe,
                n_stored_instances=n_stored,
                n_training_samples=n_training,
                storage_fraction=n_stored / n_training if n_training > 0 else 0.0,
                n_matched=0,
                n_checked=0,
                match_fraction=0.0,
                example_matches=[],
                data_leakage_confirmed=False,
                mitigations=self._build_mitigations(is_svm, is_knn, is_dp_safe),
                details={"error": "Feature dimension mismatch"},
            )
            output = self._make_report(target)
            self._write_report(output)
            return output

        # Compare stored instances to training data
        n_matched, example_matches = self._compare_instances(
            stored_instances, stored_indices, X_train
        )

        storage_fraction = n_stored / n_training if n_training > 0 else 0.0
        match_fraction = n_matched / n_stored if n_stored > 0 else 0.0
        data_leakage_confirmed = n_matched > 0

        mitigations = self._build_mitigations(is_svm, is_knn, is_dp_safe)

        self.results = InstanceBasedAttackResults(
            model_type=model_type,
            is_instance_based=True,
            is_dp_safe=is_dp_safe,
            n_stored_instances=n_stored,
            n_training_samples=n_training,
            storage_fraction=storage_fraction,
            n_matched=n_matched,
            n_checked=n_stored,
            match_fraction=match_fraction,
            example_matches=example_matches,
            data_leakage_confirmed=data_leakage_confirmed,
            mitigations=mitigations,
        )

        output = self._make_report(target)
        self._write_report(output)
        return output

    def _construct_metadata(self) -> None:
        """Construct the metadata dictionary for reporting."""
        super()._construct_metadata()
        if self.results:
            self.metadata["global_metrics"] = {
                "model_type": self.results.model_type,
                "is_instance_based": self.results.is_instance_based,
                "is_dp_safe": self.results.is_dp_safe,
                "n_stored_instances": self.results.n_stored_instances,
                "n_training_samples": self.results.n_training_samples,
                "storage_fraction": self.results.storage_fraction,
                "n_matched": self.results.n_matched,
                "match_fraction": self.results.match_fraction,
                "data_leakage_confirmed": self.results.data_leakage_confirmed,
            }

    def _get_attack_metrics_instances(self) -> dict:
        """Return attack metrics for the report structure."""
        attack_metrics_experiment = {}
        if self.results:
            attack_metrics_instances = {
                "instance_0": asdict(self.results),
            }
            attack_metrics_experiment["attack_instance_logger"] = (
                attack_metrics_instances
            )
        return attack_metrics_experiment

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report.

        Returns
        -------
        FPDF : A PDF object containing the instance-based attack report.
        """
        metadata = output["metadata"]
        metrics = metadata["global_metrics"]
        instance_data = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)

        report.title(pdf, "Instance-Based Model Attack Report")
        report.subtitle(pdf, "Introduction")
        report.line(pdf, _INTRODUCTION)

        report.subtitle(pdf, "Experiment Summary")
        report.line(
            pdf,
            f"{'sacroml_version':>30s}: {str(metadata['sacroml_version']):30s}",
            font="courier",
        )
        for key, value in metadata["attack_params"].items():
            report.line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")

        report.subtitle(pdf, "Risk Summary")
        for key in (
            "model_type",
            "is_instance_based",
            "is_dp_safe",
            "data_leakage_confirmed",
            "n_stored_instances",
            "n_training_samples",
            "storage_fraction",
            "n_matched",
            "match_fraction",
        ):
            value = metrics.get(key, "N/A")
            report.line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")

        # Example matches
        example_matches = instance_data.get("example_matches", [])
        if example_matches:
            pdf.add_page()
            report.title(pdf, "Example Matches")
            report.line(
                pdf,
                f"Showing {len(example_matches)} example(s) of training "
                "data found stored in the model (first 5 feature values):",
            )
            for i, match in enumerate(example_matches):
                report.line(
                    pdf,
                    f"  Match {i + 1}: "
                    f"stored[{match.get('stored_index', '?')}] "
                    f"= train[{match.get('training_index', '?')}]  "
                    f"values: {match.get('stored_values', [])}",
                    font="courier",
                    font_size=9,
                )

        # Mitigations
        mitigations = instance_data.get("mitigations", [])
        if mitigations:
            pdf.add_page()
            report.title(pdf, "Available Mitigations")
            for i, mitigation in enumerate(mitigations):
                report.subtitle(pdf, f"Option {i + 1}")
                report.line(pdf, mitigation)

        pdf.add_page()
        report.title(pdf, "Glossary")
        report._write_dict(pdf, _GLOSSARY)

        return pdf
