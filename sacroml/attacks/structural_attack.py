"""Structural attacks.

Runs a number of 'static' structural attacks based on:
(i) the target model's properties;
(ii) the TRE's risk appetite as applied to tables and standard regressions.

This module provides the `StructuralAttack` class, which assesses a trained
machine learning model for several common structural vulnerabilities.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from acro import ACRO
from fpdf import FPDF
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

from sacroml.attacks import report
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# --- Data Structure for Attack Results ---


@dataclass
class StructuralAttackResults:
    """Dataclass to store the results of a structural attack."""

    dof_risk: bool
    k_anonymity_risk: bool
    class_disclosure_risk: bool
    lowvals_cd_risk: bool
    unnecessary_risk: bool
    details: dict | None = None


# --- Standalone Helper Functions for Risk Assessment ---


def get_unnecessary_risk(model: BaseEstimator) -> bool:
    """Check whether model hyperparameters are in the top 20% most risky.

    This check is designed to assess whether a model is likely to be
    **unnecessarily** risky, i.e., whether it is highly likely that a different
    combination of hyper-parameters would have led to model with similar or
    better accuracy on the task but with lower membership inference risk.

    The rules were derived from an experimental study and are specific to
    certain model types.

    Parameters
    ----------
    model : BaseEstimator
        The trained model to check for risk.

    Returns
    -------
    bool
        True if the model's hyperparameters are considered high risk, otherwise False.
    """
    if isinstance(model, DecisionTreeClassifier):
        return _get_unnecessary_risk_dt(model)
    if isinstance(model, RandomForestClassifier):
        return _get_unnecessary_risk_rf(model)
    if isinstance(model, XGBClassifier):
        return _get_unnecessary_risk_xgb(model)
    return False


def _get_unnecessary_risk_dt(model: DecisionTreeClassifier) -> bool:
    """Return whether DecisionTreeClassifier parameters are high risk."""
    max_depth = float(model.max_depth) if model.max_depth else 500
    max_features = model.max_features
    min_samples_leaf = model.min_samples_leaf
    min_samples_split = model.min_samples_split
    splitter = model.splitter
    return (
        (max_depth > 7.5 and min_samples_leaf <= 7.5 and min_samples_split <= 15)
        or (
            splitter == "best"
            and max_depth > 7.5
            and min_samples_leaf <= 7.5
            and min_samples_split > 15
        )
        or (
            splitter == "best"
            and max_depth > 7.5
            and 7.5 < min_samples_leaf <= 15
            and max_features is None
        )
        or (
            splitter == "best"
            and 3.5 < max_depth <= 7.5
            and max_features is None
            and min_samples_leaf <= 7.5
        )
        or (
            splitter == "random"
            and max_depth > 7.5
            and min_samples_leaf <= 7.5
            and max_features is None
        )
    )


def _get_unnecessary_risk_rf(model: RandomForestClassifier) -> bool:
    """Return whether RandomForestClassifier parameters are high risk."""
    max_depth = float(model.max_depth) if model.max_depth else 500
    n_estimators = model.n_estimators
    max_features = model.max_features
    min_samples_leaf = model.min_samples_leaf
    min_samples_split = model.min_samples_split
    return (
        (max_depth > 3.5 and n_estimators > 35 and max_features is not None)
        or (
            max_depth > 3.5
            and n_estimators > 35
            and min_samples_split <= 15
            and max_features is None
            and model.bootstrap
        )
        or (
            max_depth > 7.5
            and 15 < n_estimators <= 35
            and min_samples_leaf <= 15
            and not model.bootstrap
        )
    )


def _get_unnecessary_risk_xgb(model: XGBClassifier) -> bool:
    """Return whether XGBClassifier parameters are high risk."""
    n_estimators = int(model.n_estimators) if model.n_estimators else 100
    max_depth = float(model.max_depth) if model.max_depth else 6
    min_child_weight = float(model.min_child_weight) if model.min_child_weight else 1.0
    return (
        (max_depth > 3.5 and 3.5 < n_estimators <= 12.5 and min_child_weight <= 1.5)
        or (max_depth > 3.5 and n_estimators > 12.5 and min_child_weight <= 3)
        or (max_depth > 3.5 and n_estimators > 62.5 and 3 < min_child_weight <= 6)
    )


# --- Standalone Helper Functions for Parameter Counting ---


def get_model_param_count(model: BaseEstimator) -> int:
    """Return the number of trained parameters in a model."""
    if isinstance(model, DecisionTreeClassifier):
        return _get_model_param_count_dt(model)
    if isinstance(model, RandomForestClassifier):
        return _get_model_param_count_rf(model)
    if isinstance(model, AdaBoostClassifier):
        return _get_model_param_count_ada(model)
    if isinstance(model, XGBClassifier):
        return _get_model_param_count_xgb(model)
    if isinstance(model, MLPClassifier):
        return _get_model_param_count_mlp(model)
    logger.warning(
        "Parameter counting not implemented for model type %s", type(model).__name__
    )
    return 0


def _get_tree_parameter_count(dtree: DecisionTreeClassifier) -> int:
    """Read the tree structure and return the number of learned parameters."""
    n_nodes = dtree.tree_.node_count
    is_leaf = dtree.tree_.children_left == dtree.tree_.children_right
    n_leaves = np.sum(is_leaf)
    n_internal_nodes = n_nodes - n_leaves
    # 2 params (feature, threshold) per internal node
    # (n_classes - 1) params per leaf node for the probability distribution
    return 2 * n_internal_nodes + n_leaves * (dtree.n_classes_ - 1)


def _get_model_param_count_dt(model: DecisionTreeClassifier) -> int:
    """Return the number of trained DecisionTreeClassifier parameters."""
    return _get_tree_parameter_count(model)


def _get_model_param_count_rf(model: RandomForestClassifier) -> int:
    """Return the number of trained RandomForestClassifier parameters."""
    return sum(_get_tree_parameter_count(member) for member in model.estimators_)


def _get_model_param_count_ada(model: AdaBoostClassifier) -> int:
    """Return the number of trained AdaBoostClassifier parameters."""
    try:  # sklearn v1.2+
        base = model.estimator
    except AttributeError:  # pragma: no cover (sklearn version <1.2)
        base = model.base_estimator

    if isinstance(base, DecisionTreeClassifier):
        return sum(_get_tree_parameter_count(member) for member in model.estimators_)
    return 0


def _get_model_param_count_xgb(model: XGBClassifier) -> int:
    """Return the number of trained XGBClassifier parameters."""
    df = model.get_booster().trees_to_dataframe()
    if df.empty:
        return 0
    n_trees = df["Tree"].max() + 1
    n_leaves = len(df[df.Feature == "Leaf"])
    n_internal_nodes = len(df) - n_leaves
    # 2 params per internal node, (n_classes-1) per leaf, one weight per tree
    return 2 * n_internal_nodes + (model.n_classes_ - 1) * n_leaves + n_trees


def _get_model_param_count_mlp(model: MLPClassifier) -> int:
    """Return the number of trained MLPClassifier parameters."""
    weights = model.coefs_
    biases = model.intercepts_
    return sum(w.size for w in weights) + sum(b.size for b in biases)


# --- Main Attack Class ---


class StructuralAttack(Attack):
    """Structural attacks based on the static structure of a model."""

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        risk_appetite_config: str = "default",
    ) -> None:
        """Construct an object to execute a structural attack.

        Parameters
        ----------
        output_dir : str
            Name of a directory to write outputs.
        write_report : bool
            Whether to generate a JSON and PDF report.
        risk_appetite_config : str
            Path to yaml file specifying TRE risk appetite.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.target: Target | None = None
        self.results: StructuralAttackResults | None = None

        # Load risk appetite from ACRO config
        myacro = ACRO(risk_appetite_config)
        self.risk_appetite_config = risk_appetite_config
        self.THRESHOLD = myacro.config["safe_threshold"]
        self.DOF_THRESHOLD = myacro.config["safe_dof_threshold"]
        logger.info(
            "Thresholds for count %i and DoF %i", self.THRESHOLD, self.DOF_THRESHOLD
        )

    def __str__(self) -> str:
        """Return the name of the attack."""
        return "Structural Attack"

    @classmethod
    def attackable(cls, target: Target) -> bool:
        """Return whether a target can be assessed with StructuralAttack."""
        if (
            target.has_model()
            and isinstance(target.model.model, BaseEstimator)
            and target.has_data()
        ):
            return True
        logger.info("WARNING: StructuralAttack requires a loadable model and data.")
        return False

    def _attack(self, target: Target) -> dict:
        """Run all structural risk assessments and returns a report dictionary.

        This is the main orchestration method, called by the base class `run` method.
        It calls helper methods to perform individual risk checks and collates
        the results into a dictionary for reporting.

        Parameters
        ----------
        target : Target
            The target object containing the model and data.

        Returns
        -------
        dict
            A dictionary containing the results and metadata of the attack.
        """
        self.target = target
        model = target.model.model

        # Calculate equivalence classes, which are needed for several checks
        equiv_classes, equiv_counts, _ = self._calculate_equivalence_classes(model)

        # Run individual risk assessments
        dof_risk = self._assess_dof_risk(model)
        k_anonymity_risk = self._assess_k_anonymity_risk(equiv_counts)
        unnecessary_risk = get_unnecessary_risk(model)
        class_disclosure_risk, lowvals_cd_risk = self._assess_class_disclosure_risk(
            equiv_classes, equiv_counts
        )

        # Collate results into the structured dataclass
        self.results = StructuralAttackResults(
            dof_risk=dof_risk,
            k_anonymity_risk=k_anonymity_risk,
            unnecessary_risk=unnecessary_risk,
            class_disclosure_risk=class_disclosure_risk,
            lowvals_cd_risk=lowvals_cd_risk,
        )

        # Let the base class generate the report dictionary.
        # It will internally call our overridden _construct_metadata method.
        output = self._make_report(target)

        # If requested, write the JSON report file.
        # The PDF is generated by the main runner script from all JSON files.
        if self.write_report:
            self._write_report(output)

        return output

    def _assess_dof_risk(self, model: BaseEstimator) -> bool:
        """Assess risk based on Residual Degrees of Freedom."""
        n_features = self.target.X_train.shape[1]
        n_samples = self.target.X_train.shape[0]
        n_params = get_model_param_count(model)

        if n_params < n_features:
            logger.info(
                "Model has fewer parameters (%d) than features (%d).",
                n_params,
                n_features,
            )

        residual_dof = n_samples - n_params
        logger.info(
            "Samples=%d, Parameters=%d, DoF=%d", n_samples, n_params, residual_dof
        )
        return residual_dof < self.DOF_THRESHOLD

    def _assess_k_anonymity_risk(self, equiv_counts: np.ndarray) -> bool:
        """Assess k-anonymity risk from equivalence class sizes."""
        if equiv_counts.size == 0:
            return False
        min_k = np.min(equiv_counts)
        logger.info("Smallest equivalence class size (k-anonymity) is %d", min_k)
        return min_k < self.THRESHOLD

    def _assess_class_disclosure_risk(
        self, equiv_classes: np.ndarray, equiv_counts: np.ndarray
    ) -> tuple[bool, bool]:
        """Assess risk of disclosing class frequencies."""
        if equiv_classes.size == 0 or equiv_counts.size == 0:
            return False, False

        freqs = equiv_classes * equiv_counts[:, np.newaxis]
        class_disclosure_risk = np.any((freqs > 0) & (freqs < self.THRESHOLD))
        lowvals_cd_risk = np.any((freqs > 0) & (freqs < self.THRESHOLD))

        return class_disclosure_risk, lowvals_cd_risk

    def _calculate_equivalence_classes(self, model: BaseEstimator) -> tuple:
        """Calculate equivalence classes based on model type and predictions."""
        if isinstance(model, DecisionTreeClassifier):
            return self._dt_get_equivalence_classes(model)
        return self._get_equivalence_classes_from_probas(model)

    def _dt_get_equivalence_classes(self, model: DecisionTreeClassifier) -> tuple:
        """Get equivalence classes for a Decision Tree via leaf nodes."""
        destinations = model.apply(self.target.X_train)
        leaves, counts = np.unique(destinations, return_counts=True)
        members = [np.where(destinations == leaf)[0] for leaf in leaves]
        sample_indices = [mem[0] for mem in members if len(mem) > 0]
        equiv_classes = model.predict_proba(self.target.X_train[sample_indices])
        return equiv_classes, counts, members

    def _get_equivalence_classes_from_probas(self, model: BaseEstimator) -> tuple:
        """Get equivalence classes based on predicted probabilities."""
        y_probs = model.predict_proba(self.target.X_train)
        equiv_classes, inverse_indices, equiv_counts = np.unique(
            y_probs, axis=0, return_inverse=True, return_counts=True
        )
        members = [np.where(inverse_indices == i)[0] for i in range(len(equiv_classes))]
        return equiv_classes, equiv_counts, members

    def _construct_metadata(self):
        """Construct the metadata dictionary for reporting."""
        super()._construct_metadata()
        self.metadata["attack_specific_output"] = {
            "attack_name": str(self),
            "risk_appetite_config": self.risk_appetite_config,
            "safe_threshold": self.THRESHOLD,
            "safe_dof_threshold": self.DOF_THRESHOLD,
        }
        if self.results:
            self.metadata["global_metrics"] = asdict(self.results)

    def _get_attack_metrics_instances(self) -> dict:
        """Return attack metrics. Required by the Attack base class."""
        # This method is required by the abstract base class.
        # Its functionality is now handled by the `results` dataclass
        # and the `_construct_metadata` method.
        # We return the metrics from the results object if available.
        if self.results:
            return asdict(self.results)
        return {}

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report using the external report module."""
        return report.create_structural_report(output)
