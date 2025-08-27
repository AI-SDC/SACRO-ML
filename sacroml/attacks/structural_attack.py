"""Structural attacks.

Runs a number of 'static' structural attacks based on:
(i) the target model's properties;
(ii) the TRE's risk appetite as applied to tables and standard regressions.

This module provides the `StructuralAttack` class, which assesses a trained
machine learning model for several common structural vulnerabilities.

These include:
- Degrees of freedom risk
- k-anonymity violations
- Class disclosure
- 'Unnecessary Risk' caused by hyper-parameters likely to lead to undue model complexity

The methodology is aligned with SACRO-ML's privacy risk framework.
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

try:
    import torch
except ImportError:
    torch = None

from sacroml.attacks import report
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Data Structure for Attack Results ---


@dataclass
class StructuralRecordLevelResults:
    """Dataclass to store record-level outcomes for structural attack."""

    k_anonymity: list[int]
    class_disclosure: list[bool]
    dof: list[int]


@dataclass
class StructuralAttackResults:
    """
    Dataclass to store the results of a structural attack.

    Attributes
    ----------
    dof_risk (bool) : Risk based on degrees of freedom.
    k_anonymity_risk (bool) : Risk based on k-anonymity violations.
    class_disclosure_risk (bool) : Risk of class label disclosure.
    lowvals_cd_risk (bool) : Risk from low-frequency class values.
    unnecessary_risk (bool) : Risk due to unnecessarily complex model structure.
    details (dict | None) : Optional additional metadata.
    """

    dof_risk: bool
    k_anonymity_risk: bool
    class_disclosure_risk: bool
    lowvals_cd_risk: bool
    unnecessary_risk: bool
    details: dict | None = None


"""
Optional additional metadata, such as model-specific notes or thresholds used.
"""

# --- Standalone Helper Functions for Risk Assessment ---


def get_unnecessary_risk(model: BaseEstimator) -> bool:
    """Check whether model hyperparameters are in the top 20% most risky.

     This check is based on a classifier trained on results from a large
     scale study described in: https://doi.org/10.48550/arXiv.2502.09396

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
    """Return whether DecisionTreeClassifier parameters are high risk.

    This function applies decision rules extracted from a trained decision tree
    classifier on hyperparameter configurations ranked by MIA AUC.
    """
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
    """Return whether RandomForestClassifier parameters are high risk.

    This function applies decision rules extracted from a trained decision tree
    classifier on hyperparameter configurations ranked by MIA AUC.
    """
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
    """Return whether XGBClassifier parameters are high risk.

    This function applies decision rules extracted from a trained decision tree
    classifier on hyperparameter configurations ranked by MIA AUC.

    If parameters have not been specified it takes the xgboost defaults
    from https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
    and here: https://xgboost.readthedocs.io/en/stable/parameter.html
    """
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
    """Return the number of trained parameters in a model.

    This includes learned weights, thresholds, and decision rules depending on
    model type. Supports DecisionTree, RandomForest, AdaBoost, XGBoost, and MLP
    classifiers.

    Parameters
    ----------
    model (BaseEstimator) : A trained scikit-learn or XGBoost model.

    Returns
    -------
    int : Estimated number of learned parameters.
    """
    if torch is not None and isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    """Structural attacks based on the static structure of a model.

    Performs structural privacy risk assessments on trained ML models.

    This class implements static structural attacks based on model architecture
    and hyperparameters, aligned with TRE risk appetite configurations.

    Attack pipeline includes:
    - Equivalence class analysis
    - Degrees of freedom check
    - k-anonymity check
    - Class disclosure risk
    - Complexity risk
    """

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
        This method orchestrates the full structural attack pipeline, including:
        - Degrees of freedom risk
        - k-anonymity risk
        - Class disclosure risk
        - Unnecessary complexity risk

        Parameters
        ----------
        target : Target
            The target object containing the model and data.

        Returns
        -------
        dict
           Attack report. A dictionary containing the results and metadata
           of the attack.

         Note:
         This method is invoked by the base class `run()` method.
         It assumes the target model has been trained and validated
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

        # Initialize record-level results
        self.record_level_results = {
            "k_anonymity": [],
            "class_disclosure": [],
            "dof": [],
        }

        # Populate record-level k-anonymity and class disclosure
        for count, class_probs in zip(equiv_counts, equiv_classes):
            self.record_level_results["k_anonymity"].append(count)
            self.record_level_results["class_disclosure"].append(
                bool(np.any((class_probs > 0) & (class_probs < self.THRESHOLD)))
            )

        # Populate record-level degrees of freedom
        n_samples = self.target.X_train.shape[0]
        n_params = get_model_param_count(model)
        residual_dof = n_samples - n_params
        self.record_level_results["dof"] = [residual_dof] * n_samples

        output = self._make_report(target)

        # If requested, write the JSON report file.
        # The PDF is generated by the main runner script from all JSON files.
        if self.write_report:
            self._write_report(output)

        return output

    def _assess_dof_risk(self, model: BaseEstimator) -> bool:
        """Assess risk based on Residual Degrees of Freedom.

        Returns
        -------
        bool : True if the model's residual degrees of freedom are below the
               safe threshold.
        """
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
        """Assess k-anonymity risk from equivalence class sizes.

        Returns
        -------
        bool : True if the smallest equivalence class size is below the safe threshold.
        """
        min_k = np.min(equiv_counts)
        logger.info("Smallest equivalence class size (k-anonymity) is %d", min_k)
        return min_k < self.THRESHOLD

    def _assess_class_disclosure_risk(
        self, equiv_classes: np.ndarray, equiv_counts: np.ndarray
    ) -> tuple[bool, bool]:
        """Assess risk of disclosing class frequencies.

        Returns
        -------
               tuple[bool, bool]:
                                 - class_disclosure_risk: True if any class
                                   frequency is below the threshold.
                                 - lowvals_cd_risk: True if low-frequency values
                                   pose a disclosure risk.
        """
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
        """Construct the metadata dictionary for reporting.

        Used internally to populate metadata for the attack report, including
        thresholds and results.
        """
        super()._construct_metadata()
        self.metadata["attack_specific_output"] = {
            "attack_name": str(self),
            "risk_appetite_config": self.risk_appetite_config,
            "safe_threshold": self.THRESHOLD,
            "safe_dof_threshold": self.DOF_THRESHOLD,
        }
        if self.results:
            self.metadata["global_metrics"] = asdict(self.results)

        # Save record-level results in the attack metrics
        self.attack_metrics = [{}]
        self.attack_metrics[-1]["individual"] = self.record_level_results

    def _get_attack_metrics_instances(self) -> dict:
        """Return attack metrics. Required by the Attack base class.

        Used internally to expose metrics from the `StructuralAttackResults` dataclass.
        """
        # This method is required by the abstract base class.
        # Its functionality is now handled by the `results` dataclass
        # and the `_construct_metadata` method.
        # We return the metrics from the results object if available.
        if self.results:
            return asdict(self.results)
        return {}

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report using the external report module.

        Returns
        -------
        FPDF : A PDF object containing the formatted structural attack report.
        """
        return report.create_structural_report(output)
