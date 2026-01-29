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
   (not defined for all types of model)
The methodology is aligned with SACRO-ML's privacy risk framework.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from acro import ACRO
from fpdf import FPDF
from scipy.stats import ks_2samp
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


ALPHA = 0.05  # Pval cut-off for statistical significance tests

# --- Data Structure for Attack Results ---


@dataclass
class StructuralRecordLevelResults:
    """Dataclass to store record-level outcomes for structural attack."""

    k_anonymity: list[int]
    class_disclosure: list[bool]
    smallgroup_risk: list[bool]


@dataclass
class StructuralAttackResults:
    """
    Dataclass to store the results of a structural attack.

    Attributes
    ----------
    unnecessary_risk (bool) : Risk due to unnecessarily complex model structure.
    train_acc (float) : target accuracy on test  set
    test_acc (float) : target accuracy on training  set
    generalisation_error (float) : The target's generalisation error
    gen_error_risk (bool) : Risk that train/ test loss distns significantly differ
    dof_risk (bool) : Risk based on degrees of freedom.
    k_anonymity_risk (bool) : Risk based on k-anonymity violations.
    class_disclosure_risk (bool) : Risk of class label disclosure.
    lowvals_cd_risk (bool) : Risk from low-frequency class values.
    details (dict | None) : Optional additional metadata.
    """

    test_acc: float
    train_acc: float
    generalisation_error: float
    gen_error_risk: bool
    unnecessary_risk: bool
    dof_risk: bool
    k_anonymity_risk: bool
    class_disclosure_risk: bool
    smallgroup_risk: bool
    details: dict | None = None


"""
Optional additional metadata, such as model-specific notes or thresholds used.
"""

# --- Standalone Helper Functions for Risk Assessment ---


def get_unnecessary_risk(model: BaseEstimator | torch.nn.Module) -> bool:
    """Check whether model hyperparameters are in the top 20% most risky.

     This check is based on a classifier trained on results from a large
     scale study described in: https://doi.org/10.48550/arXiv.2502.09396

    Parameters
    ----------
    model : BaseEstimator|torch.nn.Module
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
    logger.info("Unnecessary risk not define for models of type %s", type(model))
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


def get_model_param_count(model: BaseEstimator | torch.nn.Module) -> int:
    """Return the number of trained parameters in a model.

    This includes learned weights, thresholds, and decision rules depending on
    model type. Supports DecisionTree, RandomForest, AdaBoost, XGBoost,  MLP and torch
    classifiers.

    Parameters
    ----------
    model (BaseEstimator|torch.nn.Module) : A trained classification model.

    Returns
    -------
    int : Estimated number of learned parameters.
    """
    if torch is not None and isinstance(model, torch.nn.Module):
        return _get_model_param_count_torch(model)
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


def _get_model_param_count_torch(model: torch.nn.Module) -> int:
    """Return number of trainable parameters in a pytorch model.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    int count of trainable params
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    and hyperparameters, aligned with TRE risk appetite for 'traditional' outputs.

    Attack pipeline includes checks for:
    - Residual Degrees of freedom
    - Complexity risk
    - and uses Equivalence class analysis to identify risks of:
       - K-anonymity
       - Class disclosure:
       (partitions of decision space with zero probability for some labels)
       - Reidentification through small groups
       (partitions of decision space with some groups below the cell count threshold)
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        risk_appetite_config: str = "default",
        report_individual: bool = False,
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
        report_individual : bool
            Whether to report metrics for each individual record.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.target: Target | None = None
        self.results: StructuralAttackResults | None = None
        self.report_individual = report_individual

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
        if not target.has_model():
            logger.info("target.model.model is missing, cannot proceed")
            return False
        logger.info("Class of module is %s ", type(target.model.model))
        if (
            target.has_model()
            and isinstance(target.model.model, BaseEstimator | torch.nn.Module)
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
        - Unnecessary complexity risk
        - Equivalence class analysis leading to checks for risk of
        -- K-anonymity below threshold
        -- Class disclosure :
           presence of partitions with zero probability for one or more labels
        - smallgroup_risk :
           presence of partitions where count of records with some labels
           is below Cell Count Threshold

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
        eqclass_probas, eqclass_inv_indices, eqclass_counts = (
            self._calculate_equivalence_classes()
        )
        # check shapes are sane
        num_eqclasses, num_outputs = eqclass_probas.shape
        assert len(eqclass_counts) == num_eqclasses
        num_samples = target.y_train.shape[0]
        assert len(eqclass_inv_indices) == num_samples

        # Run different risk assessments, some just return  global value

        test_acc = self.target.model.score(self.target.X_test, self.target.y_test)
        train_acc = self.target.model.score(self.target.X_train, self.target.y_train)

        generalisation_error = self.target.model.get_generalisation_error(
            self.target.X_train,
            self.target.y_train,
            self.target.X_test,
            self.target.y_test,
        )
        gen_error_risk = self._assess_generalisation_error_risk()
        dof_risk = self._assess_dof_risk()

        unnecessary_risk = get_unnecessary_risk(model)

        # Run assessments that return global value and one for each training record
        global_krisk, record_level_kval = self._assess_k_anonymity_risk(
            eqclass_inv_indices, eqclass_counts
        )

        global_cd, record_level_cd = self._assess_class_disclosure_risk(
            eqclass_probas, eqclass_inv_indices
        )

        global_small, record_level_small = self._assess_smallgroup_risk(
            eqclass_probas, eqclass_inv_indices, eqclass_counts
        )

        # make storage for results
        self.results = StructuralAttackResults(
            test_acc=test_acc,
            train_acc=train_acc,
            generalisation_error=generalisation_error,
            gen_error_risk=gen_error_risk,
            dof_risk=dof_risk,
            unnecessary_risk=unnecessary_risk,
            k_anonymity_risk=global_krisk,
            class_disclosure_risk=global_cd,
            smallgroup_risk=global_small,
        )
        self.record_level_results = StructuralRecordLevelResults(
            k_anonymity=record_level_kval,
            class_disclosure=record_level_cd,
            smallgroup_risk=record_level_small,
        )

        output = self._make_report(target)

        # If requested, write the JSON report file.
        # The PDF is generated by the main runner script from all JSON files.
        if self.write_report:
            self._write_report(output)

        return output

    def _assess_generalisation_error_risk(self) -> bool:
        """Assess probability that generalisation error is statistically significant.

        Uses Kolmogorov-Smirnov 2 samples test
        to compare distributions of train/test losses

        Returns
        -------
        bool : True if KS test returns P<0.05
        """
        train_losses = self.target.model.get_losses(
            self.target.X_train, self.target.y_train
        )
        test_losses = self.target.model.get_losses(
            self.target.X_test, self.target.y_test
        )
        ks_2samp_results = ks_2samp(train_losses, test_losses)
        pval = ks_2samp_results.pvalue
        return pval < ALPHA

    def _assess_dof_risk(self) -> bool:
        """Assess risk based on Residual Degrees of Freedom.

        Returns
        -------
        bool : True if the model's residual degrees of freedom are below the
               safe threshold.
        """
        n_features = self.target.X_train.shape[1]
        n_samples = self.target.X_train.shape[0]
        model = self.target.model.model
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
        return bool(residual_dof < self.DOF_THRESHOLD)

    def _assess_k_anonymity_risk(
        self, eqclass_inv_indices: np.array, eqclass_counts: np.array
    ) -> tuple(bool, list):
        """Assess k-anonymity risk from equivalence class sizes.

        Returns
        -------
        bool : True if the smallest equivalence class size is below the safe threshold.
        list : size of class each record belongs to
        """
        min_k = np.min(eqclass_counts)
        logger.info("Smallest equivalence class size (k-anonymity) is %d", min_k)
        global_risk = bool(np.any(eqclass_counts < self.THRESHOLD))

        record_level: list = [int(eqclass_counts[i]) for i in eqclass_inv_indices]

        return global_risk, record_level

    def _assess_class_disclosure_risk(
        self, eqclass_probas: np.ndarray, eqclass_inv_indices: np.array
    ) -> tuple[bool, list]:
        """Assess risk of class disclosing class frequencies.

        i.e. reporting that for some groups
        there is zero probability of observing one or more labels.

        Returns
        -------
               tuple[bool, list]:
                - overall : True if any equivalence class has any near-zero values
                             in its predicted probability for any label
                - recordlevel: True if the equivalencce class a record belongs to
                               has near-zero predicted probab. for one or more labels
        """
        # list of bools-one for each equivalence class
        eqclass_cdrisks = np.any(np.isclose(eqclass_probas, 0.0), axis=1)
        overall = bool(np.any(eqclass_cdrisks))
        record_level = [bool(eqclass_cdrisks[i]) for i in eqclass_inv_indices]

        return overall, record_level

    def _assess_smallgroup_risk(
        self,
        eqclass_probas: np.ndarray,
        eqclass_inv_indices: np.array,
        eqclass_counts: np.array,
    ) -> tuple[bool, list]:
        """Assess risk of reporting on a group with only a few members for a label.

        Returns
        -------
               tuple[bool, list]:
                - Overall smallgroup_risk:
                  True if for any equivalence class for any label,
                  0 < estimated number of examples <self.THRESHOLD
                - Record level.
                  True if relation holds within the record's equivalence class
        """
        # small groups risk:
        #   report on a group of smaller than self.THRESHOLD training records
        #   freqs is an estimate:
        #    number of records in a class multtiplied by output probabilities
        freqs = eqclass_probas * eqclass_counts[:, np.newaxis]

        eqclass_smallgrouprisk = np.any((freqs > 0) & (freqs < self.THRESHOLD), axis=1)
        overall = bool(np.any(eqclass_smallgrouprisk))
        record_level = [bool(eqclass_smallgrouprisk[i]) for i in eqclass_inv_indices]

        return overall, record_level

    def _calculate_equivalence_classes(self) -> tuple:
        """Calculate equivalence classes based on model type and predictions.

        For decision trees there is one equivalence class per leaf
        For all other models an equivalence class is all the training records
        for which the model predicts the same output probabilities.

        Returns
        -------
        eq_class_probas (numpy.ndarray):
            array of output probabilities (columns)
            for all the distinct equivalence classes (rows)
        eqclass_inv_indices (np.array);
            holds indices of equivalence class for each training record
        eqclass_counts (np.array):
            holds count of members in eacgh equivalence class
        """
        model = self.target.model.model
        if isinstance(model, DecisionTreeClassifier):
            return self._dt_get_equivalence_classes()
        return self._get_equivalence_classes_from_probas()

    def _dt_get_equivalence_classes(self) -> tuple:
        """Get equivalence classes for a Decision Tree via leaf nodes."""
        model = self.target.model.model
        # find out which leaves records end up in
        destinations = model.apply(self.target.X_train)
        leaves, indices, inv_indices, counts = np.unique(
            destinations, return_index=True, return_inverse=True, return_counts=True
        )

        # get prediction probabilities for each leaf
        # this means equiv_classes may not be unique in this case (e.g. XOR problem)
        equiv_classes = self.target.model.predict_proba(self.target.X_train[indices])
        return equiv_classes, inv_indices, counts

    def _get_equivalence_classes_from_probas(self) -> tuple:
        """Get equivalence classes based on predicted probabilities."""
        y_probs = self.target.model.predict_proba(self.target.X_train)
        return np.unique(y_probs, axis=0, return_inverse=True, return_counts=True)

    def _construct_metadata(self):
        """Construct the metadata dictionary for reporting.

        Used internally to populate metadata for the attack report, including
        thresholds and results.
        """
        super()._construct_metadata()
        attack_specific_output = {
            "attack_name": str(self),
            "risk_appetite_config": self.risk_appetite_config,
            "safe_threshold": self.THRESHOLD,
            "safe_dof_threshold": self.DOF_THRESHOLD,
        }
        self.metadata["attack_params"].update(attack_specific_output)
        if self.results:
            self.metadata["global_metrics"] = asdict(self.results)

        # Save global and record-level results in the attack metrics
        self.attack_metrics = {}
        for key, val in asdict(self.results).items():
            self.attack_metrics[key] = val
        self.attack_metrics["individual"] = asdict(self.record_level_results)

    def _get_attack_metrics_instances(self) -> dict:
        """Return attack metrics. Required by the Attack base class.

        Used internally to expose metrics from the `StructuralAttackResults` dataclass.
        """
        # This method is required by the abstract base class.
        # Its functionality is now handled by the `results` dataclass
        # and the `_construct_metadata` method.
        # We return the metrics from the results object if available.
        attack_metrics_experiment = {}
        attack_metrics_instances = {}
        if self.results:
            attack_metrics_instances["instance_0"] = asdict(self.results)
            if self.report_individual and self.record_level_results:
                individuals = {"individual": asdict(self.record_level_results)}
                attack_metrics_instances["instance_0"].update(individuals)
            attack_metrics_experiment["attack_instance_logger"] = (
                attack_metrics_instances
            )
        return attack_metrics_experiment

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report using the external report module.

        Returns
        -------
        FPDF : A PDF object containing the formatted structural attack report.
        """
        return report.create_structural_report(output)
