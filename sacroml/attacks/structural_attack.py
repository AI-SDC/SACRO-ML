"""Structural attacks.

Runs a number of 'static' structural attacks based on:
(i) the target model's properties;
(ii) the TREs risk appetite as applied to tables and standard regressions.

Tree-based model types currently supported.
"""

from __future__ import annotations

import logging

import numpy as np
from acro import ACRO
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pylint: disable=chained-comparison


def get_unnecessary_risk(model: BaseEstimator) -> bool:
    """Check whether model hyperparameters are in the top 20% most risky.

    This check is designed to assess whether a model is likely to be
    **unnecessarily** risky, i.e., whether it is highly likely that a different
    combination of hyper-parameters would have led to model with similar or
    better accuracy on the task but with lower membership inference risk.

    The rules applied from an experimental study using a grid search in which:
    - max_features was one-hot encoded from the set [None, log2, sqrt]
    - splitter was encoded using 0=best, 1=random

    The target models created were then subject to membership inference attacks
    (MIA) and the hyper-param combinations rank-ordered according to MIA AUC.
    Then a decision tree trained to recognise whether hyper-params combintions
    were in the 20% most risky.  The rules below were extracted from that tree
    for the 'least risky' nodes.

    Parameters
    ----------
    model : BaseEstimator
        Model to check for risk.

    Returns
    -------
    bool
        True if high risk, otherwise False.
    """
    unnecessary_risk: bool = False
    if isinstance(model, DecisionTreeClassifier):
        unnecessary_risk = _get_unnecessary_risk_dt(model)
    elif isinstance(model, RandomForestClassifier):
        unnecessary_risk = _get_unnecessary_risk_rf(model)
    elif isinstance(model, XGBClassifier):
        unnecessary_risk = _get_unnecessary_risk_xgb(model)
    return unnecessary_risk


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
    """Return whether XGBClassifier parameters are high risk.

    Check whether params exist and using xgboost defaults if not using defaults
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


def get_tree_parameter_count(dtree: DecisionTreeClassifier) -> int:
    """Read the tree structure and return the number of learned parameters."""
    n_nodes = dtree.tree_.node_count
    left = dtree.tree_.children_left
    right = dtree.tree_.children_right
    is_leaf = np.zeros(n_nodes, dtype=int)
    for node_id in range(n_nodes):
        if left[node_id] == right[node_id]:
            is_leaf[node_id] = 1
    n_leaves = is_leaf.sum()

    # degrees of freedom
    n_internal_nodes = n_nodes - n_leaves
    n_params = 2 * n_internal_nodes  # feature id and threshold
    n_params += n_leaves * (dtree.n_classes_ - 1)  # probability distribution
    return n_params


def get_model_param_count(model: BaseEstimator) -> int:
    """Return the number of trained parameters in a model."""
    n_params: int = 0
    if isinstance(model, DecisionTreeClassifier):
        n_params = _get_model_param_count_dt(model)
    elif isinstance(model, RandomForestClassifier):
        n_params = _get_model_param_count_rf(model)
    elif isinstance(model, AdaBoostClassifier):
        n_params = _get_model_param_count_ada(model)
    elif isinstance(model, XGBClassifier):
        n_params = _get_model_param_count_xgb(model)
    elif isinstance(model, MLPClassifier):
        n_params = _get_model_param_count_mlp(model)
    return n_params


def _get_model_param_count_dt(model: DecisionTreeClassifier) -> int:
    """Return the number of trained DecisionTreeClassifier parameters."""
    return get_tree_parameter_count(model)


def _get_model_param_count_rf(model: RandomForestClassifier) -> int:
    """Return the number of trained RandomForestClassifier parameters."""
    n_params: int = 0
    for member in model.estimators_:
        n_params += get_tree_parameter_count(member)
    return n_params


def _get_model_param_count_ada(model: AdaBoostClassifier) -> int:
    """Return the number of trained AdaBoostClassifier parameters."""
    n_params: int = 0
    try:  # sklearn v1.2+
        base = model.estimator
    except AttributeError:  # sklearn version <1.2
        base = model.base_estimator
    if isinstance(base, DecisionTreeClassifier):
        for member in model.estimators_:
            n_params += get_tree_parameter_count(member)
    return n_params


def _get_model_param_count_xgb(model: XGBClassifier) -> int:
    """Return the number of trained XGBClassifier parameters."""
    df = model.get_booster().trees_to_dataframe()
    n_trees = df["Tree"].max()
    total = len(df)
    n_leaves = len(df[df.Feature == "Leaf"])
    # 2 per internal node, one per clas in leaves, one weight per tree
    return 2 * (total - n_leaves) + (model.n_classes_ - 1) * n_leaves + n_trees


def _get_model_param_count_mlp(model: MLPClassifier) -> int:
    """Return the number of trained MLPClassifier parameters."""
    weights = model.coefs_  # dtype is list of numpy.ndarrays
    biasses = model.intercepts_  # dtype is list of numpy.ndarrays
    return sum(a.size for a in weights) + sum(a.size for a in biasses)


class StructuralAttack(Attack):
    """Structural attacks based on the static structure of a model."""

    # pylint: disable=too-many-instance-attributes

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
        self.target: Target = None
        # disclosure risk
        self.k_anonymity_risk: bool = False
        self.dof_risk: bool = False
        self.unnecessary_risk: bool = False
        self.class_disclosure_risk: bool = False
        self.lowvals_cd_risk: bool = False
        # make dummy acro object and use it to extract risk appetite
        myacro = ACRO(risk_appetite_config)
        self.risk_appetite_config = risk_appetite_config
        self.THRESHOLD = myacro.config["safe_threshold"]
        self.DOF_THRESHOLD = myacro.config["safe_dof_threshold"]
        logger.info(
            "Thresholds for count %i and Dof %i", self.THRESHOLD, self.DOF_THRESHOLD
        )
        del myacro

        # metrics
        self.attack_metrics = [
            "dof_risk",
            "k_anonymity_risk",
            "class_disclosure_risk",
            "lowvals_cd_risk",
            "unnecessary_risk",
        ]
        self.yprobs = []

    def __str__(self) -> str:
        """Return the name of the attack."""
        return "Structural attack"

    def attack(self, target: Target) -> dict:
        """Run structural attack.

        To be used when code has access to Target class and trained target model.

        Parameters
        ----------
        target : attacks.target.Target
            target as a Target class object

        Returns
        -------
        dict
            Attack report.
        """
        self.target = target
        if target.model is None:
            errstr = (
                "cannot currently call StructuralAttack.attack() "
                "unless the target contains a trained model"
            )
            raise NotImplementedError(errstr)

        # get proba values for training data
        x = self.target.X_train
        y = self.target.y_train
        assert x.shape[0] == len(y), "length mismatch between trainx and trainy"
        self.yprobs = self.target.model.predict_proba(x)

        # only equivalence classes and membership once as potentially slow
        if isinstance(target.model, DecisionTreeClassifier):
            equiv = self.dt_get_equivalence_classes()
        else:
            equiv = self.get_equivalence_classes()
        equiv_classes = equiv[0]
        equiv_counts = equiv[1]
        equiv_members = equiv[2]
        errstr = "len mismatch between equiv classes and "
        assert len(equiv_classes) == len(equiv_counts), errstr + "counts"
        assert len(equiv_classes) == len(equiv_members), errstr + "membership"

        # now assess the risk
        # Degrees of Freedom
        n_params = get_model_param_count(target.model)
        residual_dof = self.target.X_train.shape[0] - n_params
        self.dof_risk = residual_dof < self.DOF_THRESHOLD

        # k-anonymity
        mink = np.min(np.array(equiv_counts))
        self.k_anonymity_risk = mink < self.THRESHOLD

        # unnecessary risk arising from poor hyper-parameter combination.
        self.unnecessary_risk = get_unnecessary_risk(self.target.model)

        # class disclosure
        freqs = np.zeros(equiv_classes.shape)
        for group in range(freqs.shape[0]):
            freqs = equiv_classes[group] * equiv_counts[group]
        self.class_disclosure_risk = np.any(freqs < self.THRESHOLD)
        freqs[freqs == 0] = 100
        self.lowvals_cd_risk = np.any(freqs < self.THRESHOLD)

        # create the report
        output = self._make_report(target)
        # write the report
        self._write_report(output)
        # return the report
        return output

    def dt_get_equivalence_classes(self) -> tuple:
        """Get details of equivalence classes based on white box inspection."""
        destinations = self.target.model.apply(self.target.X_train)
        ret_tuple = np.unique(destinations, return_counts=True)
        leaves = ret_tuple[0]
        counts = ret_tuple[1]
        members = []
        for leaf in leaves:
            ingroup = np.asarray(destinations == leaf).nonzero()[0]
            members.append(ingroup)

        equiv_classes = np.zeros((len(leaves), self.target.model.n_classes_))
        for group in range(len(leaves)):
            sample_id = members[group][0]
            sample = self.target.X_train[sample_id]
            proba = self.target.model.predict_proba(sample.reshape(1, -1))
            equiv_classes[group] = proba
        return [equiv_classes, counts, members]

    def get_equivalence_classes(self) -> tuple:
        """Get details of equivalence classes based on predicted probabilities."""
        uniques = np.unique(self.yprobs, axis=0, return_counts=True)
        equiv_classes = uniques[0]
        equiv_counts = uniques[1]
        members = []
        for prob_vals in equiv_classes:
            ingroup = np.unique(np.asarray(self.yprobs == prob_vals).nonzero()[0])
            members.append(ingroup)
        return [equiv_classes, equiv_counts, members]

    def _get_global_metrics(self, attack_metrics: list) -> dict:
        """Get dictionary summarising metrics from a metric list.

        Parameters
        ----------
        attack_metrics : list
            List of attack metrics to be reported.

        Returns
        -------
        global_metrics : dict
            Dictionary of summary metrics.
        """
        global_metrics = {}
        if attack_metrics is not None and len(attack_metrics) != 0:
            global_metrics["dof_risk"] = self.dof_risk
            global_metrics["k_anonymity_risk"] = self.k_anonymity_risk
            global_metrics["class_disclosure_risk"] = self.class_disclosure_risk
            global_metrics["unnecessary_risk"] = self.unnecessary_risk
            global_metrics["lowvals_cd_risk"] = self.lowvals_cd_risk
        return global_metrics

    def _construct_metadata(self):
        """Construct the metadata object, after attacks."""
        super()._construct_metadata()
        self.metadata["global_metrics"] = self._get_global_metrics(self.attack_metrics)

    def _get_attack_metrics_instances(self) -> dict:
        """Construct the metadata object, after attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}
        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        attack_metrics_experiment["dof_risk"] = self.dof_risk
        attack_metrics_experiment["k_anonymity_risk"] = self.k_anonymity_risk
        attack_metrics_experiment["class_disclosure_risk"] = self.class_disclosure_risk
        attack_metrics_experiment["unnecessary_risk"] = self.unnecessary_risk
        attack_metrics_experiment["lowvals_cd_risk"] = self.lowvals_cd_risk
        return attack_metrics_experiment

    def _make_pdf(self, output: dict) -> None:
        attack_name: str = output["metadata"]["attack_name"]
        logger.info("PDF report not yet implemented for %s", attack_name)
