"""
Structural_attack.py.

Runs a number of 'static' structural attacks,based on:
- the target model's properties
- the TREs risk appetite as applied to tables and standard regressions
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime

import numpy as np
from acro import ACRO

# tree-based model types currently supported
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

from aisdc.attacks import report
from aisdc.attacks.attack import Attack
from aisdc.attacks.attack_report_formatter import GenerateJSONModule
from aisdc.attacks.target import Target

logging.basicConfig(level=logging.INFO)


def get_unnecessary_risk(model: BaseEstimator) -> bool:
    """
    Checks whether a model's hyper-parameters against
    a set of rules that predict the top 20% most risky.

    This check is designed to assess whether a model is
    likely to be **unnecessarily** risky, i.e.,
    whether it is highly likely that a different combination of hyper-parameters
    would have led to model with similar or better accuracy on the task
    but with lower membership inference risk.

    The rules applied from an experimental study using a grid search in which:
    - max_features was one-hot encoded from the set [None, log2, sqrt]
    - splitter was encoded using 0=best, 1=random

    The target models created were then subject to membership inference attacks (MIA)
    and the hyper-param combinations rank-ordered according to MIA AUC.
    Then a decision tree trained to recognise whether
    hyper-params combintions were in the 20% most risky.
    The rules below were extracted from that tree for the 'least risky' nodes
    """
    # Returns 1 if high risk, otherwise 0
    if not isinstance(
        model, (DecisionTreeClassifier, RandomForestClassifier, XGBClassifier)
    ):
        return 0  # no experimental evidence to support rejection
    unnecessary_risk = 0
    max_depth = float(model.max_depth) if model.max_depth else 500

    # pylint:disable=chained-comparison,too-many-boolean-expressions
    if isinstance(model, DecisionTreeClassifier):
        max_features = model.max_features
        min_samples_leaf = model.min_samples_leaf
        min_samples_split = model.min_samples_split
        splitter = model.splitter
        if (
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
        ):
            unnecessary_risk = 1
    elif isinstance(model, RandomForestClassifier):
        n_estimators = model.n_estimators
        max_features = model.max_features
        min_samples_leaf = model.min_samples_leaf
        min_samples_split = model.min_samples_split
        if (
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
        ):
            unnecessary_risk = 1

    elif isinstance(model, XGBClassifier):
        n_estimators = model.n_estimators
        if n_estimators is None:
            n_estimators = 1000
        if (
            (
                max_depth > 3.5
                and 3.5 < n_estimators <= 12.5
                and model.min_child_weight <= 1.5
            )
            or (max_depth > 3.5 and n_estimators > 12.5 and model.min_child_weight <= 3)
            or (
                max_depth > 3.5
                and n_estimators > 62.5
                and 3 < model.min_child_weight <= 6
            )
        ):
            unnecessary_risk = 1
    return unnecessary_risk


def get_tree_parameter_count(dtree: DecisionTreeClassifier) -> int:
    """Reads the tree structure a returns the number of learned parameters."""
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
    """Returns the number of trained parameters in a model."""
    n_params = 0

    if isinstance(model, DecisionTreeClassifier):
        n_params = get_tree_parameter_count(model)

    elif isinstance(model, RandomForestClassifier):
        for member in model.estimators_:
            n_params += get_tree_parameter_count(member)

    elif isinstance(model, AdaBoostClassifier):
        try:  # sklearn v1.2+
            base = model.estimator
        except AttributeError:  # sklearn version <1.2
            base = model.base_estimator
        if isinstance(base, DecisionTreeClassifier):
            for member in model.estimators_:
                n_params += get_tree_parameter_count(member)

    # TO-DO define these for xgb, logistic regression, SVC and others
    elif isinstance(model, XGBClassifier):
        df = model.get_booster().trees_to_dataframe()
        n_trees = df["Tree"].max()
        total = len(df)
        n_leaves = len(df[df.Feature == "Leaf"])
        # 2 per internal node, one per clas in leaves, one weight per tree
        n_params = 2 * (total - n_leaves) + (model.n_classes_ - 1) * n_leaves + n_trees
    else:
        pass

    return n_params


class StructuralAttack(Attack):
    """Class to wrap a number of attacks based on the static structure of a model."""

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable = too-many-arguments
        self,
        attack_config_json_file_name: str = None,
        risk_appetite_config: str = "default",
        target_path: str = None,
        output_dir="outputs_structural",
        report_name="report_structural",
    ) -> None:
        """Constructs an object to execute a structural attack.

        Parameters
        ----------
        report_name : str
            name of the pdf and json output reports
        target_path : str
            path to the saved trained target model and target data
        risk_appetite_config : str
            path to yaml file specifying TRE risk appetite
        """

        super().__init__()
        logger = logging.getLogger("structural_attack")
        self.target: Target = None
        self.target_path = target_path
        self.attack_config_json_file_name = attack_config_json_file_name
        # disclosure risk
        self.k_anonymity_risk = 0
        self.DoF_risk = 0
        self.unnecessary_risk = 0
        self.class_disclosure_risk = 0
        self.lowvals_cd_risk = 0
        self.metadata = {}
        # make dummy acro object and use it to extract risk appetite
        myacro = ACRO(risk_appetite_config)
        self.risk_appetite_config = risk_appetite_config
        self.THRESHOLD = myacro.config["safe_threshold"]
        self.DOF_THRESHOLD = myacro.config["safe_dof_threshold"]
        logger.info(
            "Thresholds for count %i and Dof %i", self.THRESHOLD, self.DOF_THRESHOLD
        )
        del myacro
        if self.attack_config_json_file_name is not None:
            self._update_params_from_config_file()

        # metrics
        self.attack_metrics = [
            "DoF_risk",
            "k_anonymity_risk",
            "class_disclosure_risk",
            "lowvals_cd_risk",
            "unnecessary_risk",
        ]
        self.yprobs = []

        # paths for reporting
        self.output_dir = output_dir
        self.report_name = report_name

    def __str__(self):
        return "Structural attack"

    def attack(self, target: Target) -> None:
        """Programmatic attack entry point.

        To be used when code has access to Target class and trained target model

        Parameters
        ----------
        target : attacks.target.Target
            target as a Target class object
        """
        self.target = target
        if target.model is None:
            errstr = (
                "cannot currently call StructuralAttack.attack() "
                "unless the target contains a trained model"
            )
            raise NotImplementedError(errstr)

        # get proba values for training data
        x = self.target.x_train
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
        print(
            f"equiv_classes is {equiv_classes}\n"
            f"equiv_counts is {equiv_counts}\n"
            #   #f'equiv_members is {equiv_members}\n'
        )

        # now assess the risk
        # Degrees of Freedom
        n_params = get_model_param_count(target.model)
        residual_dof = self.target.x_train.shape[0] - n_params
        self.DoF_risk = 1 if residual_dof < self.DOF_THRESHOLD else 0

        # k-anonymity
        mink = np.min(np.array(equiv_counts))
        self.k_anonymity_risk = 1 if mink < self.THRESHOLD else 0

        # unnecessary risk arising from poor hyper-parameter combination.
        self.unnecessary_risk = get_unnecessary_risk(self.target.model)

        # class disclosure
        freqs = np.zeros(equiv_classes.shape)
        for group in range(freqs.shape[0]):
            freqs = equiv_classes[group] * equiv_counts[group]
        self.class_disclosure_risk = np.any(freqs < self.THRESHOLD).astype(int)
        freqs[freqs == 0] = 100
        self.lowvals_cd_risk = np.any(freqs < self.THRESHOLD).astype(int)

    def dt_get_equivalence_classes(self) -> tuple:
        """
        Gets details of equivalence classes
        based on white box inspection.
        """
        destinations = self.target.model.apply(self.target.x_train)
        ret_tuple = np.unique(destinations, return_counts=True)
        # print(f'leaves and counts:\n{ret_tuple}\n')
        leaves = ret_tuple[0]
        counts = ret_tuple[1]
        members = []
        for leaf in leaves:
            ingroup = np.asarray(destinations == leaf).nonzero()[0]
            # print(f'ingroup {ingroup},count {len(ingroup)}')
            members.append(ingroup)

        equiv_classes = np.zeros((len(leaves), self.target.model.n_classes_))
        for group in range(len(leaves)):
            sample_id = members[group][0]
            sample = self.target.x_train[sample_id]
            proba = self.target.model.predict_proba(sample.reshape(1, -1))
            equiv_classes[group] = proba
        return [equiv_classes, counts, members]

    def get_equivalence_classes(self) -> tuple:
        """
        Gets details of equivalence classes
        based on black box observation of probabilities.
        """
        uniques = np.unique(self.yprobs, axis=0, return_counts=True)
        equiv_classes = uniques[0]
        equiv_counts = uniques[1]
        members = []
        for prob_vals in equiv_classes:
            ingroup = np.unique(np.asarray(self.yprobs == prob_vals).nonzero()[0])
            members.append(ingroup)
        return [equiv_classes, equiv_counts, members]

    def _get_global_metrics(self, attack_metrics: list) -> dict:
        """Summarise metrics from a metric list.

        Returns
        -------
        global_metrics : Dict
            Dictionary of summary metrics

        Arguments
        ---------
        attack_metrics: List
            list of attack metrics to be reported
        """
        global_metrics = {}
        if attack_metrics is not None and len(attack_metrics) != 0:
            global_metrics["DoF_risk"] = self.DoF_risk
            global_metrics["k_anonymity_risk"] = self.k_anonymity_risk
            global_metrics["class_disclosure_risk"] = self.class_disclosure_risk
            global_metrics["unnecessary_risk"] = self.unnecessary_risk
            global_metrics["lowvals_cd_risk"] = self.lowvals_cd_risk

        return global_metrics

    def _construct_metadata(self):
        """Constructs the metadata object, after attacks."""
        self.metadata = {}
        # Store all args
        self.metadata["experiment_details"] = {}
        self.metadata["experiment_details"] = self.get_params()
        self.metadata["attack"] = str(self)
        # Global metrics
        self.metadata["global_metrics"] = self._get_global_metrics(self.attack_metrics)

    def _get_attack_metrics_instances(self) -> dict:
        """Constructs the metadata object, after attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}

        # for rep, name in enumerate(self.attack_metrics):
        #    #attack_metrics_instances["instance_" + str(rep)] = self.attack_metrics[rep]
        #    attack_metrics_instances["instance_" + str(name)] = self.__dict__.[name]

        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        attack_metrics_experiment["DoF_risk"] = self.DoF_risk
        attack_metrics_experiment["k_anonymity_risk"] = self.k_anonymity_risk
        attack_metrics_experiment["class_disclosure_risk"] = self.class_disclosure_risk
        attack_metrics_experiment["unnecessary_risk"] = self.unnecessary_risk
        attack_metrics_experiment["lowvals_cd_risk"] = self.lowvals_cd_risk
        return attack_metrics_experiment

    def make_report(self) -> dict:
        """Creates output dictionary structure and generates
        pdf and json outputs if filenames are given.
        """
        output = {}
        output["log_id"] = str(uuid.uuid4())
        output["log_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        self._construct_metadata()
        output["metadata"] = self.metadata

        output["attack_experiment_logger"] = self._get_attack_metrics_instances()
        # output[
        #     "dummy_attack_experiments_logger"
        # ] = self._get_dummy_attack_metrics_experiments_instances()

        report_dest = os.path.join(self.output_dir, self.report_name)
        json_attack_formatter = GenerateJSONModule(report_dest + ".json")
        json_report = report.create_json_report(output)
        json_attack_formatter.add_attack_output(json_report, "StructuralAttack")

        # pdf_report = report.create_mia_report(output)
        # report.add_output_to_pdf(report_dest, pdf_report, "StructuralAttack")
        return output


def _run_attack(args):
    """Initialise class and run attack."""

    attack_obj = StructuralAttack(
        risk_appetite_config=args.risk_appetite_config,
        target_path=args.target_path,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )

    target = Target()
    target.load(attack_obj.target_path)
    attack_obj.attack(target)
    _ = attack_obj.make_report()


def _run_attack_from_configfile(args):
    """Initialise class and run attack  using config file."""

    attack_obj = StructuralAttack(
        attack_config_json_file_name=str(args.attack_config_json_file_name),
        target_path=str(args.target_path),
    )
    target = Target()
    target.load(attack_obj.target_path)
    attack_obj.attack(target)
    _ = attack_obj.make_report()


def main():
    """Main method to parse arguments and invoke relevant method."""
    logger = logging.getLogger("main")
    parser = argparse.ArgumentParser(description="Perform a structural  attack")

    subparsers = parser.add_subparsers()

    attack_parser = subparsers.add_parser("run-attack")

    attack_parser.add_argument(
        "--output-dir",
        type=str,
        action="store",
        dest="output_dir",
        default="output_structural",
        required=False,
        help=("Directory name where output files are stored. Default = %(default)s."),
    )

    attack_parser.add_argument(
        "--report-name",
        type=str,
        action="store",
        dest="report_name",
        default="report_structural",
        required=False,
        help=(
            """Filename for the pdf and json report outputs. Default = %(default)s.
            Code will append .pdf and .json"""
        ),
    )

    attack_parser.add_argument(
        "--risk-appetite-filename",
        action="store",
        type=str,
        default="default",
        required=False,
        dest="risk_appetite_config",
        help=(
            """provide the name of the dataset-specific risk appetite filename
            using --risk-appetite-filename Default = %(default)s"""
        ),
    )

    attack_parser.add_argument(
        "--target-path",
        action="store",
        type=str,
        default=None,
        required=False,
        dest="target_path",
        help=(
            """Provide the path to the stored target usinmg
             --target-path option. Default = %(default)f"""
        ),
    )

    attack_parser.set_defaults(func=_run_attack)

    attack_parser_config = subparsers.add_parser("run-attack-from-configfile")
    attack_parser_config.add_argument(
        "-j",
        "--attack-config-json-file-name",
        action="store",
        required=True,
        dest="attack_config_json_file_name",
        type=str,
        default="config_structural_cmd.json",
        help=(
            "Name of the .json file containing details for the run. Default = %(default)s"
        ),
    )

    attack_parser_config.add_argument(
        "-t",
        "--attack-target-folder-path",
        action="store",
        required=True,
        dest="target_path",
        type=str,
        default="structural_target",
        help=(
            """Name of the target directory to load the trained target model and the target data.
            Default = %(default)s"""
        ),
    )

    attack_parser_config.set_defaults(func=_run_attack_from_configfile)

    args = parser.parse_args()

    try:
        args.func(args)
    except AttributeError as e:  # pragma:no cover
        logger.error("Invalid command. Try --help to get more details")
        logger.error(e)


if __name__ == "__main__":  # pragma:no cover
    main()
