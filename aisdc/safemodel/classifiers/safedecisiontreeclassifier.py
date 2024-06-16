"""Privacy protected Decision Tree classifier."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from dictdiffer import diff
from sklearn.tree import DecisionTreeClassifier

from aisdc.safemodel.reporting import get_reporting_string
from aisdc.safemodel.safemodel import SafeModel


def decision_trees_are_equal(
    tree1: DecisionTreeClassifier, tree2: DecisionTreeClassifier
) -> tuple[bool, str]:
    """Compare two estimators of type sklearn.tree."""
    msg = ""
    same = True

    try:
        tree1_dict = copy.deepcopy(tree1.__dict__)
        tree1_tree = tree1_dict.pop("tree_", "Absent")
        tree2_dict = copy.deepcopy(tree2.__dict__)
        tree2_tree = tree2_dict.pop("tree_", "Absent")

        # comparison on list of "simple" parameters
        match = list(diff(tree1_dict, tree2_dict, expand=True))
        num_differences = len(match)
        if num_differences > 0:
            same = False
            msg += get_reporting_string(
                name="basic_params_differ", length=num_differences
            )
            for i in range(num_differences):
                if match[i][0] == "change":
                    msg += f"parameter {match[i][1]} changed from {match[i][2][1]} "
                    msg += f"to {match[i][2][0]}\n"
                else:
                    msg += f"{match[i]}\n"

        # now internal tree params
        same2, msg2 = decision_tree_internal_trees_are_equal(tree1_tree, tree2_tree)
        if same2 is False:
            same = False
            msg += msg2

    except BaseException as error:  # pylint:disable=broad-except  #pragma:no cover
        msg += get_reporting_string(name="unable_to_check", error=error)
        same = False

    return same, msg


def decision_tree_internal_trees_are_equal(
    tree1_tree: Any, tree2_tree: Any
) -> tuple[bool, str]:
    """Test for equality of the internal structures in a sklearn.tree._tree.

    For example, the structure, feature and threshold in each internal node etc.
    """
    same = True
    msg = ""
    tree_internal_att_names = (
        "capacity",
        "children_left",
        "children_right",
        "feature",
        "impurity",
        "max_depth",
        "n_node_samples",
        "node_count",
        "threshold",
        "value",
        "weighted_n_node_samples",
    )

    try:
        if tree1_tree == "Absent" and tree2_tree == "Absent":
            msg += get_reporting_string(name="neither_tree_trained")
            # "neither tree trained"
        elif tree1_tree == "Absent":
            msg += get_reporting_string(name="tree1_not_trained")
            # "tree1 not trained"
            same = False
        elif tree2_tree == "Absent":
            msg += get_reporting_string(name="tree2_not_trained")
            # "tree2 not trained"
            same = False
        else:
            for attr in tree_internal_att_names:
                t1val = getattr(tree1_tree, attr)
                t2val = getattr(tree2_tree, attr)
                if isinstance(t1val, np.ndarray):
                    if not np.array_equal(t1val, t2val):
                        msg += get_reporting_string(
                            name="internal_attribute_differs", attr=attr
                        )
                        same = False
                elif t1val != t2val:
                    msg += get_reporting_string(
                        name="internal_attribute_differs", attr=attr
                    )
                    same = False
    except BaseException as error:  # pylint:disable=broad-except #pragma:no cover
        msg += get_reporting_string(name="exception_occurred", error=error)
    return same, msg


def get_tree_k_anonymity(thetree: DecisionTreeClassifier, X: Any) -> int:
    """Return the smallest number of data items in any leaf."""
    leaves = thetree.apply(X)
    uniqs_counts = np.unique(leaves, return_counts=True)
    return np.min(uniqs_counts[1])


class SafeDecisionTreeClassifier(SafeModel, DecisionTreeClassifier):  # pylint: disable=too-many-ancestors
    """Privacy protected Decision Tree classifier."""

    def __init__(self, **kwargs: dict) -> None:
        """Create model and apply constraints to params."""
        SafeModel.__init__(self)
        self.basemodel_paramnames = [
            "criterion",
            "splitter",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "random_state",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "class_weight",
            "ccp_alpha",
        ]

        the_kwds = {}
        for key, val in kwargs.items():
            if key in self.basemodel_paramnames:
                the_kwds[key] = val
        DecisionTreeClassifier.__init__(self, **the_kwds)
        self.model_type: str = "DecisionTreeClassifier"
        super().preliminary_check(apply_constraints=False, verbose=True)
        self.ignore_items = [
            "model_save_file",
            "basemodel_paramnames",
            "ignore_items",
            "timestamp",
        ]
        self.examine_seperately_items = ["tree_"]
        self.k_anonymity = 0

    def additional_checks(
        self, curr_separate: dict, saved_separate: dict
    ) -> tuple[str, str]:
        """Decision Tree-specific checks."""
        # call the super function to deal with any items that are lists
        # just in case we add any in the future
        msg, disclosive = super().additional_checks(curr_separate, saved_separate)
        # now deal with the decision-tree specific things
        # which for now means the attribute "tree_" which is a sklearn tree
        same, msg = decision_tree_internal_trees_are_equal(
            curr_separate["tree_"], saved_separate["tree_"]
        )
        if not same:
            disclosive = True
        if len(curr_separate) > 1:
            msg += get_reporting_string(name="unexpected_item")
        return msg, disclosive

    def fit(  # pylint: disable=arguments-differ
        self, x: np.ndarray, y: np.ndarray
    ) -> None:
        """Fit model and store k-anonymity and model dict."""
        super().fit(x, y)
        # calculate k-anonymity her since we have the tainigf data
        leaves = self.apply(x)
        uniqs_counts = np.unique(leaves, return_counts=True)
        self.k_anonymity = np.min(uniqs_counts[1])
        self.saved_model = copy.deepcopy(self.__dict__)
