"""Privacy protected Decision Tree classifier."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from dictdiffer import diff
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree

from ..safemodel import SafeModel


def decision_trees_are_equal(
    tree1: sklearn.tree, tree2: sklearn.tree
) -> tuple[bool, str]:
    """Compares two estimators of type sklearn.tree
    e.g. two decisionTreeClassifiers
    """
    msg = ""
    same = True

    try:
        tree1_dict = copy.deepcopy(tree1.__dict__)
        tree1_tree = tree1_dict.pop("tree_", "Absent")
        tree2_dict = copy.deepcopy(tree2.__dict__)
        tree2_tree = tree2_dict.pop("tree_", "Absent")

        # comparison on list of "simple" parameters
        match = list(diff(tree1_dict, tree2_dict, expand=True))
        if len(match) > 0:
            same = False
            msg += f"Warning: basic parameters differ in {len(match)} places:\n"
            for i in range(len(match)):
                if match[i][0] == "change":
                    msg += f"parameter {match[i][1]} changed from {match[i][2][1]} "
                    msg += f"to {match[i][2][0]}\n"
                else:
                    msg += f"{match[i]}\n"

        # now internal tree params
        same2, msg2 = decision_tree_internal_trees_are_equal(tree1_tree, tree2_tree)
        if same2 == False:
            same = False
            msg += msg2

    except BaseException as error:
        msg += f"Unable to check as an exception occurred: {error}"
        same = False

    return same, msg


def decision_tree_internal_trees_are_equal(
    tree1_tree: Tree, tree2_tree: Tree
) -> tuple[bool, str]:
    """Tests for equality of the internal structures in a sklearn.tree._tree
    e.g. the structure, feature and threshold in each internal node etc."""

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
            msg += "neither tree trained"
        elif tree1_tree == "Absent":
            msg += "tree1 not trained"
            same = False
        elif tree2_tree == "Absent":
            msg += "tree2 not trained"
            same = False
        else:
            for attr in tree_internal_att_names:
                t1val = getattr(tree1_tree, attr)
                t2val = getattr(tree2_tree, attr)
                if isinstance(t1val, np.ndarray):
                    if not np.array_equal(t1val, t2val):
                        msg += f"internal tree attribute {attr} differs\n"
                        same = False
                else:
                    if t1val != t2val:
                        msg += f"internal tree attribute {attr} differs\n"
                        same = False
    except BaseException as error:
        msg += f"An exception occurred: {error}"
    return same, msg


def get_tree_k_anonymity(thetree: sklearn.tree) -> int:
    leaves = thetree.apply(X)
    uniqs_counts = np.unique(leaves, return_counts=True)
    k_anonymity = np.min(uniqs_counts[1])
    # print(f' leaf ids {uniqs_counts[0]} and counts {uniqs_counts[1]} the  k-anonymity of the tree is {k_anonymity}')
    return k_anonymity


class SafeDecisionTreeClassifier(SafeModel, DecisionTreeClassifier):
    """Privacy protected Decision Tree classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Creates model and applies constraints to params."""
        SafeModel.__init__(self)
        DecisionTreeClassifier.__init__(self, **kwargs)
        self.model_type: str = "DecisionTreeClassifier"
        super().preliminary_check(apply_constraints=True, verbose=True)
        self.ignore_items = ["model_save_file", "ignore_items"]
        self.examine_seperately_items = ["tree_"]

    def additional_checks(
        self, curr_separate: dict, saved_separate: dict
    ) -> tuple[str, str]:
        """Decision Tree-specific checks"""
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
            msg += (
                "unexpected item in curr_seperate dict "
                " passed by generic additional checks."
            )

        return msg, disclosive

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Do fit and then store k-anonymity and  model dict"""
        super().fit(x, y)
        # calculate k-anonymity her since we have the tainigf data
        leaves = self.apply(x)
        uniqs_counts = np.unique(leaves, return_counts=True)
        self.k_anonymity = np.min(uniqs_counts[1])
        self.saved_model = copy.deepcopy(self.__dict__)
