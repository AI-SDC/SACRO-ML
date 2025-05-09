"""Privacy protected Random Forest classifier."""

from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sacroml.safemodel.reporting import get_reporting_string
from sacroml.safemodel.safemodel import SafeModel

from .safedecisiontreeclassifier import decision_trees_are_equal

# pylint: disable=too-many-ancestors


class SafeRandomForestClassifier(SafeModel, RandomForestClassifier):
    """Privacy protected Random Forest classifier."""

    def __init__(self, **kwargs: dict) -> None:
        """Create model and apply constraints to params."""
        SafeModel.__init__(self)
        self.basemodel_paramnames = [
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "bootstrap",
            "oob_score",
            "n_jobs",
            "random_state",
            "verbose",
            "warm_start",
            "class_weight",
            "ccp_alpha",
            "max_samples",
        ]

        the_kwds = {}
        for key, val in kwargs.items():
            if key in self.basemodel_paramnames:
                the_kwds[key] = val
        RandomForestClassifier.__init__(self, **the_kwds)
        self.model_type: str = "RandomForestClassifier"
        super().preliminary_check(apply_constraints=False, verbose=True)
        self.ignore_items = [
            "model_save_file",
            "ignore_items",
            "estimator_",  # this is an object
            "timestamp",
        ]
        self.examine_seperately_items = ["estimators_", "estimator"]
        self.k_anonymity = 0

    def additional_checks(  # pylint: disable=too-many-nested-blocks
        self, curr_separate: dict, saved_separate: dict
    ) -> tuple[str, str]:
        """Perform Random Forest specific checks.

        NOTE: this is never called if the model has not been fitted.
        """
        msg = ""
        disclosive = False
        # now the relevant random-forest specific things
        for item in self.examine_seperately_items:
            # template for class of things that make up forest
            if item == "estimator":
                if type(curr_separate[item]) != type(saved_separate[item]):
                    msg += get_reporting_string(
                        name="param_changed_from_to",
                        key="estimator",
                        val=saved_separate[item],
                        cur_val=curr_separate[item],
                    )
                    disclosive = True
            # the forest itself
            elif item == "estimators_":
                try:
                    num1 = len(curr_separate[item])
                    num2 = len(saved_separate[item])
                    if num1 != num2:
                        msg += get_reporting_string(
                            name="different_num_estimators", num1=num1, num2=num2
                        )
                        disclosive = True
                    else:
                        changed = False
                        num_diff_trees = 0
                        for idx in range(num1):
                            same, _ = decision_trees_are_equal(
                                curr_separate[item][idx], saved_separate[item][idx]
                            )
                            if not same:
                                changed = True
                                num_diff_trees += 1
                        if changed:
                            msg += get_reporting_string(
                                name="forest_estimators_differ", idx=num_diff_trees
                            )
                            disclosive = True
                except (
                    BaseException  # pylint: disable=broad-except
                ) as error:  # pragma:no coverion
                    msg += get_reporting_string(
                        name="unable_to_check_item", item=item, error=error
                    )
                    same = False

        return msg, disclosive

    # pylint: disable=arguments-differ
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit model and store model dict."""
        super().fit(x, y)
        self.k_anonymity = self.get_k_anonymity(x)
        self.saved_model = copy.deepcopy(self.__dict__)

    def get_k_anonymity(self, x: np.ndarray) -> int:
        """Calculate the k-anonymity of a random forest model.

        The k-anonymity is the minimum of the anonymity for each record.
        That is defined as the size of the set of records which
        appear in the same leaf as the record in every tree.
        """
        # dataset must be 2-D
        assert len(x.shape) == 2

        num_records = x.shape[0]
        num_trees = self.n_estimators
        k_anon_val = np.zeros(num_records, dtype=int)

        # ending leaf node by record(row) and tree (column)
        all_leaves = np.zeros((num_records, num_trees), dtype=int)
        for this_tree in range(num_trees):
            this_leaves = self.estimators_[this_tree].apply(x)
            for record in range(num_records):
                all_leaves[record][this_tree] = this_leaves[record]

        for record in range(num_records):
            # start by assuming everything co-occurs
            appears_together = list(range(num_records))
            # iterate through trees
            for this_tree in range(num_trees):
                this_leaf = all_leaves[record][this_tree]

                together = copy.copy(appears_together)
                # removing records which go to other leaves
                for other_record in together:
                    if all_leaves[other_record][this_tree] != this_leaf:
                        appears_together.remove(other_record)

            k_anon_val[record] = len(appears_together)
        return k_anon_val.min()
