"""This module contains unit tests for the SafeDecisionTreeClassifier."""

from __future__ import annotations

import os
import pickle

import joblib
import numpy as np
from sklearn import datasets

from aisdc.safemodel import reporting
from aisdc.safemodel.classifiers import SafeDecisionTreeClassifier
from aisdc.safemodel.classifiers.safedecisiontreeclassifier import (
    decision_trees_are_equal,
    get_tree_k_anonymity,
)


def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris["data"], dtype=np.float64)
    y = np.asarray(iris["target"], dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


def test_reporting():
    """Check that getting report name requires name."""
    correct_msg = "Error - get_reporting_string: No 'name' given"
    msg = reporting.get_reporting_string()
    assert msg == correct_msg


def test_decision_trees_are_equal():
    """Test the code that compares two decision trees."""
    x1, y = get_data()
    model1 = SafeDecisionTreeClassifier(random_state=1)
    model1.fit(x1, y)

    # same
    model2 = SafeDecisionTreeClassifier(random_state=1)
    model2.fit(x1, y)
    same, _ = decision_trees_are_equal(model1, model2)

    # one or both untrained
    model3 = SafeDecisionTreeClassifier(random_state=1, max_depth=7)
    same, msg = decision_trees_are_equal(model1, model3)
    assert same is False
    assert len(msg) > 0
    same, msg = decision_trees_are_equal(model3, model1)
    assert same is False
    assert len(msg) > 0
    same, msg = decision_trees_are_equal(model3, model3)
    assert same is True
    # assert len(msg)>0

    # different
    # x2=x1+1
    model3 = SafeDecisionTreeClassifier(random_state=1, max_depth=7)
    model3.criterion = "entropy"
    model3.fit(x1, y)
    same, msg = decision_trees_are_equal(model1, model3)
    print(f"diff msg = {msg}")
    assert same is False

    # wrong type
    same, _ = decision_trees_are_equal(model1, "aString")
    assert same is False


def test_get_tree_k_anonymity():
    """Getting k_anonymity
    50 data points randomly split in 2, single layer
    so k should be ~25.
    """
    x = np.random.rand(50, 2)
    y = np.ones(50)
    for i in range(x.shape[0]):
        if x[i][0] < 0.5:
            y[i] = 0
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=1)
    model.fit(x, y)
    k = get_tree_k_anonymity(model, x)
    # print(f'k={k}')
    assert k > 10


def test_decisiontree_unchanged():
    """SafeDecisionTreeClassifier using unchanged values."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1, min_samples_leaf=5)
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_decisiontree_safe_recommended():
    """SafeDecisionTreeClassifier using recommended values."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(
        random_state=1, max_depth=10, min_samples_leaf=10
    )
    model.min_samples_leaf = 5
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_decisiontree_safe_1():
    """SafeDecisionTreeClassifier with safe changes."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(
        random_state=1, max_depth=10, min_samples_leaf=10
    )
    model.min_samples_leaf = 10
    model.fit(x, y)
    assert model.score(x, y) == 0.9536423841059603
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_decisiontree_safe_2():
    """SafeDecisionTreeClassifier with safe changes."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1, min_samples_leaf=10)
    model.fit(x, y)
    assert model.score(x, y) == 0.9536423841059603
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_decisiontree_unsafe_1():
    """SafeDecisionTreeClassifier with unsafe changes."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(
        random_state=1, max_depth=10, min_samples_leaf=10
    )
    model.min_samples_leaf = 1
    model.fit(x, y)
    assert model.score(x, y) == 1
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter min_samples_leaf = 1 identified as less than the recommended "
        "min value of 5."
    )
    assert msg == correct_msg
    assert disclosive is True


# no longer relevant because of changed default functionality of priminary_checK()
# def test_decisiontree_unsafe_2():
#     """SafeDecisionTreeClassifier with unsafe changes - automatically fixed."""
#     x, y = get_data()
#     model = SafeDecisionTreeClassifier(random_state=1, min_samples_leaf=1)
#     model.fit(x, y)
#     assert model.score(x, y) == 0.9668874172185431
#     msg, disclosive = model.preliminary_check()
#     correct_msg = "Model parameters are within recommended ranges.\n"
#     assert msg == correct_msg
#     assert disclosive is False


def test_decisiontree_save():
    """SafeDecisionTreeClassifier model saving."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1, min_samples_leaf=50)
    model.fit(x, y)
    assert model.score(x, y) == 0.9470198675496688
    # test pickle
    model.save("dt_test.pkl")
    with open("dt_test.pkl", "rb") as file:
        pkl_model = pickle.load(file)
    assert pkl_model.score(x, y) == 0.9470198675496688
    # test joblib
    model.save("dt_test.sav")
    with open("dt_test.sav", "rb") as file:
        sav_model = joblib.load(file)
    assert sav_model.score(x, y) == 0.9470198675496688

    # cleanup
    for name in ("dt_test.pkl", "dt_test.sav"):
        if os.path.exists(name) and os.path.isfile(name):
            os.remove(name)


def test_decisiontree_hacked_postfit():
    """SafeDecisionTreeClassifier changes made to parameters after fit() called."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1, min_samples_leaf=1)
    model.min_samples_leaf = 1
    model.fit(x, y)
    assert model.score(x, y) == 1.0
    model.min_samples_leaf = 5

    # forcing errors in internal tree checking by hacking tree
    model.tree_.max_depth = 5
    model.tree_.value[0][-1] = 1

    # At first glance the model now looks ok
    msg1, disclosive = model.preliminary_check()
    correct_msg1 = reporting.get_reporting_string(name="within_recommended_ranges")
    print(f"correct msg1: {correct_msg1}\n", f"actual  msg1: {msg1}")
    assert msg1 == correct_msg1
    assert disclosive is False

    # but on closer inspection not
    msg2, disclosive2 = model.posthoc_check()
    part1 = reporting.get_reporting_string(name="basic_params_differ", length=1)
    part2 = "parameter min_samples_leaf changed from 1 to 5 after model was fitted.\n"
    part3 = reporting.get_reporting_string(
        name="internal_attribute_differs", attr="max_depth"
    )
    part4 = reporting.get_reporting_string(
        name="internal_attribute_differs", attr="value"
    )

    correct_msg2 = part1 + part2 + part3 + part4
    print(f"Correct msg2 : {correct_msg2}\n" f"actual mesg2 : {msg2}")
    assert msg2 == correct_msg2
    assert disclosive2 is True


def test_data_hiding():
    """What if the hacking was really obscure
    like putting something in the exceptions list
     then adding data to current and saved copies.
    """
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1, min_samples_leaf=5)
    model.fit(x, y)
    # now the hack
    model.examine_seperately_items = ["tree_", "bad_string"]
    model.bad_string = "something disclosive"
    model.saved_model["bad_string"] = "something disclosive"
    part1 = reporting.get_reporting_string(name="basic_params_differ", length=1)
    part2 = "('remove', 'examine_seperately_items', [(1, 'bad_string')])"
    part3 = reporting.get_reporting_string(name="unexpected_item")
    correct_msg = part1 + part2 + part3
    msg, disclosive = model.posthoc_check()
    print(f"Correct msg : {correct_msg}\n" f"actual  msg : {msg}")
    assert disclosive
    assert msg == correct_msg
