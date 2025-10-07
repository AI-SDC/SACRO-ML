"""Tests for the SafeRandomForestClassifier."""

from __future__ import annotations

import copy
import pickle

import joblib
import numpy as np
from sklearn import datasets

from sacroml.safemodel.classifiers import SafeRandomForestClassifier
from sacroml.safemodel.reporting import get_reporting_string

EXPECTED_ACC = 0.9470198675496688  # 5 estimators, min_samples_leaf 5


class DummyClassifier:
    """Dummy Classifier that always returns predictions of zero."""

    def __init__(self):
        """Empty init."""

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Empty fit."""

    def predict(self, x: np.ndarray):
        """Predict all ones."""


def get_data():
    """Return data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris["data"], dtype=np.float64)
    y = np.asarray(iris["target"], dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


def test_randomforest_unchanged():
    """Test using recommended values."""
    x, y = get_data()
    model = SafeRandomForestClassifier(
        random_state=1, n_estimators=5, min_samples_leaf=5
    )
    model.fit(x, y)
    assert model.score(x, y) == EXPECTED_ACC
    msg, disclosive = model.preliminary_check()
    correct_msg = get_reporting_string(name="within_recommended_ranges")
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert not disclosive


def test_randomforest_recommended():
    """Test using recommended values."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1, n_estimators=5)
    model.min_samples_leaf = 6
    model.fit(x, y)
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert not disclosive


def test_randomforest_unsafe_1():
    """Test with unsafe changes."""
    x, y = get_data()
    model = SafeRandomForestClassifier(
        random_state=1, n_estimators=5, min_samples_leaf=5
    )
    model.bootstrap = False
    model.fit(x, y)
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter bootstrap = False identified as different than the recommended "
        "fixed value of True."
    )
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert disclosive


def test_randomforest_unsafe_2():
    """Test with unsafe changes."""
    model = SafeRandomForestClassifier(random_state=1, n_estimators=5)
    model.bootstrap = True
    model.min_samples_leaf = 2
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter min_samples_leaf = 2 identified as less than the recommended "
        "min value of 5."
    )
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert disclosive


def test_randomforest_unsafe_3():
    """Test with unsafe changes."""
    model = SafeRandomForestClassifier(random_state=1, n_estimators=5)
    model.bootstrap = False
    model.min_samples_leaf = 2
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter bootstrap = False identified as different than the recommended "
        "fixed value of True."
        "- parameter min_samples_leaf = 2 identified as less than the recommended "
        "min value of 5."
    )
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert disclosive


def test_randomforest_save():
    """Test model saving."""
    x, y = get_data()
    model = SafeRandomForestClassifier(
        random_state=1, n_estimators=5, min_samples_leaf=5
    )
    model.fit(x, y)
    # test pickle
    model.save("rf_test.pkl")
    with open("rf_test.pkl", "rb") as file:
        pkl_model = pickle.load(file)
    assert pkl_model.score(x, y) == EXPECTED_ACC
    # test joblib
    model.save("rf_test.sav")
    with open("rf_test.sav", "rb") as file:
        sav_model = joblib.load(file)
    assert sav_model.score(x, y) == EXPECTED_ACC


def test_randomforest_hacked_postfit():
    """Test changes made to parameters after fit() called."""
    x, y = get_data()
    model = SafeRandomForestClassifier(
        random_state=1, n_estimators=5, min_samples_leaf=5
    )
    model.bootstrap = False
    model.fit(x, y)
    model.bootstrap = True
    # preliminary check looks ok
    msg, disclosive = model.preliminary_check()
    correct_msg = get_reporting_string(name="within_recommended_ranges")
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert not disclosive
    # but more detailed analysis says not
    msg2, disclosive2 = model.posthoc_check()
    part1 = get_reporting_string(name="basic_params_differ", length=1)
    part2 = get_reporting_string(
        name="param_changed_from_to", key="bootstrap", val=False, cur_val=True
    )
    part3 = ""
    correct_msg2 = part1 + part2 + part3

    assert msg2 == correct_msg2, f"{msg2}\n should be {correct_msg2}"
    assert disclosive2


def test_not_fitted():
    """Test Posthoc_check() called on unfitted model."""
    unfitted_model = SafeRandomForestClassifier(random_state=1, n_estimators=5)

    # not fitted
    msg, disclosive = unfitted_model.posthoc_check()
    part1 = get_reporting_string(name="error_not_called_fit")
    part2 = get_reporting_string(name="recommend_do_not_release")
    correct_msg = part1 + part2
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert disclosive


def test_randomforest_modeltype_changed():
    """Test model type has been changed after fit()."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1, n_estimators=5)
    correct_msg = ""
    # check code that tests estimator_
    model.fit(x, y)
    model.estimator = "DummyClassifier()"

    # hide some data
    for i in range(5):  # changed lengths get picked up in different test
        model.estimators_[i] = x[i, :]

    msg, disclosive = model.posthoc_check()
    correct_msg = get_reporting_string(name="forest_estimators_differ", idx=5)
    correct_msg += get_reporting_string(
        name="param_changed_from_to",
        key="estimator",
        val="DecisionTreeClassifier()",
        cur_val="DummyClassifier()",
    )
    print(f"Correct: {correct_msg} Actual: {msg}")

    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert disclosive, "should have been flagged as disclosive"


def test_randomforest_hacked_postfit_trees_removed():
    """Test various combinations of removing trees."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1, n_estimators=5)
    # code that checks estimators_ :
    # one other or both missing or different number or size
    model.fit(x, y)

    # tree removed from current
    the_estimators = model.__dict__.pop("estimators_")
    msg, disclosive = model.posthoc_check()
    correct_msg = get_reporting_string(name="current_item_removed", item="estimators_")
    assert disclosive, "should be flagged as disclosive"
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"

    # trees removed from both
    _ = model.saved_model.pop("estimators_")
    msg, disclosive = model.posthoc_check()
    correct_msg = get_reporting_string(name="both_item_removed", item="estimators_")
    assert disclosive, "should be flagged as disclosive"
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"

    # trees just removed from saved
    model.estimators_ = the_estimators
    msg, disclosive = model.posthoc_check()
    correct_msg = get_reporting_string(name="saved_item_removed", item="estimators_")
    assert disclosive, "should be flagged as disclosive"
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"


def test_randomforest_hacked_postfit_trees_swapped():
    """Test trees swapped with those from a different random forest."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1, n_estimators=5)
    diffsizemodel = SafeRandomForestClassifier(
        random_state=1, n_estimators=5, max_depth=2
    )

    # code that checks estimators_ :
    # one other or both missing or different number or size
    model.fit(x, y)
    diffsizemodel.fit(x, y)

    # swap saved models
    the_saved_model = copy.deepcopy(model.saved_model)
    the_saved_diffmodel = copy.deepcopy(diffsizemodel.saved_model)
    model.saved_model = copy.deepcopy(the_saved_diffmodel)
    diffsizemodel.saved_model = copy.deepcopy(the_saved_model)
    model.posthoc_check()
    msg, disclosive = diffsizemodel.posthoc_check()
    part1 = get_reporting_string(name="basic_params_differ", length=1)
    part2 = get_reporting_string(
        name="param_changed_from_to", key="max_depth", val="None", cur_val="2"
    )
    part3 = get_reporting_string(name="forest_estimators_differ", idx=5)
    part4 = ""
    correct_msg = part1 + part2 + part3 + part4
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert disclosive, "should be flagged as disclosive"


def test_randomforest_hacked_postfit_moretrees():
    """Test trees added after fit."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1, n_estimators=5)
    diffsizemodel = SafeRandomForestClassifier(random_state=1, n_estimators=10)
    model.fit(x, y)
    diffsizemodel.fit(x, y)

    # swap saved models
    the_saved_model = copy.deepcopy(model.saved_model)
    the_saved_diffmodel = copy.deepcopy(diffsizemodel.saved_model)
    model.saved_model = copy.deepcopy(the_saved_diffmodel)
    diffsizemodel.saved_model = copy.deepcopy(the_saved_model)
    msg, disclosive = diffsizemodel.posthoc_check()
    part1 = get_reporting_string(name="basic_params_differ", length=1)
    part2 = get_reporting_string(
        name="param_changed_from_to", key="n_estimators", val="5", cur_val="10"
    )
    part3 = get_reporting_string(name="different_num_estimators", num1=10, num2=5)
    part4 = ""
    correct_msg = part1 + part2 + part3 + part4
    assert msg == correct_msg, f"{msg}\n should be {correct_msg}"
    assert disclosive, "should be flagged as disclosive"
