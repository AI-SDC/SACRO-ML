"""This module contains unit tests for the SafeDecisionTreeClassifier."""

import pickle

import joblib
import numpy as np
from sklearn import datasets

from safemodel.classifiers import SafeDecisionTreeClassifier


def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


def test_decisiontree_unchanged():
    """SafeDecisionTreeClassifier using unchanged values."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1)
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_decisiontree_safe_recommended():
    """SafeDecisionTreeClassifier using recommended values."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1)
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
    model = SafeDecisionTreeClassifier(random_state=1)
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
    model = SafeDecisionTreeClassifier(random_state=1)
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


def test_decisiontree_unsafe_2():
    """SafeDecisionTreeClassifier with unsafe changes - automatically fixed."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1, min_samples_leaf=1)
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


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


def test_decisiontree_hacked_postfit():
    """SafeDecisionTreeClassifier changes made to parameters after fit() called."""
    x, y = get_data()
    model = SafeDecisionTreeClassifier(random_state=1, min_samples_leaf=1)
    model.min_samples_leaf = 1
    model.fit(x, y)
    assert model.score(x, y) == 1.0
    model.min_samples_leaf = 5
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False
    msg2, disclosive2 = model.posthoc_check()
    correct_msg2 = (
        "Warning: basic parameters differ in 1 places:\n"
        "parameter min_samples_leaf changed from 1 to 5 after model was fitted.\n"
    )
    assert msg2 == correct_msg2
    assert disclosive2 is True
