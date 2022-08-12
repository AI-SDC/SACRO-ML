"""This module contains unit tests for the SafeRandomForestClassifier."""

import pickle

import joblib
import numpy as np
from sklearn import datasets

from safemodel.classifiers import SafeRandomForestClassifier


def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


def test_randomforest_unchanged():
    """SafeRandomForestClassifier using recommended values."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1)
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_randomforest_recommended():
    """SafeRandomForestClassifier using recommended values."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1)
    model.min_samples_leaf = 6
    model.fit(x, y)
    print(f"model.dict={model.__dict__}")
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_randomforest_unsafe_1():
    """SafeRandomForestClassifier with unsafe changes."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1)
    model.bootstrap = False
    model.fit(x, y)
    assert model.score(x, y) == 0.9735099337748344
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter bootstrap = False identified as different than the recommended "
        "fixed value of True."
    )
    assert msg == correct_msg
    assert disclosive is True


def test_randomforest_unsafe_2():
    """SafeRandomForestClassifier with unsafe changes."""
    model = SafeRandomForestClassifier(random_state=1)
    model.bootstrap = True
    model.min_samples_leaf = 2
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter min_samples_leaf = 2 identified as less than the recommended "
        "min value of 5."
    )
    assert msg == correct_msg
    assert disclosive is True


def test_randomforest_unsafe_3():
    """SafeRandomForestClassifier with unsafe changes."""
    model = SafeRandomForestClassifier(random_state=1)
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
    assert msg == correct_msg
    assert disclosive is True


def test_randomforest_save():
    """SafeRandomForestClassifier model saving."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1, min_samples_leaf=50)
    model.fit(x, y)
    assert model.score(x, y) == 0.6622516556291391
    # test pickle
    model.save("rf_test.pkl")
    with open("rf_test.pkl", "rb") as file:
        pkl_model = pickle.load(file)
    assert pkl_model.score(x, y) == 0.6622516556291391
    # test joblib
    model.save("rf_test.sav")
    with open("rf_test.sav", "rb") as file:
        sav_model = joblib.load(file)
    assert sav_model.score(x, y) == 0.6622516556291391


def test_randomforest_hacked_postfit():
    """SafeRandomForestClassifier changes made to parameters after fit() called."""
    x, y = get_data()
    model = SafeRandomForestClassifier(random_state=1)
    model.bootstrap = False
    model.fit(x, y)
    assert model.score(x, y) == 0.9735099337748344
    model.bootstrap = True
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False
    msg2, disclosive2 = model.posthoc_check()
    correct_msg2 = (
        "Warning: basic parameters differ in 1 places:\n"
        "parameter bootstrap changed from False to True after model was fitted.\n"
        "Warning: at least one non-matching value for parameter list estimators_.\n"
        "Warning: at least one non-matching value for parameter list estimators_.\n"
    )
    assert msg2 == correct_msg2
    assert disclosive2 is True
