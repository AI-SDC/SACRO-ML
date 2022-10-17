""" tests for fnctionality in super class"""
import os
import pickle

import joblib
import numpy as np
from sklearn import datasets

from safemodel.reporting import get_reporting_string
from safemodel.safemodel import SafeModel

notok_start = get_reporting_string(name="warn_possible_disclosure_risk")
ok_start = get_reporting_string(name="within_recommended_ranges")


class DummyClassifier:
    """Dummy Classifier that always returns predictions of zero"""

    def __init__(self, at_least_5f=5.0, at_most_5i=5, exactly_boo="boo"):
        """dummy init"""
        self.at_least_5f = at_least_5f
        self.at_most_5i = at_most_5i
        self.exactly_boo = exactly_boo

    def fit(self, x: np.ndarray, y: np.ndarray):
        """dummy fit"""

    def predict(self, x: np.ndarray):
        """predict all ones"""
        return np.ones(x.shape[0])


def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris["data"], dtype=np.float64)
    y = np.asarray(iris["target"], dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


class SafeDummyClassifier(
    SafeModel, DummyClassifier
):  # pylint:disable=too-many-instance-attributes
    """Privacy protected dummy classifier."""

    def __init__(self, **kwargs) -> None:
        """Creates model and applies constraints to params."""
        SafeModel.__init__(self)
        self.basemodel_paramnames = ("at_least_5f", "at_most_5i", "exactly_boo")
        the_kwds = {}
        for key, val in kwargs.items():
            if key in self.basemodel_paramnames:
                the_kwds[key] = val
        DummyClassifier.__init__(self, **the_kwds)
        self.model_type: str = "DummyClassifier"
        self.ignore_items = ["model_save_file", "basemodel_paramnames", "ignore_items"]
        # create an item to test additional_checks()
        self.examine_seperately_items = ["newthing"]
        self.newthing = {"myStringKey": "aString", "myIntKey": 42}

    def set_params(self, **kwargs):
        """sets params"""
        for key, val in kwargs.items():  # pylint:disable=unused-variable
            self.key = val  # pylint:disable=attribute-defined-outside-init


def test_params_checks_ok():
    """test parameter  checks ok"""
    model = SafeDummyClassifier()

    correct_msg = ok_start
    msg, disclosive = model.preliminary_check()
    print(
        f"exactly_boo is {model.exactly_boo} with type{type(model.exactly_boo).__name__}"
    )
    assert msg == ok_start, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert disclosive is False


def test_params_checks_too_low():
    """test parameter  checks too low"""
    model = SafeDummyClassifier()

    model.at_least_5f = 4.0
    msg, disclosive = model.preliminary_check()
    assert disclosive is True
    correct_msg = notok_start + get_reporting_string(
        name="less_than_min_value",
        key="at_least_5f",
        cur_val=model.at_least_5f,
        val=5.0,
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"


def test_params_checks_too_high():
    """test parameter  checks too high"""
    model = SafeDummyClassifier()

    model.at_most_5i = 6
    msg, disclosive = model.preliminary_check()
    assert disclosive is True
    correct_msg = notok_start + get_reporting_string(
        name="greater_than_max_value", key="at_most_5i", cur_val=model.at_most_5i, val=5
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"


def test_params_checks_not_equal():
    """test parameter  checks not equal"""
    model = SafeDummyClassifier()

    model.exactly_boo = "foo"
    msg, disclosive = model.preliminary_check()
    assert disclosive is True
    correct_msg = notok_start + get_reporting_string(
        name="different_than_fixed_value",
        key="exactly_boo",
        cur_val=model.exactly_boo,
        val="boo",
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"


def test_params_checks_wrong_type_str():
    """test parameter  checks wrong type - strings given"""
    model = SafeDummyClassifier()

    model.at_least_5f = "five"
    model.at_most_5i = "five"

    msg, disclosive = model.preliminary_check()
    assert disclosive is True
    correct_msg = notok_start
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_least_5f",
        cur_val=model.at_least_5f,
        val=5.0,
    )
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_least_5f",
        cur_val=model.at_least_5f,
        val="float",
    )
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_most_5i",
        cur_val=model.at_most_5i,
        val=5,
    )
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_most_5i",
        cur_val=model.at_most_5i,
        val="int",
    )

    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"


def test_params_checks_wrong_type_float():
    """test parameter  checks wrong_type_float"""
    model = SafeDummyClassifier()

    model.exactly_boo = 5.0
    model.at_most_5i = 5.0

    _, disclosive = model.preliminary_check()
    assert disclosive is True
    correct_msg = notok_start

    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_most_5",
        cur_val=model.at_most_5i,
        val="int",
    )
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="exactly_5",
        cur_val=model.exactly_boo,
        val="int",
    )


def test_params_checks_wrong_type_int():
    """test parameter  checks wrong_type_intt"""
    model = SafeDummyClassifier()

    model.exactly_boo = 5
    model.at_least_5f = 5

    msg, disclosive = model.preliminary_check()
    assert disclosive is True
    correct_msg = notok_start

    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_least_5f",
        cur_val=model.at_least_5f,
        val="float",
    )
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="exactly_boo",
        cur_val=model.exactly_boo,
        val="str",
    )
    correct_msg += get_reporting_string(
        name="different_than_fixed_value",
        key="exactly_boo",
        cur_val=model.exactly_boo,
        val="boo",
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"


def test_saves():
    """checks that save functions as expected"""
    model = SafeDummyClassifier()
    x, y = get_data()
    model.fit(x, y)

    # no name provided
    model.save()

    # no extension
    model.save("mymodel")

    # unsupported extension
    model.save("mymodel.unsupported")

    # cannot be pickled
    model.square = lambda x: x * x #pylint: disable=attribute-defined-outside-init
    model.save("unpicklable.pkl")

    # cannot be joblibbed
    model.square = lambda x: x * x#pylint: disable=attribute-defined-outside-init
    model.save("unpicklable.sav")

    # cleanup
    for name in ("dummy.pkl", "dummy.sav", "unpicklable.pkl", "unpicklable.sav"):
        if os.path.exists(name) and os.path.isfile(name):
            os.remove(name)


def test_loads():
    """basic check that making, changing,saving,loading model works"""
    model = SafeDummyClassifier()
    x, y = get_data()
    model.fit(x, y)
    # change something in model
    model.newthing["myStringKey"] = "this_should_be_present"
    assert model.newthing["myStringKey"] == "this_should_be_present"

    # pkl
    model.save("dummy.pkl")
    with open("dummy.pkl", "rb") as file:
        model2 = pickle.load(file)
    assert model2.newthing["myStringKey"] == "this_should_be_present"

    # joblib
    model.save("dummy.sav")
    with open("dummy.sav", "rb") as file:
        model2 = joblib.load(file)
    assert model2.newthing["myStringKey"] == "this_should_be_present"


def test__apply_constraints():
    """tests constraints can be applied as expected"""

    # wrong type
    model = SafeDummyClassifier()
    model.at_least_5f = 3.9
    model.at_most_5i = 6.2
    model.exactly_boo = "five"

    assert model.at_least_5f == 3.9
    assert model.at_most_5i == 6.2
    assert model.exactly_boo == "five"

    msg, _ = model.preliminary_check(verbose=True, apply_constraints=True)

    assert model.at_least_5f == 5.0
    assert model.at_most_5i == 5
    assert model.exactly_boo == "boo"

    # checks that type changes happen correctly
    model = SafeDummyClassifier()
    model.at_least_5f = int(6.0)
    model.at_most_5i = float(4.2)
    model.exactly_boo = "five"

    assert model.at_least_5f == int(6)
    assert model.at_most_5i == 4.2
    assert model.exactly_boo == "five"

    msg, _ = model.preliminary_check(verbose=True, apply_constraints=True)
    assert model.at_least_5f == 6.0
    assert model.at_most_5i == 4
    assert model.exactly_boo == "boo"

    correct_msg = notok_start
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_least_5f",
        cur_val=int(6),
        val="float",
    )
    correct_msg += get_reporting_string(
        name="change_param_type", key="at_least_5f", val="float"
    )
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_most_5i",
        cur_val=4.2,
        val="int",
    )
    correct_msg += get_reporting_string(
        name="change_param_type", key="at_most_5i", val="int"
    )
    correct_msg += get_reporting_string(
        name="different_than_fixed_value",
        key="exactly_boo",
        cur_val="five",
        val="boo",
    )
    correct_msg += get_reporting_string(
        name="changed_param_equal", key="exactly_boo", val="boo"
    )

    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"

    model = SafeDummyClassifier()
    adict = {"a": "67543", "b": 6542}
    model.exactly_boo = adict
    msg, _ = model.preliminary_check(verbose=True, apply_constraints=True)
    correct_msg = notok_start
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="exactly_boo",
        cur_val={"a": "67543", "b": 6542},
        val="str",
    )
    correct_msg += get_reporting_string(
        name="not_implemented_for_change", key="exactly_boo", cur_val=adict, val="str"
    )
    correct_msg += get_reporting_string(
        name="different_than_fixed_value",
        key="exactly_boo",
        cur_val={"a": "67543", "b": 6542},
        val="boo",
    )
    correct_msg += get_reporting_string(
        name="changed_param_equal", key="exactly_boo", val="boo"
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
