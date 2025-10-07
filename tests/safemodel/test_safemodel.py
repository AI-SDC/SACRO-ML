"""Test safemodel super class."""

from __future__ import annotations

import copy
import os
import pickle

import joblib
import numpy as np
import yaml
from sklearn import datasets
from sklearn.base import BaseEstimator

from sacroml.safemodel.reporting import get_reporting_string
from sacroml.safemodel.safemodel import SafeModel

notok_start = get_reporting_string(name="warn_possible_disclosure_risk")
ok_start = get_reporting_string(name="within_recommended_ranges")


class DummyClassifier(BaseEstimator):
    """Dummy Classifier that always returns predictions of zero."""

    def __init__(
        self, at_least_5f=5.0, at_most_5i=5, exactly_boo="boo", key_a=True, key_b=True
    ):
        """Instantiate a dummy classifier."""
        self.at_least_5f = at_least_5f
        self.at_most_5i = at_most_5i
        self.exactly_boo = exactly_boo
        self.key_a = key_a
        self.key_b = key_b

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit a dummy classifier."""

    def predict(self, x: np.ndarray):  # pragma: no cover
        """Predict all ones."""
        return np.ones(x.shape[0])


def get_data():
    """Return data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris["data"], dtype=np.float64)
    y = np.asarray(iris["target"], dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


class SafeDummyClassifier(SafeModel, DummyClassifier):
    """Privacy protected dummy classifier."""

    def __init__(self, **kwargs) -> None:
        """Create model and applies constraints to params."""
        SafeModel.__init__(self)
        self.basemodel_paramnames = (
            "at_least_5f",
            "at_most_5i",
            "exactly_boo",
            "key_a",
            "key_b",
        )
        the_kwds = {}
        for key, val in kwargs.items():
            if key in self.basemodel_paramnames:
                the_kwds[key] = val
        DummyClassifier.__init__(self, **the_kwds)
        self.model_type: str = "DummyClassifier"
        self.ignore_items = [
            "model_save_file",
            "basemodel_paramnames",
            "ignore_items",
            "timestamp",
        ]
        # create an item to test additional_checks()
        self.examine_seperately_items = ["newthing"]
        self.newthing = ["myStringKey", "aString", "myIntKey", "42"]

    def set_params(self, **kwargs):  # pragma: no cover
        """Set params."""
        for _, val in kwargs.items():
            self.key = val

    def fit(self, x: np.ndarray, y: np.ndarray):  # noqa: ARG002
        """Fit a safe dummy classifier."""
        self.saved_model = copy.deepcopy(self.__dict__)


def test_params_checks_ok():
    """Test parameter checks ok."""
    model = SafeDummyClassifier()

    correct_msg = ok_start
    msg, disclosive = model.preliminary_check()
    print(
        f"exactly_boo is {model.exactly_boo} with "
        f"type{type(model.exactly_boo).__name__}"
    )
    assert msg == ok_start, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert not disclosive


def test_params_checks_too_low():
    """Test parameter checks too low."""
    model = SafeDummyClassifier()

    model.at_least_5f = 4.0
    msg, disclosive = model.preliminary_check()
    assert disclosive
    correct_msg = notok_start + get_reporting_string(
        name="less_than_min_value",
        key="at_least_5f",
        cur_val=model.at_least_5f,
        val=5.0,
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"


def test_params_checks_too_high():
    """Test parameter checks too high."""
    model = SafeDummyClassifier()

    model.at_most_5i = 6
    msg, disclosive = model.preliminary_check()
    assert disclosive
    correct_msg = notok_start + get_reporting_string(
        name="greater_than_max_value", key="at_most_5i", cur_val=model.at_most_5i, val=5
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"


def test_params_checks_not_equal():
    """Test parameter checks not equal."""
    model = SafeDummyClassifier()

    model.exactly_boo = "foo"
    msg, disclosive = model.preliminary_check()
    assert disclosive
    correct_msg = notok_start + get_reporting_string(
        name="different_than_fixed_value",
        key="exactly_boo",
        cur_val=model.exactly_boo,
        val="boo",
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"


def test_params_checks_wrong_type_str():
    """Test parameter checks wrong type - strings given."""
    model = SafeDummyClassifier()

    model.at_least_5f = "five"
    model.at_most_5i = "five"

    msg, disclosive = model.preliminary_check()
    assert disclosive
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
    """Test parameter checks wrong_type_float."""
    model = SafeDummyClassifier()

    model.exactly_boo = 5
    model.at_most_5i = 5

    _, disclosive = model.preliminary_check()
    assert disclosive
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
    """Test parameter checks wrong_type_intt."""
    model = SafeDummyClassifier()

    model.exactly_boo = 5
    model.at_least_5f = 5

    msg, disclosive = model.preliminary_check()
    assert disclosive
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


def test_check_unknown_param():
    """Test handling of malformed json rule."""
    model = SafeDummyClassifier()
    _, _ = model.preliminary_check()
    odd_rule = {"operator": "unknown", "keyword": "exactly_boo", "value": "some_val"}
    msg, disclosive = model._SafeModel__check_model_param(odd_rule, False)
    correct_msg = get_reporting_string(
        name="unknown_operator",
        key=odd_rule["keyword"],
        val=odd_rule["value"],
        cur_val="boo",
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert not disclosive


def test_check_model_param_or():
    """Test or conditions in rules.json.

    The and condition is tested by the decision tree tests.
    """
    # ok
    model = SafeDummyClassifier()
    msg, disclosive = model.preliminary_check()
    correct_msg = ok_start
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert not disclosive

    part1 = get_reporting_string(
        name="different_than_fixed_value",
        key="key_a",
        cur_val=False,
        val="True",
    )
    part2 = get_reporting_string(
        name="different_than_fixed_value",
        key="key_b",
        cur_val=False,
        val="True",
    )

    # or - branch 1
    model = SafeDummyClassifier(key_a=False)
    correct_msg = ok_start + part1
    msg, disclosive = model.preliminary_check()
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert not disclosive

    # or  branch 2
    model = SafeDummyClassifier(key_b=False)
    correct_msg = ok_start + part2
    msg, disclosive = model.preliminary_check()
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert not disclosive

    # fail or
    model = SafeDummyClassifier(key_a=False, key_b=False)
    correct_msg = notok_start + part1 + part2
    msg, disclosive = model.preliminary_check()
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert disclosive


def test_saves():
    """Test that save functions as expected."""
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
    model.square = lambda x: x * x
    model.save("unpicklable.pkl")

    # cannot be joblibbed
    model.square = lambda x: x * x
    model.save("unpicklable.sav")


def test_loads():
    """Test that making, changing, saving, loading model works."""
    model = SafeDummyClassifier()
    x, y = get_data()
    model.fit(x, y)
    # change something in model
    model.exactly_boo = "this_should_be_present"
    assert model.exactly_boo == "this_should_be_present"

    # pkl
    model.save("dummy.pkl")
    with open("dummy.pkl", "rb") as file:
        model2 = pickle.load(file)
    assert model2.exactly_boo == "this_should_be_present"

    # joblib
    model.save("dummy.sav")
    with open("dummy.sav", "rb") as file:
        model2 = joblib.load(file)
    assert model2.exactly_boo == "this_should_be_present"


def test_apply_constraints():
    """Test constraints can be applied as expected."""
    # wrong type
    model = SafeDummyClassifier()
    model.at_least_5f = 3.9
    model.at_most_5i = 6.2
    model.exactly_boo = "five"

    assert model.at_least_5f == 3.9
    assert model.at_most_5i == 6.2
    assert model.exactly_boo == "five"

    msg, _ = model.preliminary_check(verbose=True, apply_constraints=True)

    assert model.at_least_5f == 5
    assert model.at_most_5i == 5
    assert model.exactly_boo == "boo"

    # checks that type changes happen correctly
    model = SafeDummyClassifier()
    model.at_least_5f = 6
    model.at_most_5i = 4.2
    model.exactly_boo = "five"

    assert model.at_least_5f == 6
    assert model.at_most_5i == 4.2
    assert model.exactly_boo == "five"

    msg, _ = model.preliminary_check(verbose=True, apply_constraints=True)
    assert model.at_least_5f == 6
    assert model.at_most_5i == 4
    assert model.exactly_boo == "boo"

    correct_msg = notok_start
    correct_msg += get_reporting_string(
        name="different_than_recommended_type",
        key="at_least_5f",
        cur_val=6,
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


def test_get_saved_model_exception():
    """Test the exception handling in get_current_and_saved_models()."""
    model = SafeDummyClassifier()
    # add generator which can't be pickled or copied
    model.a_generator = (i for i in [1, 2, 3])
    current, saved = model.get_current_and_saved_models()
    assert saved == {}  # since we haven;t called fit()
    assert "a_generator" not in current


def test_generic_additional_tests():
    """Test the class generic additional tests.

    For this purpose SafeDummyClassifier() defines
    self.newthing = {"myStringKey": "aString", "myIntKey": 42}.
    """
    model = SafeDummyClassifier()
    x, y = get_data()
    model.fit(x, y)

    # ok
    msg, disclosive = model.posthoc_check()
    correct_msg = ""
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert not disclosive

    # different lengths
    model.saved_model["newthing"] += ("extraA",)
    msg, disclosive = model.posthoc_check()
    print(
        f"contents of new then saved\n{model.newthing}\n{model.saved_model['newthing']}"
    )

    correct_msg = "Warning: different counts of values for parameter newthing.\n"
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert disclosive

    # different thing in list
    model.newthing += ("extraB",)
    msg, disclosive = model.posthoc_check()
    correct_msg = (
        "Warning: at least one non-matching value for parameter list newthing.\n"
    )
    print(
        f"contents of new then saved\n{model.newthing}\n{model.saved_model['newthing']}"
    )
    assert msg == correct_msg, f"Correct msg:\n{correct_msg}\nActual msg:\n{msg}\n"
    assert disclosive


def test_request_release_without_attacks():
    """Test request release works and check the content of the yaml file."""
    model = SafeDummyClassifier()
    x, y = get_data()
    model.fit(x, y)
    # give it k_anonymity
    model.k_anonymity = 5

    # no file provided, has k_anonymity

    res_dir = "RES"
    yaml_filename = os.path.normpath(os.path.join(f"{res_dir}", "target.yaml"))
    model_filename = os.path.normpath(os.path.join(f"{res_dir}", "model.pkl"))

    model.request_release(path=res_dir, ext="pkl")

    # check that pikle and the yaml files have been created
    assert os.path.isfile(model_filename)
    assert os.path.isfile(yaml_filename)

    # check the content of the yaml file
    with open(f"{yaml_filename}", encoding="utf-8") as fp:
        yaml_data = yaml.safe_load(fp)

        details, _ = model.preliminary_check(verbose=False)
        msg_post, _ = model.posthoc_check()
        k_anonymity = f"{model.k_anonymity}"
        recommendation = "Do not allow release"
        reason = details + msg_post

        assert {
            "researcher": model.researcher,
            "model_type": model.model_type,
            "details": details,
            "k_anonymity": k_anonymity,
            "recommendation": recommendation,
            "reason": reason,
            "timestamp": model.timestamp,
        } in yaml_data["safemodel"]
