"""
Example demonstrating the attribute inference attacks.

Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.attribute_inference_example

"""
import os

# ignore unused imports because it depends on whether data file is present
from sklearn.datasets import fetch_openml  # pylint:disable=unused-import
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (  # pylint:disable=unused-import
    LabelEncoder,
    OneHotEncoder,
)

from aisdc.attacks import attribute_attack  # pylint: disable = import-error
from aisdc.attacks.attribute_attack import (
    _get_bounds_risk,
    _infer_categorical,
    _unique_max,
)
from tests.test_attacks_via_safemodel import get_nursery_dataset

# pylint: disable = duplicate-code


def cleanup_file(name: str):
    """removes unwanted files or directory"""
    if os.path.exists(name) and os.path.isfile(name):  # h5
        os.remove(name)


def common_setup():
    """basic commands to get ready to test some code"""
    data = get_nursery_dataset()
    model = RandomForestClassifier(bootstrap=False)
    model.fit(data.x_train, data.y_train)
    attack_args = attribute_attack.AttributeAttackArgs(
        n_cpu=7, report_name="aia_report"
    )

    return data, model, attack_args


def test_attack_args():
    """tests methods in the attack_args class"""
    _, _, attack_args = common_setup()
    _ = attack_args.__str__()  # pylint:disable=unnecessary-dunder-call
    attack_args.set_param("newkey", True)
    thedict = attack_args.get_args()
    assert thedict["newkey"] is True


def test_unique_max():
    """tests the _unique_max helper function"""
    has_unique = (0.3, 0.5, 0.2)
    no_unique = (0.5, 0.5)
    assert _unique_max(has_unique, 0.0) is True
    assert _unique_max(has_unique, 0.6) is False
    assert _unique_max(no_unique, 0.0) is False


def test_categorical_via_modified_attack_brute_force():
    """test lots of functionality for categoricals
    using code from brute_force but without multiprocessing
    """
    data, model, _ = common_setup()

    threshold = 0
    feature = 0
    # make predictions
    _infer_categorical(model, data, feature, threshold)
    # or don't because threshold is too high
    threshold = 999
    _infer_categorical(model, data, feature, threshold)


def test_continuous_via_modified_bounds_risk():
    """tests a lot of the code for continuous variables
    via a copy of the _get_bounds_risk()
    modified not to use multiprocessing
    """
    data, model, _ = common_setup()
    _ = _get_bounds_risk(model, "dummy", 8, data.x_train, data.x_test)


# test below covers a lot of the plotting etc.
def test_AIA_on_nursery():
    """tests running AIA on the nursery data
    with an added continuous feature"""
    data, model, attack_args = common_setup()

    attack_obj = attribute_attack.AttributeAttack(attack_args)
    attack_obj.attack(data, model)

    output = attack_obj.make_report()
    output = output["attack_metrics"]


def test_cleanup():
    """tidies up any files created"""
    files_made = (
        "delete-me.json",
        "aia_example.json",
        "aia_example.pdf",
        "aia_report_cat_frac.png",
        "aia_report_cat_risk.png",
        "aia_report_quant_risk.png",
        "aia_report.pdf",
        "aia_report.json",
    )
    for fname in files_made:
        cleanup_file(fname)
