"""
Tests attacks called via safemodel classes
uses a subsampled nursery dataset as this tests more of the attack code
currently using random forests
"""
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks import attribute_attack, likelihood_attack, worst_case_attack
from aisdc.attacks.dataset import Data
from aisdc.safemodel.classifiers import SafeDecisionTreeClassifier

# pylint: disable=too-many-locals,bare-except,duplicate-code,unnecessary-dunder-call


def cleanup_file(name: str):
    """removes unwanted files or directory"""
    if os.path.exists(name) and os.path.isfile(name):  # h5
        os.remove(name)


def get_nursery_dataset() -> Data:
    """returns a randomly sampled 10+10% of
    the nursery data set as a Data object
    if needed fetches it from openml and saves. it

    """

    the_file = os.path.normpath("tests/datasets/nursery_as_dataset.pickle")
    save_locally = True
    need_download = True

    if Path(the_file).is_file():
        try:
            with open(the_file, "rb") as f:
                the_data = pickle.load(f)
            need_download = False
        except:  # pragma: no cover
            need_download = True

    if need_download:  # pragma: no cover
        nursery_data = fetch_openml(data_id=26, as_frame=True)
        x = np.asarray(nursery_data.data, dtype=str)
        y = np.asarray(nursery_data.target, dtype=str)
        # change labels from recommend to priority for the two odd cases
        num = len(y)
        for i in range(num):
            if y[i] == "recommend":
                y[i] = "priority"

        indices: list[list[int]] = [
            [0, 1, 2],  # parents
            [3, 4, 5, 6, 7],  # has_nurs
            [8, 9, 10, 11],  # form
            [12, 13, 14, 15],  # children
            [16, 17, 18],  # housing
            [19, 20],  # finance
            [21, 22, 23],  # social
            [24, 25, 26],  # health
            [27],  # dummy
        ]

        # [Researcher] Split into training and test sets
        # target model train / test split - these are strings
        (
            x_train_orig,
            x_test_orig,
            y_train_orig,
            y_test_orig,
        ) = train_test_split(
            x,
            y,
            test_size=0.05,
            stratify=y,
            shuffle=True,
        )

        # now resample the training data reduce number of examples
        _, x_train_orig, _, y_train_orig = train_test_split(
            x_train_orig,
            y_train_orig,
            test_size=0.05,
            stratify=y_train_orig,
            shuffle=True,
        )

        # [Researcher] Preprocess dataset
        # one-hot encoding of features and integer encoding of labels
        label_enc = LabelEncoder()
        feature_enc = OneHotEncoder()
        x_train = feature_enc.fit_transform(x_train_orig).toarray()
        y_train = label_enc.fit_transform(y_train_orig)
        x_test = feature_enc.transform(x_test_orig).toarray()
        y_test = label_enc.transform(y_test_orig)

        # add dummy continuous valued attribute from N(0.5,0.05)
        dummy_tr = 0.5 + 0.05 * np.random.randn(x_train.shape[0])
        dummy_te = 0.5 + 0.05 * np.random.randn(x_test.shape[0])
        dummy_all = np.hstack((dummy_tr, dummy_te)).reshape(-1, 1)
        dummy_tr = dummy_tr.reshape(-1, 1)
        dummy_te = dummy_te.reshape(-1, 1)

        x_train = np.hstack((x_train, dummy_tr))
        x_train_orig = np.hstack((x_train_orig, dummy_tr))
        x_test = np.hstack((x_test, dummy_te))
        x_test_orig = np.hstack((x_test_orig, dummy_te))
        xmore = np.concatenate((x_train_orig, x_test_orig))
        n_features = np.shape(x_train_orig)[1]

        # [TRE / Researcher] Wrap the data in a dataset object
        the_data = Data()
        the_data.name = "nursery"
        the_data.add_processed_data(x_train, y_train, x_test, y_test)
        for i in range(n_features - 1):
            the_data.add_feature(nursery_data.feature_names[i], indices[i], "onehot")
        the_data.add_feature("dummy", indices[n_features - 1], "float")
        the_data.add_raw_data(
            xmore, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig
        )

    if need_download and save_locally:  # pragma:no cover
        # make directory if needed then save
        # tested in local versions but don't want this in CI testing
        output_file = Path(the_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(the_file, "wb") as f:
            pickle.dump(the_data, f)

    return the_data


def test_attacks_via_request_release():
    """make vulnerable,hacked model then call request_release"""
    the_data = get_nursery_dataset()
    assert the_data.__str__() == "nursery"

    # build a broken model and hack it so lots of reasons to fail and be vulnerable
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=10, min_samples_leaf=1)
    model.fit(the_data.x_train, the_data.y_train)
    model.min_samples_leaf = 10
    model.request_release(filename="vulnerable_hacked.pkl", data_obj=the_data)
    files_made = (
        "vulnerable_hacked_attribute_res.json",
        "vulnerable_hacked_lira_res.json",
        "vulnerable_hacked_worst_case_res.json",
        "vulnerable_hacked_data.pickle",
        "vulnerable_hacked.pkl",
    )
    for fname in files_made:
        cleanup_file(fname)


def test_run_attack_lira():
    """calls the lira attack via safemodel"""
    the_data = get_nursery_dataset()
    assert the_data.__str__() == "nursery"

    # build a model
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5)
    model.fit(the_data.x_train, the_data.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive

    print(np.unique(the_data.y_test, return_counts=True))

    print(np.unique(model.predict(the_data.x_test), return_counts=True))

    fname = "delete-me"
    metadata = model.run_attack(the_data, "lira", fname)
    files_made = (
        "delete-me.json",
        "lira_example_report.json",
        "lira_example_report.pdf",
        "log_roc.png",
    )
    for fname in files_made:
        cleanup_file(fname)
    assert len(metadata) > 0  # something has been added


def test_run_attack_worstcase():
    """calls the worst case attack via safemodel"""
    the_data = get_nursery_dataset()
    assert the_data.__str__() == "nursery"

    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5)
    model.fit(the_data.x_train, the_data.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive

    fname = "delete-me"
    metadata = model.run_attack(the_data, "worst_case", fname)
    files_made = ("delete-me.json", "log_roc.png")
    for fname in files_made:
        cleanup_file(fname)
    assert len(metadata) > 0  # something has been added


def test_run_attack_attribute():
    """calls the attribute  attack via safemodel"""
    the_data = get_nursery_dataset()
    assert the_data.__str__() == "nursery"

    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5)
    model.fit(the_data.x_train, the_data.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive

    fname = "delete-me"
    metadata = model.run_attack(the_data, "attribute", fname)
    files_made = (
        "delete-me.json",
        "aia_example.json",
        "aia_example.pdf",
        "aia_report_cat_frac.png",
        "aia_report_cat_risk.png",
        "aia_report_quant_risk.png",
    )
    for fname in files_made:
        cleanup_file(fname)
    assert len(metadata) > 0  # something has been added


def test_attack_args():
    """tests the attack arguments class"""
    fname = "aia_example"
    attack_args = attribute_attack.AttributeAttackArgs(report_name=fname)
    attack_args.set_param("foo", "boo")
    assert attack_args.get_args()["foo"] == "boo"
    assert fname in attack_args.__str__()

    fname = "liraa"
    attack_args = likelihood_attack.LIRAAttackArgs(report_name=fname)
    attack_args.set_param("foo", "boo")
    assert attack_args.get_args()["foo"] == "boo"
    assert fname in attack_args.__str__()

    fname = "wca"
    attack_args = worst_case_attack.WorstCaseAttackArgs(report_name=fname)
    attack_args.set_param("foo", "boo")
    assert attack_args.get_args()["foo"] == "boo"
    assert fname in attack_args.__str__()


def test_run_attack_unknown():
    """calls an unknown attack via safemodel"""
    the_data = get_nursery_dataset()
    assert the_data.__str__() == "nursery"

    # build a model
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5)
    model.fit(the_data.x_train, the_data.y_train)

    fname = "delete-me"
    metadata = model.run_attack(the_data, "unknown", fname)
    files_made = ("delete-me.json",)
    for fname in files_made:
        cleanup_file(fname)
    assert metadata["outcome"] == "unrecognised attack type requested"
