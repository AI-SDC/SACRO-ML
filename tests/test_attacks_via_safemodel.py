"""
Tests attacks called via safemodel classes
uses a subsampled nursery dataset as this tests more of the attack code
currently using random forests.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks import attribute_attack, likelihood_attack, worst_case_attack
from aisdc.attacks.target import Target
from aisdc.safemodel.classifiers import SafeDecisionTreeClassifier

# pylint: disable=too-many-locals,bare-except,duplicate-code,unnecessary-dunder-call

RES_DIR = "RES"


def clean():
    """Removes unwanted results."""
    if os.path.exists(RES_DIR):
        shutil.rmtree(RES_DIR)


def get_target(model: sklearn.base.BaseEstimator) -> Target:
    """Wrap the model and data in a Target object.
    Uses A randomly sampled 10+10% of the nursery data set.
    """

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

    # wrap
    target = Target(model=model)
    target.name = "nursery"
    target.add_processed_data(x_train, y_train, x_test, y_test)
    for i in range(n_features - 1):
        target.add_feature(nursery_data.feature_names[i], indices[i], "onehot")
    target.add_feature("dummy", indices[n_features - 1], "float")
    target.add_raw_data(xmore, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    return target


def test_attacks_via_request_release():
    """Make vulnerable,hacked model then call request_release."""
    # build a broken model and hack it so lots of reasons to fail and be vulnerable
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=10, min_samples_leaf=1)
    target = get_target(model)
    assert target.__str__() == "nursery"
    model.fit(target.x_train, target.y_train)
    model.min_samples_leaf = 10
    model.request_release(path=RES_DIR, ext="pkl", target=target)
    clean()


def test_run_attack_lira():
    """Calls the lira attack via safemodel."""
    # build a model
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=10)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive

    print(np.unique(target.y_test, return_counts=True))
    print(np.unique(model.predict(target.x_test), return_counts=True))
    metadata = model.run_attack(target, "lira", RES_DIR, "lira_res")
    clean()
    assert len(metadata) > 0  # something has been added


def test_run_attack_worstcase():
    """Calls the worst case attack via safemodel."""
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=20)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive
    metadata = model.run_attack(target, "worst_case", RES_DIR, "wc_res")
    clean()
    assert len(metadata) > 0  # something has been added


def test_run_attack_attribute():
    """Calls the attribute  attack via safemodel."""
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=10)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive
    metadata = model.run_attack(target, "attribute", RES_DIR, "attr_res")
    clean()
    assert len(metadata) > 0  # something has been added


def test_attack_args():
    """Tests the attack arguments class."""
    fname = "aia_example"
    attack_obj = attribute_attack.AttributeAttack(
        output_dir="output_attribute", report_name=fname
    )
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"
    assert fname == attack_obj.report_name

    fname = "liraa"
    attack_obj = likelihood_attack.LIRAAttack(
        output_dir="output_lira", report_name=fname
    )
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"
    assert fname == attack_obj.report_name

    fname = "wca"
    attack_obj = worst_case_attack.WorstCaseAttack(
        output_dir="output_worstcase", report_name=fname
    )
    attack_obj.__dict__["foo"] = "boo"
    assert attack_obj.__dict__["foo"] == "boo"
    assert fname == attack_obj.report_name
    shutil.rmtree("output_attribute")
    shutil.rmtree("output_lira")
    shutil.rmtree("output_worstcase")


def test_run_attack_unknown():
    """Calls an unknown attack via safemodel."""
    # build a model
    model = SafeDecisionTreeClassifier(random_state=1, max_depth=5)
    target = get_target(model)
    model.fit(target.x_train, target.y_train)
    metadata = model.run_attack(target, "unknown", RES_DIR, "unk")
    clean()
    assert metadata["outcome"] == "unrecognised attack type requested"
