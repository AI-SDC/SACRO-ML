"""Common utility functions for testing."""

import contextlib
import os
import shutil
from datetime import date

import numpy as np
import pytest
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.target import Target

np.random.seed(1)

folders = [
    "RES",
    "dt.sav",
    "fit.tf",
    "fit2.tf",
    "keras_save.tf",
    "outputs",
    "output_attribute",
    "output_lira",
    "output_worstcase",
    "outputs_factory",
    "outputs_lira",
    "outputs_multiple_attacks",
    "outputs_structural",
    "refit.tf",
    "release_dir",
    "safekeras.tf",
    "save_test",
    "target",
    "target_factory",
    "test_lira_target",
    "test_output_lira",
    "test_output_sa",
    "test_output_worstcase",
    "test_worstcase_target",
    "tests/test_aia_target",
    "tests/test_multiple_target",
    "tfsaves",
    "training_artefacts",
]

files = [
    "1024-WorstCase.png",
    "2048-WorstCase.png",
    "attack.txt",
    "attack.yaml",
    "dummy.pkl",
    "dummy.sav",
    "dummy_model.txt",
    "example_filename.json",
    "filename_should_be_changed.txt",
    "filename_to_rewrite.json",
    "results.txt",
    "rf_test.pkl",
    "rf_test.sav",
    "safekeras.h5",
    "target.json",
    "test.json",
    "test_data.csv",
    "test_preds.csv",
    "train_data.csv",
    "train_preds.csv",
    "unpicklable.pkl",
    "unpicklable.sav",
    "ypred_test.csv",
    "ypred_train.csv",
]


@pytest.fixture(name="cleanup", autouse=True)
def _cleanup():
    """Remove created files and directories."""
    yield

    for folder in folders:
        with contextlib.suppress(Exception):
            shutil.rmtree(folder)

    files.append(  # from attack_report_formater.py
        "ATTACK_RESULTS" + str(date.today().strftime("%d_%m_%Y")) + ".json"
    )

    for file in files:
        with contextlib.suppress(Exception):
            os.remove(file)


@pytest.fixture
def get_target(request) -> Target:  # pylint: disable=too-many-locals
    """Return a target object with test data and fitted model.

    Uses a randomly sampled 10+10% of the nursery data set.
    """
    model: sklearn.BaseEstimator = request.param

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
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        x, y, test_size=0.05, stratify=y, shuffle=True, random_state=1
    )

    # now resample the training data reduce number of examples
    _, X_train_orig, _, y_train_orig = train_test_split(
        X_train_orig,
        y_train_orig,
        test_size=0.05,
        stratify=y_train_orig,
        shuffle=True,
        random_state=1,
    )

    # [Researcher] Preprocess dataset
    # one-hot encoding of features and integer encoding of labels
    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    X_train = feature_enc.fit_transform(X_train_orig).toarray()
    y_train = label_enc.fit_transform(y_train_orig)
    X_test = feature_enc.transform(X_test_orig).toarray()
    y_test = label_enc.transform(y_test_orig)

    # add dummy continuous valued attribute from N(0.5,0.05)
    dummy_tr = 0.5 + 0.05 * np.random.randn(X_train.shape[0])
    dummy_te = 0.5 + 0.05 * np.random.randn(X_test.shape[0])
    dummy_tr = dummy_tr.reshape(-1, 1)
    dummy_te = dummy_te.reshape(-1, 1)

    X_train = np.hstack((X_train, dummy_tr))
    X_train_orig = np.hstack((X_train_orig, dummy_tr))
    X_test = np.hstack((X_test, dummy_te))
    X_test_orig = np.hstack((X_test_orig, dummy_te))
    xmore = np.concatenate((X_train_orig, X_test_orig))
    n_features = np.shape(X_train_orig)[1]

    # fit model
    model.fit(X_train, y_train)

    # wrap
    target = Target(model=model)
    target.dataset_name = "nursery"
    target.add_processed_data(X_train, y_train, X_test, y_test)
    for i in range(n_features - 1):
        target.add_feature(nursery_data.feature_names[i], indices[i], "onehot")
    target.add_feature("dummy", indices[n_features - 1], "float")
    target.add_raw_data(xmore, y, X_train_orig, y_train_orig, X_test_orig, y_test_orig)
    return target
