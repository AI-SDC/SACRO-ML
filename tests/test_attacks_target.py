"""Code to test the file attacks/target.py."""

from __future__ import annotations

import builtins
import io
import os

import numpy as np
import pytest
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks.target import Target

# pylint: disable=redefined-outer-name


def patch_open(open_func, files):
    """Helper function for cleaning up created files."""

    def open_patched(  # pylint: disable=too-many-arguments
        path,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        if "w" in mode and not os.path.isfile(path):
            files.append(path)
        return open_func(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    return open_patched


@pytest.fixture
def cleanup_files(monkeypatch):
    """Automatically remove created files."""
    files = []
    monkeypatch.setattr(builtins, "open", patch_open(builtins.open, files))
    monkeypatch.setattr(io, "open", patch_open(io.open, files))
    yield
    for file in files:
        try:
            os.remove(file)
        except FileNotFoundError:  # pragma: no cover
            pass


def test_target(cleanup_files):  # pylint:disable=too-many-locals,unused-argument
    """
    Returns a randomly sampled 10+10% of
    the nursery data set as a Target object
    if needed fetches it from openml and saves. it.
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

    # add dummy continuous valued attribute
    dummy_tr = np.random.rand(x_train.shape[0], 1)
    dummy_te = np.random.rand(x_test.shape[0], 1)
    x_train = np.hstack((x_train, dummy_tr))
    x_train_orig = np.hstack((x_train_orig, dummy_tr))
    x_test = np.hstack((x_test, dummy_te))
    x_test_orig = np.hstack((x_test_orig, dummy_te))

    n_features = np.shape(x_train_orig)[1]

    # [Researcher] Wrap the data in a Target object
    target = Target(model=RandomForestClassifier(n_estimators=5, max_depth=5))
    target.name = "nursery"
    target.add_processed_data(x_train, y_train, x_test, y_test)
    target.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features - 1):
        target.add_feature(nursery_data.feature_names[i], indices[i], "onehot")
    target.add_feature("dummy", indices[n_features - 1], "float")

    # [Researcher] Saves the target model and data
    target.save("save_test")

    # [TRE] Loads the target model and data
    tre_target = Target()
    tre_target.load("save_test")

    assert tre_target.model.get_params() == target.model.get_params()
    assert tre_target.name == target.name
    assert tre_target.features == target.features
    assert tre_target.n_samples == target.n_samples
    assert tre_target.n_samples_orig == target.n_samples_orig
    assert tre_target.n_features == target.n_features
    assert np.array_equal(tre_target.x_train, target.x_train)
    assert np.array_equal(tre_target.y_train, target.y_train)
    assert np.array_equal(tre_target.x_test, target.x_test)
    assert np.array_equal(tre_target.y_test, target.y_test)
    assert np.array_equal(tre_target.x_orig, target.x_orig)
    assert np.array_equal(tre_target.y_orig, target.y_orig)
    assert np.array_equal(tre_target.x_train_orig, target.x_train_orig)
    assert np.array_equal(tre_target.y_train_orig, target.y_train_orig)
    assert np.array_equal(tre_target.x_test_orig, target.x_test_orig)
    assert np.array_equal(tre_target.y_test_orig, target.y_test_orig)
