"""Pytest configuration for the test suite."""

import matplotlib as mpl

mpl.use("Agg")

import contextlib
import os
import shutil
import types
from datetime import date

import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.target import Target

np.random.seed(1)


def pytest_sessionfinish(session, exitstatus):
    """Remove test-created target directories after all tests complete."""
    for folder in ("target_pytorch", "target_sklearn"):
        with contextlib.suppress(Exception):
            shutil.rmtree(folder)


folders = [
    "RES",
    "dt.sav",
    "fit.tf",
    "fit2.tf",
    "outputs",
    "output_attribute",
    "output_lira",
    "output_pytorch",
    "output_sklearn",
    "output_worstcase",
    "outputs_factory",
    "outputs_lira",
    "outputs_multiple_attacks",
    "outputs_structural",
    "refit.tf",
    "release_dir",
    "save_test",
    "target",
    "target_factory",
    "target_sklearn",
    "target_pytorch",
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


def _generate_nursery_data(n_samples=2000, random_state=1):
    """Generate synthetic categorical data mimicking the nursery dataset.

    Uses make_classification to create a learnable classification problem,
    then discretises continuous features into categories matching the
    OpenML nursery dataset (data_id=26) structure so that one-hot encoding
    yields the same column layout.
    """
    feature_specs = [
        ("parents", ["great_pret", "pretentious", "usual"]),
        (
            "has_nurs",
            ["critical", "less_proper", "proper", "slightly_prob", "very_crit"],
        ),
        ("form", ["complete", "foster", "other", "others"]),
        ("children", ["1", "2", "3", "more"]),
        ("housing", ["convenient", "less_proper", "slightly_prob"]),
        ("finance", ["convenient", "inconv"]),
        ("social", ["non_prob", "slightly_prob", "very_recom"]),
        ("health", ["not_recom", "priority", "recommended"]),
    ]
    target_classes = ["not_recom", "priority", "spec_prior", "very_recom"]

    n_features = len(feature_specs)
    n_classes = len(target_classes)

    X_cont, y_int = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=random_state,
    )

    # Discretise each continuous feature into categories via percentile binning
    feature_names = []
    columns = []
    for i, (name, categories) in enumerate(feature_specs):
        feature_names.append(name)
        n_cats = len(categories)
        percentiles = np.linspace(0, 100, n_cats + 1)[1:-1]
        bins = np.percentile(X_cont[:, i], percentiles)
        bin_indices = np.digitize(X_cont[:, i], bins)
        columns.append(np.array([categories[idx] for idx in bin_indices]))

    data = np.column_stack(columns)
    target = np.array([target_classes[idx] for idx in y_int])

    return types.SimpleNamespace(data=data, target=target, feature_names=feature_names)


@pytest.fixture
def get_target(request) -> Target:
    """Return a target object with test data and fitted model.

    Uses synthetic categorical data generated with make_classification,
    discretised to match the nursery dataset structure.
    """
    model: sklearn.BaseEstimator = request.param

    nursery_data = _generate_nursery_data()
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
    n_features = np.shape(X_train_orig)[1]

    # fit model
    model.fit(X_train, y_train)

    # wrap
    target = Target(
        model=model,
        dataset_name="nursery",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train_orig,
        y_train_orig=y_train_orig,
        X_test_orig=X_test_orig,
        y_test_orig=y_test_orig,
    )
    for i in range(n_features - 1):
        target.add_feature(nursery_data.feature_names[i], indices[i], "onehot")
    target.add_feature("dummy", indices[n_features - 1], "float")
    return target


@pytest.fixture
def get_target_multiclass() -> Target:
    """
    Return a target object with multiclass test data and fitted model.

    Uses RF target model with continuous-valued features and
    label encoding for target class.
    """
    # make some data
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, shuffle=True, random_state=1
    )

    # fit model
    model = RandomForestClassifier(bootstrap=False)
    model.fit(X_train, y_train)

    # wrap
    target = Target(
        model=model,
        dataset_name="multiclass",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )
    for i in range(2):
        target.add_feature(f"V{i}", [i], "float")
    return target
