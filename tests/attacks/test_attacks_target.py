"""Test Target class."""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from sacroml.attacks.target import Target

RES_DIR = "save_test"


@pytest.mark.parametrize("get_target", [RandomForestClassifier()], indirect=True)
def test_target(get_target):
    """Test Target object creation, saving, and loading."""
    # create target
    target = get_target

    # save target
    target.save(RES_DIR)

    # test loading target
    tre_target = Target()
    tre_target.load(RES_DIR)

    assert tre_target.model.get_params() == target.model.get_params()
    assert tre_target.dataset_name == target.dataset_name
    assert tre_target.features == target.features
    assert tre_target.n_features == target.n_features
    assert np.array_equal(tre_target.X_train, target.X_train)
    assert np.array_equal(tre_target.y_train, target.y_train)
    assert np.array_equal(tre_target.X_test, target.X_test)
    assert np.array_equal(tre_target.y_test, target.y_test)
    assert np.array_equal(tre_target.X_train_orig, target.X_train_orig)
    assert np.array_equal(tre_target.y_train_orig, target.y_train_orig)
    assert np.array_equal(tre_target.X_test_orig, target.X_test_orig)
    assert np.array_equal(tre_target.y_test_orig, target.y_test_orig)

    # test creating target with data added via constructor
    new_target = Target(
        model=target.model,
        X_train=target.X_train,
        y_train=target.y_train,
        X_test=target.X_test,
        y_test=target.y_test,
        X_train_orig=target.X_train_orig,
        y_train_orig=target.y_train_orig,
        X_test_orig=target.X_test_orig,
        y_test_orig=target.y_test_orig,
    )
    assert new_target


def test_save_skips_data_arrays_when_dataset_module_set(tmp_path):
    """Test data arrays are not serialised when a dataset module path is set.

    After a round-trip load() followed by
    save(), data array pkl files must not be written when a dataset module
    is provided.
    """
    module_path = tmp_path / "mock_dataset.py"
    module_path.write_text(
        "import numpy as np\n"
        "from sacroml.attacks.data import SklearnDataHandler\n\n\n"
        "class MockDataset(SklearnDataHandler):\n"
        "    def __init__(self):\n"
        "        pass\n\n"
        "    def __len__(self):\n"
        "        return 10\n\n"
        "    def get_data(self):\n"
        "        return np.zeros((10, 2)), np.zeros(10)\n\n"
        "    def get_raw_data(self):\n"
        "        return None\n\n"
        "    def get_subset(self, X, y, indices):\n"
        "        idx = list(indices)\n"
        "        return X[idx], y[idx]\n",
        encoding="utf-8",
    )

    X_train = np.zeros((6, 2))
    y_train = np.zeros(6)
    X_test = np.zeros((4, 2))
    y_test = np.zeros(4)
    model = RandomForestClassifier(n_estimators=1, random_state=0)
    model.fit(X_train, y_train)

    # Simulate the state after load() calls load_sklearn_dataset():
    # dataset_module_path is set AND data arrays are populated in memory.
    save_dir = str(tmp_path / "target_dataset_module")
    target = Target(
        model=model,
        dataset_module_path=str(module_path),
        dataset_name="MockDataset",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    target.save(save_dir)

    for arr_name in ["X_train", "y_train", "X_test", "y_test"]:
        assert not os.path.exists(os.path.join(save_dir, f"{arr_name}.pkl"))
