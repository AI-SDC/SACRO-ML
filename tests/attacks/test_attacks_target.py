"""Test Target class."""

from __future__ import annotations

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
    assert tre_target.n_samples == target.n_samples
    assert tre_target.n_samples_orig == target.n_samples_orig
    assert tre_target.n_features == target.n_features
    assert np.array_equal(tre_target.X_train, target.X_train)
    assert np.array_equal(tre_target.y_train, target.y_train)
    assert np.array_equal(tre_target.X_test, target.X_test)
    assert np.array_equal(tre_target.y_test, target.y_test)
    assert np.array_equal(tre_target.X_orig, target.X_orig)
    assert np.array_equal(tre_target.y_orig, target.y_orig)
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
    assert new_target.n_samples == target.n_samples
    assert new_target.n_samples_orig == target.n_samples_orig
