"""Test Target class."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from aisdc.attacks.target import Target

RES_DIR = "save_test"


@pytest.mark.parametrize("get_target", [RandomForestClassifier()], indirect=True)
def test_target(get_target):
    """Test a randomly sampled 10% of the nursery dataset as a Target object."""
    target = get_target

    # [Researcher] Saves the target model and data
    target.save(RES_DIR)

    # [TRE] Loads the target model and data
    tre_target = Target()
    tre_target.load(RES_DIR)

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
