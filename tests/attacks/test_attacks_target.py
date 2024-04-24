"""Code to test the file attacks/target.py."""

from __future__ import annotations

import builtins
import io
import os

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from aisdc.attacks.target import Target

from ..common import clean, get_target

# pylint: disable=redefined-outer-name

RES_DIR = "save_test"


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


def test_target(cleanup_files):  # pylint:disable=unused-argument
    """
    Returns a randomly sampled 10+10% of
    the nursery data set as a Target object
    if needed fetches it from openml and saves. it.
    """
    target = get_target(model=RandomForestClassifier(n_estimators=5, max_depth=5))

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

    clean(RES_DIR)
