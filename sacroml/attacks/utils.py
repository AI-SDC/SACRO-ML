"""Utility functions for attacks."""

import contextlib
import importlib
import logging
import os
import pickle
from typing import Any

import numpy as np
from scipy.stats import shapiro
from sklearn.base import BaseEstimator

from sacroml.attacks.model import Model
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS: float = 1e-16  # Used to avoid numerical issues


def check_and_update_dataset(target: Target) -> Target:
    """Check that it is safe to use class variables to index prediction arrays.

    This has two steps:
    1. Replacing the values in y_train with their position in
    target.model.classes (will normally result in no change)
    2. Removing from the test set any rows corresponding to classes that
    are not in the training set.
    """
    if (
        not isinstance(target.model.model, BaseEstimator)
        or target.y_train is None
        or target.y_test is None
        or target.X_train is None
        or target.X_test is None
    ):
        return target

    y_train_new = []
    classes = list(target.model.get_classes())
    for y in target.y_train:
        y_train_new.append(classes.index(y))
    target.y_train = np.array(y_train_new, int)
    logger.info(
        "new y_train has values and counts: %s",
        np.unique(target.y_train, return_counts=True),
    )
    ok_pos = []
    y_test_new = []
    for i, y in enumerate(target.y_test):
        if y in classes:
            ok_pos.append(i)
            y_test_new.append(classes.index(y))
    if len(y_test_new) != len(target.X_test):  # pragma: no cover
        target.X_test = target.X_test[ok_pos, :]
    target.y_test = np.array(y_test_new, int)
    logger.info(
        "new y_test has values and counts: %s",
        np.unique(target.y_test, return_counts=True),
    )
    return target


def train_shadow_models(
    shadow_clf: Model,
    combined_x_train: np.ndarray,
    combined_y_train: np.ndarray,
    n_train_rows: int,
    n_shadow_models: int,
    shadow_path: str,
) -> None:
    """Train and save shadow models.

    Reuses any saved models that are available.

    Parameters
    ----------
    shadow_clf : Model
        A classifier that will be trained to form the shadow models.
    combined_x_train : np.ndarray
        Array of combined train and test features.
    combined_y_train : np.ndarray
        Array of combined train and test labels.
    n_train_rows : int
        Number of samples in the training set.
    n_shadow_models : int
        Number of shadow models to train.
    shadow_path : str
        Location to save shadow models.
    """
    logger.info("Training shadow models")

    n_models_trained: int = get_n_shadow_models(shadow_path)
    if n_models_trained > 0:  # pragma: no cover
        logger.info("Found %d models previously trained", n_models_trained)

    n_combined: int = combined_x_train.shape[0]
    indices: np.ndarray = np.arange(0, n_combined, 1)

    for idx in range(n_models_trained, n_shadow_models):
        if idx % 10 == 0:
            logger.info("Trained %d models", idx)

        # Pick the indices to use for training this shadow model
        np.random.seed(idx)
        indices_train = np.random.choice(indices, n_train_rows, replace=False)
        indices_test = np.setdiff1d(indices, indices_train)

        # Fit the shadow model
        shadow_clf.set_params(random_state=idx)
        shadow_clf.fit(
            combined_x_train[indices_train, :],
            combined_y_train[indices_train],
        )

        # Save model and indices
        save_shadow_model(shadow_path, idx, shadow_clf, indices_train, indices_test)


def save_shadow_model(
    shadow_path: str,
    idx: int,
    model: Any,
    indices_train: np.ndarray,
    indices_test: np.ndarray,
) -> None:
    """Save a trained shadow model."""
    path: str = os.path.normpath(f"{shadow_path}/{idx}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(path, "indices_train.pkl"), "wb") as f:
        pickle.dump(indices_train, f)
    with open(os.path.join(path, "indices_test.pkl"), "wb") as f:
        pickle.dump(indices_test, f)


def get_shadow_model(shadow_path: str, idx: int) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return a shadow model and indices previously saved."""
    path: str = os.path.normpath(f"{shadow_path}/{idx}")
    with open(os.path.join(path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(path, "indices_train.pkl"), "rb") as f:
        indices_train = pickle.load(f)
    with open(os.path.join(path, "indices_train.pkl"), "rb") as f:
        indices_test = pickle.load(f)
    return model, indices_train, indices_test


def get_n_shadow_models(shadow_path: str) -> int:
    """Return the number shadow models saved."""
    count: int = 0
    for item in os.listdir(shadow_path):  # pragma: no cover
        item_path = os.path.join(shadow_path, item)
        if os.path.isdir(item_path):
            count += 1
    return count


def get_p_normal(samples: np.ndarray) -> float:
    """Test whether a set of samples is normally distributed."""
    p_normal: float = np.nan
    if np.nanvar(samples) > EPS:
        with contextlib.suppress(ValueError):
            _, p_normal = shapiro(samples)
    return p_normal


def logit(p: float) -> float:
    """Return standard logit.

    Parameters
    ----------
    p : float
        value to evaluate logit at.

    Returns
    -------
    float
        logit(p)

    Notes
    -----
    If `p` is close to 0 or 1, evaluating the log will result in numerical
    instabilities. This code thresholds `p` at `EPS` and `1 - EPS` where `EPS`
    defaults at 1e-16.
    """
    p = min(p, 1 - EPS)
    p = max(p, EPS)
    return np.log(p / (1 - p))


def get_class_by_name(class_path: str):
    """Return a class given its name."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
