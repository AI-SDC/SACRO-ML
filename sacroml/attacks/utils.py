"""Utility functions for attacks."""

import contextlib
import importlib
import logging
import os
import pickle
from typing import Any

import numpy as np
from scipy.stats import shapiro

from sacroml.attacks.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS: float = 1e-16  # Used to avoid numerical issues


def train_shadow_models(
    shadow_clf: Model,
    combined_x_train: np.ndarray,
    combined_y_train: np.ndarray,
    n_train_rows: int,
    n_shadow_models: int,
    output_dir: str,
) -> None:
    """Train and save shadow models.

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
    output_dir : str
        Location to save shadow models.
    """
    logger.info("Training shadow models")

    n_combined: int = combined_x_train.shape[0]
    indices: np.ndarray = np.arange(0, n_combined, 1)

    for idx in range(n_shadow_models):
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
        save_shadow_model(output_dir, idx, shadow_clf, indices_train, indices_test)


def save_shadow_model(
    output_dir: str,
    idx: int,
    model: Any,
    indices_train: np.ndarray,
    indices_test: np.ndarray,
) -> None:
    """Save a trained shadow model."""
    path: str = os.path.normpath(f"{output_dir}/shadow_models/{idx}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(path, "indices_train.pkl"), "wb") as f:
        pickle.dump(indices_train, f)
    with open(os.path.join(path, "indices_test.pkl"), "wb") as f:
        pickle.dump(indices_test, f)


def get_shadow_model(output_dir: str, idx: int) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return a shadow model and indices previously saved."""
    path: str = os.path.normpath(f"{output_dir}/shadow_models/{idx}")
    with open(os.path.join(path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(path, "indices_train.pkl"), "rb") as f:
        indices_train = pickle.load(f)
    with open(os.path.join(path, "indices_train.pkl"), "rb") as f:
        indices_test = pickle.load(f)
    return model, indices_train, indices_test


def get_n_shadow_models(output_dir: str) -> int:
    """Return the number shadow models saved."""
    path: str = os.path.normpath(f"{output_dir}/shadow_models")
    count: int = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
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


def get_class_by_name(class_path: str):
    """Return a class given its name."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
