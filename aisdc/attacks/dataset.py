"""dataset.py - class to represent a dataset"""

from __future__ import annotations

import numpy as np


class Data:  # pylint: disable=too-many-instance-attributes
    """Stripped down Data class"""

    def __init__(self) -> None:
        self.name: str = ""
        self.n_samples: int = 0
        self.x_train: np.ndarray
        self.y_train: np.ndarray
        self.x_test: np.ndarray
        self.y_test: np.ndarray

        self.features: dict = {}
        self.n_features: int = 0
        self.x_orig: np.ndarray
        self.y_orig: np.ndarray
        self.x_train_orig: np.ndarray
        self.y_train_orig: np.ndarray
        self.x_test_orig: np.ndarray
        self.y_test_orig: np.ndarray
        self.n_samples_orig: int = 0

    def add_processed_data(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Add a processed and split dataset"""
        self.x_train = x_train
        self.y_train = np.array(y_train, int)
        self.x_test = x_test
        self.y_test = np.array(y_test, int)
        self.n_samples = len(x_train) + len(y_train)

    def add_feature(self, name: str, indices: list[int], encoding: str) -> None:
        """Add a feature description to the data dictionary."""
        index: int = len(self.features)
        self.features[index] = {
            "name": name,
            "indices": indices,
            "encoding": encoding,
        }
        self.n_features = len(self.features)

    def add_raw_data(  # pylint: disable=too-many-arguments
        self,
        x_orig: np.ndarray,
        y_orig: np.ndarray,
        x_train_orig: np.ndarray,
        y_train_orig: np.ndarray,
        x_test_orig: np.ndarray,
        y_test_orig: np.ndarray,
    ) -> None:
        """Add original unprocessed dataset"""
        self.x_orig = x_orig
        self.y_orig = y_orig
        self.x_train_orig = x_train_orig
        self.y_train_orig = y_train_orig
        self.x_test_orig = x_test_orig
        self.y_test_orig = y_test_orig
        self.n_samples_orig = len(x_orig)

    def __str__(self):
        return self.name
