"""Nursery dataset handler for scikit-learn models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.data import SklearnDataHandler

random_state = 1


class Nursery(SklearnDataHandler):  # pylint: disable=too-many-instance-attributes
    """Nursery dataset handler."""

    def __init__(self) -> None:
        """Fetch and process the nursery dataset."""
        # Get original dataset
        nursery_data = fetch_openml(data_id=26, as_frame=True)
        self.X_orig = np.asarray(nursery_data.data, dtype=str)
        self.y_orig = np.asarray(nursery_data.target, dtype=str)

        # Use only a sample to speed testing
        self.X_orig, _, self.y_orig, _ = train_test_split(
            self.X_orig,
            self.y_orig,
            test_size=0.95,
            stratify=self.y_orig,
            random_state=random_state,
        )

        # Process dataset
        self.label_enc = LabelEncoder()
        self.feature_enc = OneHotEncoder()
        self.X = self.feature_enc.fit_transform(self.X_orig).toarray()
        self.y = self.label_enc.fit_transform(self.y_orig)

        # Feature encoding information (only required for attribute inference)
        self.feature_indices = [
            [0, 1, 2],  # parents
            [3, 4, 5, 6, 7],  # has_nurs
            [8, 9, 10, 11],  # form
            [12, 13, 14, 15],  # children
            [16, 17, 18],  # housing
            [19, 20],  # finance
            [21, 22, 23],  # social
            [24, 25, 26],  # health
        ]
        self.feature_names = nursery_data.feature_names

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.X)

    def get_raw_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the original raw data arrays."""
        return self.X_orig, self.y_orig

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the processed data arrays."""
        return self.X, self.y

    def get_subset(
        self, X: np.ndarray, y: np.ndarray, indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a subset of the data."""
        return X[indices], y[indices]

    def get_train_test_indices(self) -> tuple[Sequence[int], Sequence[int]]:
        """Return train and test set indices."""
        indices = range(len(self))
        train, test = train_test_split(
            indices, test_size=0.5, stratify=self.y, random_state=random_state
        )
        return train, test
