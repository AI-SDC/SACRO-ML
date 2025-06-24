"""Pytorch dataset for testing."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from sacroml.attacks.model_pytorch import numpy_to_dataloader


class SyntheticData:  # pylint: disable=too-many-instance-attributes
    """Synthetic Dataset."""

    def __init__(self, x_dim: int, y_dim: int, random_state: int | None = None) -> None:
        """Create some test data."""
        self.X_orig, self.y_orig = make_classification(
            n_samples=50,
            n_features=x_dim,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=y_dim,
            n_clusters_per_class=1,
            random_state=random_state,
        )
        self.X_orig = np.asarray(self.X_orig)
        self.y_orig = np.asarray(self.y_orig)

        # Preprocess
        input_encoder = StandardScaler()
        X = input_encoder.fit_transform(self.X_orig)
        y = self.y_orig  # leave as labels

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, shuffle=True, random_state=random_state
        )

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)

    def get_train_loader(self) -> DataLoader:
        """Return a training data loader."""
        return numpy_to_dataloader(self.X_train, self.y_train)

    def get_test_loader(self) -> DataLoader:
        """Return a testing data loader."""
        return numpy_to_dataloader(self.X_test, self.y_test)
