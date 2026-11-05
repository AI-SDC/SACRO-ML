"""Example regression dataset handler with synthetic data.

PyTorch datasets must implement `sacroml.attacks.data.PyTorchDataHandler`.
"""

from collections.abc import Sequence

import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from sacroml.attacks.data import PyTorchDataHandler

random_state = 2


class Synthetic(PyTorchDataHandler):
    """Synthetic dataset handler."""

    def __init__(self) -> None:
        """Create synthetic regression data."""
        self.X, self.y = make_regression(
            n_samples=200,
            n_features=4,
            n_informative=4,
            n_targets=4,
            noise=0,
            random_state=random_state,
        )

        # Store raw unprocessed data
        X_tensor = torch.FloatTensor(self.X)
        y_tensor = torch.FloatTensor(self.y)
        self.raw_dataset = TensorDataset(X_tensor, y_tensor)

        # Preprocess
        self.feature_encoder = StandardScaler()
        X_transformed = self.feature_encoder.fit_transform(self.X)

        self.label_encoder = StandardScaler()
        y_transformed = self.label_encoder.fit_transform(self.y)

        # Store processed data
        X_tensor = torch.FloatTensor(X_transformed)
        y_tensor = torch.FloatTensor(y_transformed)
        self.dataset = TensorDataset(X_tensor, y_tensor)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def get_raw_dataset(self) -> Dataset | None:
        """Return a raw unprocessed dataset."""
        return self.raw_dataset

    def get_dataset(self) -> Dataset:
        """Return a preprocessed dataset."""
        return self.dataset

    def get_dataloader(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> DataLoader:
        """Return a data loader with a requested subset of samples."""
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

    def get_train_test_indices(self) -> tuple[Sequence[int], Sequence[int]]:
        """Return train and test set indices."""
        indices = range(len(self))
        train, test = train_test_split(
            indices, test_size=0.2, random_state=random_state
        )
        return train, test
