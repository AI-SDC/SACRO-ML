"""Synthetic dataset handler."""

from collections.abc import Sequence

import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from sacroml.attacks.data import PyTorchDataHandler

random_state = 2


class Synthetic(PyTorchDataHandler):
    """Synthetic dataset handler."""

    def __init__(self) -> None:
        """Create synthetic data."""
        self.X, self.y = make_classification(
            n_samples=50,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=random_state,
        )

        # Store raw unprocessed data
        X_tensor = torch.FloatTensor(self.X)
        y_tensor = torch.LongTensor(self.y)
        self.raw_dataset = TensorDataset(X_tensor, y_tensor)

        # Preprocess
        self.feature_encoder = StandardScaler()
        X_transformed = self.feature_encoder.fit_transform(self.X)

        # Store processed data
        X_tensor = torch.FloatTensor(X_transformed)
        y_tensor = torch.LongTensor(self.y)
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
            indices, test_size=0.2, stratify=self.y, random_state=random_state
        )
        return train, test
