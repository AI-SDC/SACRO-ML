"""Synthetic dataset handler."""

import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class Synthetic:
    """Synthetic dataset handler."""

    def __init__(self) -> None:
        """Create synthetic data."""
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=50,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=2,
        )

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, shuffle=True, random_state=2
        )

        # Scale the features
        scaler = StandardScaler()
        X_train_transformed = scaler.fit_transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train_transformed)
        self.X_test = torch.FloatTensor(X_test_transformed)
        self.y_train = torch.LongTensor(y_train)
        self.y_test = torch.LongTensor(y_test)

        self.trainset = TensorDataset(self.X_train, self.y_train)
        self.testset = TensorDataset(self.X_test, self.y_test)

    def get_train_loader(self, batch_size: int = 32) -> DataLoader:
        """Return a training data loader."""
        return DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

    def get_test_loader(self, batch_size: int = 32) -> DataLoader:
        """Return a testing data loader."""
        return DataLoader(self.testset, batch_size=batch_size, shuffle=False)
