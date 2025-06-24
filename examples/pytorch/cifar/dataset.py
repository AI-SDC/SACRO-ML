"""CIFAR10 dataset handling."""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Cifar10:
    """CIFAR10 dataset handler."""

    def __init__(self) -> None:
        """Fetch and process CIFAR10."""
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def get_train_loader(
        self, batch_size: int = 32, num_workers: int = 0
    ) -> DataLoader:
        """Return a training data loader."""
        return torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    def get_test_loader(self, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        """Return a testing data loader."""
        return torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def dataloader_to_numpy(self, dataloader):
        """Convert DataLoader to numpy arrays."""
        all_inputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                all_inputs.append(inputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        X = np.concatenate(all_inputs, axis=0)
        y = np.concatenate(all_labels, axis=0)

        return X, y
