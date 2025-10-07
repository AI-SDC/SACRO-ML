"""Example dataset handler for CIFAR10.

PyTorch datasets must implement `sacroml.attacks.data.PyTorchDataHandler`.
"""

from collections.abc import Sequence

from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from sacroml.attacks.data import PyTorchDataHandler


class Cifar10(PyTorchDataHandler):
    """CIFAR10 dataset handler."""

    def __init__(self) -> None:
        """Fetch and process CIFAR10."""
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_set = CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )

        test_set = CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        self.dataset = ConcatDataset([train_set, test_set])

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

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def get_raw_dataset(self) -> Dataset | None:
        """Return a raw unprocessed dataset."""
        # Raw data only required for attribute inference
        return None

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
        train = range(50000)
        test = range(50000, 60000)
        return train, test
