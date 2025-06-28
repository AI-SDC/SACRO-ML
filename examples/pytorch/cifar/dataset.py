"""CIFAR10 dataset handler."""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


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

        self.trainset = CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )

        self.testset = CIFAR10(
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

    def get_train_loader(self, batch_size: int = 32) -> DataLoader:
        """Return a training data loader."""
        return DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

    def get_test_loader(self, batch_size: int = 32) -> DataLoader:
        """Return a testing data loader."""
        return DataLoader(self.testset, batch_size=batch_size, shuffle=False)
