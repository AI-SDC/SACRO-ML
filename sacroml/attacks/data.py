"""Abstract PyTorch dataset handler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch.utils.data import DataLoader, Dataset


class DataHandler(ABC):
    """PyTorch dataset handling interface."""

    @abstractmethod
    def __init__(self) -> None:
        """Instantiate a dataset handler."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset."""

    @abstractmethod
    def get_dataset(self) -> Dataset:
        """Return a processed dataset.

        Returns
        -------
        Dataset
            A (processed) PyTorch dataset.
        """

    @abstractmethod
    def get_dataloader(
        self, dataset: Dataset, indices: Sequence[int], batch_size: int = 32
    ) -> DataLoader:
        """Return a data loader with a requested subset of samples.

        Parameters
        ----------
        dataset : Dataset
            A (processed) PyTorch dataset.
        indices : Sequence[int]
            The indices to load from the dataset.
        batch_size : int
            The batch_size to sample the dataset.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader.
        """

    @abstractmethod
    def get_train_test_indices(self) -> tuple[Sequence[int], Sequence[int]]:
        """Return train and test set indices.

        Returns
        -------
        Sequence[int]
            Indices of the training samples.
        Sequence[int]
            Indices of the test samples.
        """
