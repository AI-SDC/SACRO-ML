"""Abstract data handler supporting both PyTorch and scikit-learn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset


class BaseDataHandler(ABC):
    """Base data handling interface."""

    @abstractmethod
    def __init__(self) -> None:
        """Instantiate a data handler."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset."""


class PyTorchDataHandler(BaseDataHandler):
    """PyTorch dataset handling interface."""

    @abstractmethod
    def get_dataset(self) -> Dataset:
        """Return a processed dataset.

        Returns
        -------
        Dataset
            A (processed) PyTorch dataset.
        """

    @abstractmethod
    def get_raw_dataset(self) -> Dataset | None:
        """Return a raw unprocessed dataset.

        Returns
        -------
        Dataset | None
            An unprocessed PyTorch dataset.
        """

    @abstractmethod
    def get_dataloader(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        batch_size: int = 32,
        shuffle: bool = False,
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
        shuffle : bool
            Whether to shuffle the data.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader.
        """


class SklearnDataHandler(BaseDataHandler):  # pragma: no cover
    """Scikit-learn data handling interface."""

    @abstractmethod
    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the processed data arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Features (X) and targets (y) as numpy arrays.
        """

    @abstractmethod
    def get_raw_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the original unprocessed data arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            Features (X) and targets (y) as numpy arrays.
        """

    @abstractmethod
    def get_subset(
        self, X: np.ndarray, y: np.ndarray, indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a subset of the data.

        Parameters
        ----------
        X : np.ndarray
            Feature array.
        y : np.ndarray
            Target array.
        indices : Sequence[int]
            The indices to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Subset of features and targets.
        """
