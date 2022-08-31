"""dataset.py - class to represent a dataset"""

import numpy as np

class Data:
    """Stripped down Data class"""
    def __init__(self) -> None:
        self.name: str = ""
        self.n_samples: int = 0
        self.n_features: int = 0

        self.x_train: np.ndarray
        self.y_train: np.ndarray

        self.x_test: np.ndarray
        self.y_test: np.ndarray

    def add_processed_data(
        self,
        x_train: np.ndarray,
        y_train:np.ndarray,
        x_test:np.ndarray,
        y_test:np.ndarray) -> None:
        """Add a processed and split dataset"""
        self.x_train = x_train
        self.y_train = np.array(y_train, int)
        self.x_test = x_test
        self.y_test = np.array(y_test, int)
        self.n_samples = len(x_train) + len(y_train)

    def __str__(self):
        return self.name
