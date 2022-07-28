"""Privacy protected Support Vector Classifier."""

from __future__ import annotations

import copy

import numpy as np

from ..safemodel import SafeModel
from .dp_svc import DPSVC


class SafeSVC(SafeModel, DPSVC):
    """Privacy protected Support Vector Classifier."""

    def __init__(self, C=1.0, gamma="scale", dhat=1000, eps=10, **kwargs) -> None:
        """Initialises a differentially private SVC."""
        SafeModel.__init__(self)
        DPSVC.__init__(self)
        self.model_type: str = "SVC"
        self.ignore_items = [
            "model_save_file",
            "ignore_items",
            "train_features",
            "train_labels",
            "unique_labels",
            "train_labels",
            "weights",
            "noisy_weights",
        ]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Do fit and then store model dict"""
        super().fit(x, y)
        self.saved_model = copy.deepcopy(self.__dict__)
