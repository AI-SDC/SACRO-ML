"""Sklearn target models."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import torch
from torch import cuda

from sacroml.attacks.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PytorchModel(Model):
    """Interface to Pytorch models."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Instantiate a target model.

        Parameters
        ----------
        model : Any
            Trained target model.
        """
        super().__init__(model)

    def get_generalisation_error(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """Return the model generalisation error for a set of samples.

        Parameters
        ----------
        X_train : np.ndarray
            Features of the training samples to be scored.
        y_train : np.ndarray
            Labels of the training samples to be scored.
        X_test : np.ndarray
            Features of the test samples to be scored.
        y_test : np.ndarray
            Labels of the test samples to be scored.

        Returns
        -------
        float
            Model generalisation error.
        """
        train = self.model.score(X_train, y_train)
        test = self.model.score(X_test, y_test)
        return test - train

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the model scores for a set of samples.

        Parameters
        ----------
        X : np.ndarray
            Features of the samples to be scored.
        y : np.ndarray
            Labels of the samples to be scored.

        Returns
        -------
        float
            Model score.
        """
        return self.model.score(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the model predictions for a set of samples.

        Parameters
        ----------
        X : np.ndarray
            Features of the samples to be predicted.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        return self.model.predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        """Fit the model.

        Parameters
        ----------
        X : np.ndarray
            Features of the samples to be fitted.
        y : np.ndarray
            Labels of the samples to be fitted.

        Returns
        -------
        self
            Fitted model.
        """
        return self.model.fit(X, y)

    def clone(self) -> Model:
        """Return a clone of the model.

        A new model with the same parameters that has not been fit on any data.

        Returns
        -------
        Model
            A cloned model.
        """
        model: torch.nn.Module = deepcopy(self.model)
        model.apply(reset_weights)
        return PytorchModel(model)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the model predicted probabilities for a set of samples.

        Parameters
        ----------
        X : np.ndarray
            Features of the samples whose probabilities are to be returned.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        device = "cuda:0" if cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self.model(x_tensor)

        self.model.to("cpu")
        return logits.cpu().numpy()  # NOT SOFTMAXED

    def get_classes(self) -> np.ndarray:
        """Return the classes the model was trained to predict.

        Returns
        -------
        np.ndarray
            Classes.
        """
        return self.model.classes_

    def set_params(self, **kwargs) -> Model:
        """Set the parameters of this model.

        Parameters
        ----------
        **kwargs : dict
            Model parameters.

        Returns
        -------
        self
            Instance of model class.
        """
        return self.model.set_params(**kwargs)

    def get_params(self) -> dict:
        """Get the parameters of this model.

        Returns
        -------
        dict
            Model parameters.
        """
        return self.model.get_params()

    def get_name(self) -> str:
        """Get the name of this model.

        Returns
        -------
        str
            Model name.
        """
        return type(self.model).__name__

    def save(self, path: str) -> None:
        """Save the model to persistent storage.

        Parameters
        ----------
        path : str
            Path including file extension to save model.
        """
        torch.save(self.model, path)

    @classmethod
    def load(cls, path: str) -> PytorchModel:
        """Load the model from persistent storage.

        Parameters
        ----------
        path : str
            Path including file extension to load model.

        Returns
        -------
        PytorchModel
            A loaded pytorch model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.load(path, map_location=device)
            model.eval()
            return cls(model)
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model: {e}") from e


def reset_weights(layer: torch.nn.Module) -> None:
    """Reset the layer weights."""
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
