"""Pytorch target models."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import torch
from torch import cuda
from torch.nn.functional import softmax

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
        train = self.score(X_train, y_train)
        test = self.score(X_test, y_test)
        return test - train

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the model accuracy for a set of samples.

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
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the model error for a set of samples.

        Parameters
        ----------
        X : np.ndarray
            Features of the samples to be scored.
        y : np.ndarray
            Labels of the samples to be scored.

        Returns
        -------
        float
            Model loss.
        """
        x_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            loss = self.model.criterion(logits, y_tensor)

        return loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the model predictions for a set of samples.

        Parameters
        ----------
        X : np.ndarray
            Features of the samples to be predicted.

        Returns
        -------
        np.ndarray
            Model predictions (label encoding).
        """
        x_tensor = torch.FloatTensor(X)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            _, y_pred = torch.max(logits, 1)

        return y_pred.numpy()

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
            probabilities = softmax(logits, dim=1)

        self.model.to("cpu")
        return probabilities.cpu().numpy()  # should be softmax values

    def get_classes(self) -> np.ndarray:
        """Return the classes the model was trained to predict.

        Returns
        -------
        np.ndarray
            Classes.
        """
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Linear):
                n_outputs: int = module.out_features
                n_outputs = max(2, n_outputs)  # Deal with binary output
                return np.arange(n_outputs)
        return self.model.classes

    def set_params(self, **kwargs) -> Model:  # noqa: ARG002
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
        return self.model  # do nothing

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
            model = torch.load(path, weights_only=False, map_location=device)
            model.eval()
            return cls(model)
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model: {e}") from e


def reset_weights(layer: torch.nn.Module) -> None:
    """Reset the layer weights."""
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
