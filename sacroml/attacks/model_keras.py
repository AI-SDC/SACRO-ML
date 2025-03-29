"""Keras target models.

TEMPLATE - NEEDS COMPLETING - REMOVE NOQAs
"""

from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf

from sacroml.attacks.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KerasModel(Model):
    """Interface to tensorflow.keras.Model models."""

    def __init__(self, model: tf.keras.Model) -> None:
        """Instantiate a target model.

        Parameters
        ----------
        model : Any
            Trained target model.
        """
        super().__init__(model)

    def get_generalisation_error(
        self,
        X_train: np.ndarray,  # noqa: ARG002
        y_train: np.ndarray,  # noqa: ARG002
        X_test: np.ndarray,  # noqa: ARG002
        y_test: np.ndarray,  # noqa: ARG002
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
        return np.nan

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
        metrics = self.model.evaluate(X, y, verbose=0)
        if len(metrics) > 1:
            return metrics[1]  # Accuracy for classification
        return -metrics[0]  # Negative loss for regression

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
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:  # noqa: ARG002
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
        return self.model

    def clone(self) -> Model:
        """Return a copy of the model.

        Returns
        -------
        Model
            A cloned model.
        """
        return self

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
        return X

    def get_classes(self) -> np.ndarray:
        """Return the classes the model was trained to predict.

        Returns
        -------
        np.ndarray
            Classes.
        """
        return np.ones(10)

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
        return self.model

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

    @classmethod
    def load(cls, path: str) -> KerasModel:
        """Load the model from persistent storage.

        Parameters
        ----------
        path : str
            Path including file extension to load model.

        Returns
        -------
        KerasModel
            A loaded keras model.
        """
        model = tf.keras.models.load_model(path)
        return cls(model)
