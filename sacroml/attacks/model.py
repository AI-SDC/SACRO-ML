"""Base class for target models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Model(ABC):  # pylint: disable=too-many-instance-attributes
    """Interface to a target model."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: Any,
        model_path: str = "",
        model_module_path: str = "",
        model_name: str = "",
        model_params: dict | None = None,
        train_module_path: str = "",
        train_params: dict | None = None,
    ) -> None:
        """Instantiate a target model.

        Parameters
        ----------
        model : Any
            Model.
        model_path : str
            Path (including extension) of a saved model.
        model_module_path : str
            Path (including extension) of Python module containing model class.
        model_name : str
            Class name of model.
        model_params : dict | None
            Hyperparameters for instantiating the model.
        train_module_path : str
            Path (including extension) of Python module containing train function.
        train_params : dict | None
            Hyperparameters for training the model.
        """
        self.model: Any = model
        self.model_path: str = model_path
        self.model_module_path: str = model_module_path
        self.model_name: str = model_name
        self.model_params: dict = {} if model_params is None else model_params
        self.train_module_path: str = train_module_path
        self.train_params: dict = {} if train_params is None else train_params

        if self.model is not None:
            self.model_name = type(self.model).__name__
            self.model_type = type(self).__name__

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def clone(self) -> Model:
        """Return a copy of the model.

        Returns
        -------
        Model
            A cloned model.
        """

    @abstractmethod
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

    @abstractmethod
    def get_classes(self) -> np.ndarray:
        """Return the classes the model was trained to predict.

        Returns
        -------
        np.ndarray
            Classes.
        """

    @abstractmethod
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

    @abstractmethod
    def get_params(self) -> dict:
        """Get the parameters of this model.

        Returns
        -------
        dict
            Model parameters.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to persistent storage.

        Parameters
        ----------
        path : str
            Path including file extension to save model.
        """

    @classmethod
    @abstractmethod
    def load(  # pylint: disable=too-many-arguments
        cls,
        model_path: str,
        model_module_path: str,
        model_name: str,
        model_params: dict,
        train_module_path: str,
        train_params: dict,
    ) -> Model:
        """Load the model from persistent storage.

        Parameters
        ----------
        model_path : str
            Path (including file extension) of a saved model.
        model_module_path : str
            Path (including file extension) of Python module containing model class.
        model_name : str
            Class name of model.
        model_params : dict
            Hyperparameters for instantiating the model.
        train_module_path : str
            Path (including extension) of Python module containing train function.
        train_params : dict
            Hyperparameters for training the model.

        Returns
        -------
        Model
            A loaded model.
        """
