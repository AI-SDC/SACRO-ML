"""Sklearn target models."""

from __future__ import annotations

import logging

import numpy as np
import sklearn

from sacroml.attacks.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SklearnModel(Model):
    """Interface to sklearn.base.BaseEstimator models."""

    def __init__(self, model: sklearn.base.BaseEstimator) -> None:
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
        if hasattr(self.model, "score"):
            try:
                train = self.model.score(X_train, y_train)
                test = self.model.score(X_test, y_test)
                return test - train
            except sklearn.exceptions.NotFittedError:
                return np.nan
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
        model: sklearn.base.BaseEstimator = sklearn.base.clone(self.model)
        return SklearnModel(model)

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
        return self.model.predict_proba(X)

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
