"""Sklearn target models."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import sklearn

from sacroml.attacks.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SklearnModel(Model):
    """Interface to sklearn.base.BaseEstimator models."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: sklearn.base.BaseEstimator,
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
        model : sklearn.base.BaseEstimator
            Scikit-learn model.
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
        super().__init__(
            model=model,
            model_path=model_path,
            model_module_path=model_module_path,
            model_name=model_name,
            model_params=model_params,
            train_module_path=train_module_path,
            train_params=train_params,
        )

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
            except sklearn.exceptions.NotFittedError:  # pragma: no cover
                return np.nan
        return np.nan  # pragma: no cover

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
        """Return a copy of the model.

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

    def save(self, path: str) -> None:
        """Save the model to persistent storage.

        Parameters
        ----------
        path : str
            Path including file extension to save model.
        """
        ext: str = Path(path).suffix
        if ext == ".pkl":
            with open(path, "wb") as fp:
                pickle.dump(self.model, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported file format for saving a model: {ext}")

    @classmethod
    def load(  # pylint: disable=too-many-arguments
        cls,
        model_path: str,
        model_module_path: str,
        model_name: str,
        model_params: dict,
        train_module_path: str,
        train_params: dict,
    ) -> SklearnModel:
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
        SklearnModel
            A loaded sklearn model.
        """
        ext: str = Path(model_path).suffix
        if ext == ".pkl":
            with open(model_path, "rb") as fp:
                model = pickle.load(fp)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported file format for loading a model: {ext}")
        return cls(
            model,
            model_path=model_path,
            model_module_path=model_module_path,
            model_name=model_name,
            model_params=model_params,
            train_module_path=train_module_path,
            train_params=train_params,
        )
