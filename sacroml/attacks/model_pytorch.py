"""Pytorch target models."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import torch
from torch import cuda
from torch.nn.functional import softmax

from sacroml.attacks.model import Model, create_model, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PytorchModel(Model):
    """Interface to Pytorch models."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: torch.nn.Module,
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
        model : torch.nn.Module
            Pytorch model.
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_tensor = torch.FloatTensor(X).to(device)
        model = self.model.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(x_tensor)
            _, y_pred = torch.max(logits, 1)

        return y_pred.cpu().numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> PytorchModel:
        """Fit a model from scratch.

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
        #  Create a new model using the provided model class
        self.model = create_model(
            self.model_module_path, self.model_name, self.model_params
        )
        #  Fit using the provided train function
        return train_model(self.model, self.train_module_path, self.train_params, X, y)

    def clone(self) -> PytorchModel:
        """Return a copy of the model.

        Returns
        -------
        Model
            A cloned model.
        """
        model: torch.nn.Module = deepcopy(self.model)

        return PytorchModel(
            model=model,
            model_path=self.model_path,
            model_module_path=self.model_module_path,
            model_name=self.model_name,
            model_params=self.model_params,
            train_module_path=self.train_module_path,
            train_params=self.train_params,
        )

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

    def get_classes(self) -> np.ndarray:  # pragma: no cover
        """Return the classes the model was trained to predict.

        Returns
        -------
        np.ndarray
            Classes.
        """
        last_linear = None

        # First try to find a named output layer
        names = ["output", "classifier", "head", "pred"]
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
                if any(output_name in name.lower() for output_name in names):
                    n_outputs = max(2, module.out_features)  # Deal with binary
                    logger.debug("Assuming model outputs %d classes", n_outputs)
                    return np.arange(n_outputs)

        # If no named layer found, use the last Linear layer in the model
        if last_linear is not None:
            n_outputs = max(2, last_linear.out_features)
            logger.debug("Assuming model outputs %d classes", n_outputs)
            return np.arange(n_outputs)

        # Last attempt, see if a classes attribute was defined
        try:
            return self.model.classes
        except AttributeError as error:
            raise AttributeError(
                "No Linear layer found and model has no 'classes' attribute"
            ) from error

    def set_params(self, **kwargs) -> PytorchModel:
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
        if "random_state" in kwargs:
            torch.manual_seed(kwargs["random_state"])
            torch.cuda.manual_seed_all(kwargs["random_state"])

        return self.model

    def get_params(self) -> dict:  # pragma: no cover
        """Get the parameters of this model.

        Returns
        -------
        dict
            Model parameters.
        """
        return self.model_params

    def save(self, path: str) -> None:
        """Save the model to persistent storage.

        Parameters
        ----------
        path : str
            Path including file extension to save model.
        """
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(  # pylint: disable=too-many-arguments
        cls,
        model_path: str,
        model_module_path: str,
        model_name: str,
        model_params: dict,
        train_module_path: str,
        train_params: dict,
    ) -> PytorchModel:
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
        PytorchModel
            A loaded pytorch model.
        """
        # Create model
        model = create_model(model_module_path, model_name, model_params)

        # Load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Return a new PytorchModel
        return cls(
            model,
            model_path=model_path,
            model_module_path=model_module_path,
            model_name=model_name,
            model_params=model_params,
            train_module_path=train_module_path,
            train_params=train_params,
        )
