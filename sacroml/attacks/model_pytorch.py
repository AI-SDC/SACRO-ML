"""Pytorch target models."""

from __future__ import annotations

import importlib
import logging
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import cuda
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset

from sacroml.attacks.model import Model, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PytorchModel(Model):
    """Interface to Pytorch models."""

    def __init__(
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

    def get_losses(self, data: np.ndarray, labels: np.ndarray) -> np.array:
        """Return the losses for a given set of samples.

        Parameters
        ----------
        data : np.ndarray
            Features of the records for whicgh losses are required
        labels : np.ndarray
            Actual labels

        Returns
        -------
        np.array
            array of losses (1.0 - predicted confidence for correct label)
        """
        labelidxs = self.get_label_indices(labels)
        numrows = len(data)
        allprobs = self.predict_proba(data)
        losses = np.zeros(numrows)
        for i in range(numrows):
            losses[i] = 1.0 - allprobs[i, labelidxs[i]]
        return losses

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
        self.model.to(device)
        self.model.eval()

        # Create a dataloader
        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Compute predictions
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(device)
                logits = self.model(x_batch)
                _, y_pred = torch.max(logits, 1)
                all_preds.append(y_pred.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        self.model.to("cpu")
        return all_preds.numpy().astype(np.float64)

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
        # Create a dataloader
        dataloader: DataLoader = numpy_to_dataloader(X, y, batch_size=32, shuffle=True)
        #  Fit using the provided train function
        return train_model(
            self.model, self.train_module_path, self.train_params, dataloader
        )

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

        # Create a dataloader
        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Compute the probabilities
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(device)
                logits = self.model(x_batch)
                logits = torch.where(  # guard against nans
                    torch.isfinite(logits), logits, torch.zeros_like(logits)
                )
                probs = softmax(logits, dim=1)
                all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        self.model.to("cpu")
        return all_probs.numpy().astype(np.float64)

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
    def load(
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


def dataloader_to_numpy(dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Convert DataLoader to numpy arrays.

    Parameters
    ----------
    dataloader : DataLoader
        Pytorch DataLoader to convert.

    Returns
    -------
    np.ndarray, np.ndarray
        Numpy arrays.
    """
    all_inputs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            all_inputs.append(inputs.cpu().numpy().astype(np.float64))
            all_labels.append(labels.cpu().numpy().astype(np.int32))

    X = np.concatenate(all_inputs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def numpy_to_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = False
):
    """Convert numpy arrays to PyTorch DataLoader."""
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: Any, train_module_path: str, train_params: dict, dataloader: DataLoader
) -> Any:
    """Train a model from a code path.

    Parameters
    ----------
    model : Any
        Model to train.
    train_module_path : str
        Path to Python code containing a train function.
    train_params : dict
        Parameters for executing the train function.
    dataloader : DataLoader
        DataLoader with training data.

    Returns
    -------
    Any
        Trained model.
    """
    try:
        # Convert file path to module path
        train_module_path = train_module_path.replace("/", ".").replace("\\", ".")
        train_module_path = train_module_path.rstrip(".py")

        # Import training function
        module = importlib.import_module(train_module_path)
        train_function = module.train

        # Train model
        train_function(model=model, dataloader=dataloader, **train_params)
        return model

    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to train model: {e}") from e
