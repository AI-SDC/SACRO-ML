"""Store information about the target model and data."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from sacroml.attacks.data import PyTorchDataHandler, SklearnDataHandler
from sacroml.attacks.model import create_dataset
from sacroml.attacks.model_pytorch import PytorchModel, dataloader_to_numpy
from sacroml.attacks.model_sklearn import SklearnModel

MODEL_REGISTRY: dict[str, Any] = {
    "PytorchModel": PytorchModel,
    "SklearnModel": SklearnModel,
}

DATA_ATTRIBUTES: list[str] = [
    "X_train",
    "y_train",
    "X_test",
    "y_test",
    "X_train_orig",
    "y_train_orig",
    "X_test_orig",
    "y_test_orig",
    "proba_train",
    "proba_test",
    "indices_train",
    "indices_test",
]

ARRAYS: tuple[str, ...] = (
    "X_train",
    "y_train",
    "X_test",
    "y_test",
    "X_train_orig",
    "y_train_orig",
    "X_test_orig",
    "y_test_orig",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Target:
    """Store information about the target model and data.

    Attributes
    ----------
    model : Any
        Trained target model.
    model_path : str
        Path to a saved model.
    model_module_path : str
        Path to module containing model class.
    model_name : str
        Class name of model.
    model_params : dict or None
        Hyperparameters for instantiating the model.
    train_module_path : str
        Path to module containing training function.
    train_params : dict or None
        Hyperparameters for training the model.
    dataset_name : str
        The name of the dataset.
    dataset_module_path : str
        Path to module containing dataset loading function.
    features : dict
        Dictionary describing the dataset features.
    X_train : np.ndarray or None
        The (processed) training inputs.
    y_train : np.ndarray or None
        The (processed) training outputs.
    X_test : np.ndarray or None
        The (processed) testing inputs.
    y_test : np.ndarray or None
        The (processed) testing outputs.
    X_train_orig : np.ndarray or None
        The original (unprocessed) training inputs.
    y_train_orig : np.ndarray or None
        The original (unprocessed) training outputs.
    X_test_orig : np.ndarray or None
        The original (unprocessed) testing inputs.
    y_test_orig : np.ndarray or None
        The original (unprocessed) testing outputs.
    proba_train : np.ndarray or None
        The model predicted training probabilities.
    proba_test : np.ndarray or None
        The model predicted testing probabilities.
    indices_train : Sequence[int] or None
        Indices of training set samples.
    indices_test : Sequence[int] or None
        Indices of test set samples.
    safemodel : list
        Results of safemodel disclosure checking.
    """

    # Model attributes
    model: Any = None
    model_path: str = ""
    model_module_path: str = ""
    model_name: str = ""
    model_params: dict | None = None
    train_module_path: str = ""
    train_params: dict | None = None

    # Dataset attributes
    dataset_name: str = ""
    dataset_module_path: str = ""
    features: dict = field(default_factory=dict)

    # Data arrays
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    X_train_orig: np.ndarray | None = None
    y_train_orig: np.ndarray | None = None
    X_test_orig: np.ndarray | None = None
    y_test_orig: np.ndarray | None = None
    proba_train: np.ndarray | None = None
    proba_test: np.ndarray | None = None
    indices_train: Sequence[int] | None = None
    indices_test: Sequence[int] | None = None

    # Safemodel properties
    safemodel: list = field(default_factory=list)

    def __post_init__(self):
        """Initialise the model wrapper after dataclass creation."""
        self.model = self._wrap_model(self.model)

    def _wrap_model(self, model: Any) -> Any:
        """Wrap the model in a wrapper class."""
        if model is None:
            return None

        if isinstance(model, sklearn.base.BaseEstimator):
            return SklearnModel(
                model=model,
                model_path=self.model_path,
                model_module_path=self.model_module_path,
                model_name=self.model_name,
                model_params=self.model_params,
                train_module_path=self.train_module_path,
                train_params=self.train_params,
            )
        if isinstance(model, torch.nn.Module):
            return PytorchModel(
                model=model,
                model_path=self.model_path,
                model_module_path=self.model_module_path,
                model_name=self.model_name,
                model_params=self.model_params,
                train_module_path=self.train_module_path,
                train_params=self.train_params,
            )
        if isinstance(model, (SklearnModel, PytorchModel)):
            return model
        raise ValueError(f"Unsupported model type: {type(model)}")  # pragma: no cover

    def load_pytorch_dataset(self) -> None:  # pragma: no cover
        """Wrap dataset for Pytorch models given a dataset Python script."""
        if self.indices_train is None or self.indices_test is None:
            raise ValueError("Can't load dataset module without indices.")
        try:
            # Create a new data handler with a supplied class
            handler: PyTorchDataHandler = create_dataset(
                self.dataset_module_path, self.dataset_name
            )

            # Get processed data
            data = handler.get_dataset()
            train_loader = handler.get_dataloader(data, self.indices_train)
            test_loader = handler.get_dataloader(data, self.indices_test)

            self.X_train, self.y_train = dataloader_to_numpy(train_loader)
            self.X_test, self.y_test = dataloader_to_numpy(test_loader)

            # Get raw unprocessed data
            data = handler.get_raw_dataset()
            if data:
                train_loader = handler.get_dataloader(data, self.indices_train)
                test_loader = handler.get_dataloader(data, self.indices_test)

                self.X_train_orig, self.y_train_orig = dataloader_to_numpy(train_loader)
                self.X_test_orig, self.y_test_orig = dataloader_to_numpy(test_loader)

            # Display array shapes
            for arr in ARRAYS:
                if (array := getattr(self, arr)) is not None:
                    logger.info("Loaded: %s shape: %s", arr, array.shape)

        except Exception as e:  # pragma: no cover
            raise ValueError(f"Failed to load data using class: {e}") from e

    def load_sklearn_dataset(self) -> None:  # pragma: no cover
        """Wrap dataset for scikit-learn models given a dataset Python script."""
        if self.indices_train is None or self.indices_test is None:
            raise ValueError("Can't load dataset module without indices.")
        try:
            # Create a new data handler with a supplied class
            handler: SklearnDataHandler = create_dataset(
                self.dataset_module_path, self.dataset_name
            )

            # Get processed data
            X, y = handler.get_data()
            self.X_train, self.y_train = handler.get_subset(X, y, self.indices_train)
            self.X_test, self.y_test = handler.get_subset(X, y, self.indices_test)

            # Get raw unprocessed data
            data = handler.get_raw_data()
            if data:
                X, y = data
                self.X_train_orig, self.y_train_orig = handler.get_subset(
                    X, y, self.indices_train
                )
                self.X_test_orig, self.y_test_orig = handler.get_subset(
                    X, y, self.indices_test
                )

            # Display array shapes
            for arr in ARRAYS:
                if (array := getattr(self, arr)) is not None:
                    logger.info("Loaded: %s shape: %s", arr, array.shape)

        except Exception as e:  # pragma: no cover
            raise ValueError(f"Failed to load data using class: {e}") from e

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.features)

    def add_feature(self, name: str, indices: list[int], encoding: str) -> None:
        """Add a feature description to the data dictionary."""
        index: int = len(self.features)
        self.features[index] = {
            "name": name,
            "indices": indices,
            "encoding": encoding,
        }

    def add_safemodel_results(self, data: list) -> None:
        """Add safemodel disclosure checking results."""
        self.safemodel = data

    def has_model(self) -> bool:
        """Return whether the target has a loaded model."""
        return self.model is not None and self.model.model is not None

    def has_data(self) -> bool:
        """Return whether the target has all processed data."""
        attrs: list[str] = ["X_train", "y_train", "X_test", "y_test"]
        return all(getattr(self, attr) is not None for attr in attrs)

    def has_raw_data(self) -> bool:
        """Return whether the target has all raw data."""
        attrs: list[str] = [
            "X_train_orig",
            "y_train_orig",
            "X_test_orig",
            "y_test_orig",
        ]
        return all(getattr(self, attr) is not None for attr in attrs)

    def has_probas(self) -> bool:
        """Return whether the target has all probability data."""
        return self.proba_train is not None and self.proba_test is not None

    def get_generalisation_error(self) -> float:
        """Calculate model generalisation error."""
        if not (self.has_model() and self.has_data()):
            return np.nan
        return self.model.get_generalisation_error(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

    def save(self, path: str = "target", ext: str = "pkl") -> None:
        """Save target to persistent storage."""
        path = os.path.normpath(path)
        os.makedirs(path, exist_ok=True)

        # Create target dictionary
        target = {
            "dataset_name": self.dataset_name,
            "dataset_module_path": self.dataset_module_path,
            "features": self.features,
            "generalisation_error": self.get_generalisation_error(),
            "safemodel": self.safemodel,
        }

        # Save model
        self._save_model(path, ext, target)

        # Save dataset module
        self._save_dataset_module(path, target)

        # Save data arrays
        for attr in DATA_ATTRIBUTES:
            self._save_array(path, target, attr)

        # Save YAML config
        yaml_path = os.path.join(path, "target.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(target, f, default_flow_style=False, sort_keys=False)

    def load(self, path: str = "target") -> None:
        """Load target from persistent storage."""
        yaml_path = os.path.join(path, "target.yaml")

        with open(yaml_path, encoding="utf-8") as f:
            target = yaml.safe_load(f)

        # Load basic attributes
        for attr in ["dataset_name", "safemodel"]:
            if attr in target:
                setattr(self, attr, target[attr])

        # Load features (convert string keys to int)
        if "features" in target:
            self.features = {int(k): v for k, v in target["features"].items()}

        # Load paths
        if "dataset_module_path" in target and target["dataset_module_path"] != "":
            self.dataset_module_path = os.path.join(path, target["dataset_module_path"])

        # Load model and data
        self._load_model(path, target)
        for attr in DATA_ATTRIBUTES:
            self._load_array(path, target, attr)

        if self.dataset_module_path != "":
            if isinstance(self.model, PytorchModel):
                self.load_pytorch_dataset()
            elif isinstance(self.model, SklearnModel):
                self.load_sklearn_dataset()
            else:  # pragma: no cover
                logger.warning("Dataset module supplied for unsupported model type.")

    def _save_model(self, path: str, ext: str, target: dict) -> None:
        """Save model to disk."""
        if self.model is None:  # pragma: no cover
            return

        target.update(
            {
                "model_type": self.model.model_type,
                "model_name": self.model.model_name,
                "model_params": self.model.get_params(),
            }
        )

        # Copy module files
        if self.model_module_path:
            shutil.copy2(self.model_module_path, os.path.join(path, "model.py"))
            target["model_module_path"] = "model.py"

        if getattr(self.model, "train_module_path", ""):
            shutil.copy2(self.model.train_module_path, os.path.join(path, "train.py"))
            target["train_module_path"] = "train.py"
            target["train_params"] = self.model.train_params

        # Save model
        model_path = os.path.join(path, f"model.{ext}")
        self.model.save(model_path)
        target["model_path"] = f"model.{ext}"

    def _save_dataset_module(self, path: str, target: dict) -> None:
        """Save dataset module."""
        if self.dataset_module_path:  # pragma: no cover
            shutil.copy2(self.dataset_module_path, os.path.join(path, "dataset.py"))
            target["dataset_module_path"] = "dataset.py"

    def _save_array(self, path: str, target: dict, attr_name: str) -> None:
        """Save numpy array as pickle."""
        arr = getattr(self, attr_name)
        if arr is not None:
            arr_path = os.path.join(path, f"{attr_name}.pkl")
            with open(arr_path, "wb") as f:
                pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
            target[f"{attr_name}_path"] = f"{attr_name}.pkl"
        else:
            target[f"{attr_name}_path"] = ""

    def _load_model(self, path: str, target: dict) -> None:
        """Load model from disk."""
        model_type = target.get("model_type", "")
        if not model_type or model_type not in MODEL_REGISTRY:  # pragma: no cover
            logger.info("Cannot load model: %s", model_type)
            return

        model_class = MODEL_REGISTRY[model_type]
        self.model = model_class.load(
            model_path=os.path.join(path, target.get("model_path", "")),
            model_module_path=os.path.join(path, target.get("model_module_path", "")),
            model_name=target.get("model_name", ""),
            model_params=target.get("model_params", {}),
            train_module_path=os.path.join(path, target.get("train_module_path", "")),
            train_params=target.get("train_params", {}),
        )
        logger.info("Loaded: %s : %s", model_type, target.get("model_name", ""))

    def _load_array(self, path: str, target: dict, attr_name: str) -> None:
        """Load array from disk."""
        path_key = f"{attr_name}_path"
        if path_key in target and target[path_key]:
            arr_path = os.path.join(path, target[path_key])
            self.load_array(arr_path, attr_name)

    def load_array(self, arr_path: str, attr_name: str) -> None:
        """Load array from pickle or CSV file."""
        _, ext = os.path.splitext(arr_path)

        if ext == ".pkl":
            arr = self._load_pickle(arr_path, attr_name)
        elif ext == ".csv":  # pragma: no cover
            arr = self._load_csv(arr_path, attr_name)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported file extension: {ext}")

        setattr(self, attr_name, arr)

    def _load_pickle(self, path: str, name: str) -> Any:  # pragma: no cover
        """Load array from pickle file."""
        try:
            with open(path, "rb") as f:
                arr = pickle.load(f)
                if hasattr(arr, "shape"):
                    logger.info("%s shape: %s", name, arr.shape)
                elif isinstance(arr, Sequence) and not isinstance(arr, str):
                    logger.info("%s length: %d", name, len(arr))
                else:
                    logger.info("%s is a scalar value", name)
                return arr
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Pickle file not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading pickle file {path}: {e}") from e

    def _load_csv(self, path: str, name: str) -> np.ndarray:  # pragma: no cover
        """Load array from CSV file."""
        try:
            arr = pd.read_csv(path, header=None).to_numpy()
            logger.info("%s shape: %s", name, arr.shape)
            return arr
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found: {path}") from e
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"CSV file is empty: {path}") from e
        except Exception as e:
            raise ValueError(f"Error reading CSV file {path}: {e}") from e
