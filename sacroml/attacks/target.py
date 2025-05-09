"""Store information about the target model and data."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from sacroml.attacks.model_pytorch import PytorchModel
from sacroml.attacks.model_sklearn import SklearnModel

registry: dict = {
    "PytorchModel": PytorchModel,
    "SklearnModel": SklearnModel,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Target:  # pylint: disable=too-many-instance-attributes
    """Store information about the target model and data."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        model: Any = None,
        model_path: str = "",
        model_module_path: str = "",
        model_name: str = "",
        model_params: dict | None = None,
        train_module_path: str = "",
        train_params: dict | None = None,
        dataset_name: str = "",
        dataset_module_path: str = "",
        features: dict | None = None,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        X_orig: np.ndarray | None = None,
        y_orig: np.ndarray | None = None,
        X_train_orig: np.ndarray | None = None,
        y_train_orig: np.ndarray | None = None,
        X_test_orig: np.ndarray | None = None,
        y_test_orig: np.ndarray | None = None,
        proba_train: np.ndarray | None = None,
        proba_test: np.ndarray | None = None,
    ) -> None:
        """Store information about a target model and associated data.

        Parameters
        ----------
        model : Any
            Trained target model.
        model_path : str
            Path to a saved model.
        model_module_path : str
            Path to module containing model class.
        model_name : str
            Class name of model.
        model_params : dict | None
            Hyperparameters for instantiating the model.
        train_module_path : str
            Path to module containing training function.
        train_params : dict | None
            Hyperparameters for training the model.
        dataset_name : str
            The name of the dataset.
        dataset_module_path : str
            Path to module containing dataset loading function.
        features : dict
            Dictionary describing the dataset features.
        X_train : np.ndarray | None
            The (processed) training inputs.
        y_train : np.ndarray | None
            The (processed) training outputs.
        X_test : np.ndarray | None
            The (processed) testing inputs.
        y_test : np.ndarray | None
            The (processed) testing outputs.
        X_orig : np.ndarray | None
            The original (unprocessed) dataset inputs.
        y_orig : np.ndarray | None
            The original (unprocessed) dataset outputs.
        X_train_orig : np.ndarray | None
            The original (unprocessed) training inputs.
        y_train_orig : np.ndarray | None
            The original (unprocessed) training outputs.
        X_test_orig : np.ndarray | None
            The original (unprocessed) testing inputs.
        y_test_orig : np.ndarray | None
            The original (unprocessed) testing outputs.
        proba_train : np.ndarray | None
            The model predicted training probabilities.
        proba_test : np.ndarray | None
            The model predicted testing probabilities.
        """
        # Model - details
        if isinstance(model, sklearn.base.BaseEstimator):
            self.model = SklearnModel(
                model=model,
                model_path=model_path,
                model_module_path=model_module_path,
                model_name=model_name,
                model_params=model_params,
                train_module_path=train_module_path,
                train_params=train_params,
            )
        elif isinstance(model, torch.nn.Module):
            self.model = PytorchModel(
                model=model,
                model_path=model_path,
                model_module_path=model_module_path,
                model_name=model_name,
                model_params=model_params,
                train_module_path=train_module_path,
                train_params=train_params,
            )
        elif isinstance(model, (SklearnModel, PytorchModel)):
            self.model = model
        elif model is not None:  # pragma: no cover
            raise ValueError(f"Unsupported model type: {type(model)}")
        else:  # for subsequent model loading
            self.model = None

        # Model - code
        self.model_module_path = model_module_path

        # Data - model predicted probabilities
        self.proba_train: np.ndarray | None = proba_train
        self.proba_test: np.ndarray | None = proba_test

        # Dataset - details
        self.dataset_name: str = dataset_name

        # Dataset - code
        self.dataset_module_path = dataset_module_path

        #  Dataset - processed
        self.X_train: np.ndarray | None = X_train
        self.y_train: np.ndarray | None = y_train
        self.X_test: np.ndarray | None = X_test
        self.y_test: np.ndarray | None = y_test
        self.n_samples: int = 0
        if X_train is not None and X_test is not None:
            self.n_samples = len(X_train) + len(X_test)

        #  Dataset - unprocessed
        self.X_orig: np.ndarray | None = X_orig
        self.y_orig: np.ndarray | None = y_orig
        self.X_train_orig: np.ndarray | None = X_train_orig
        self.y_train_orig: np.ndarray | None = y_train_orig
        self.X_test_orig: np.ndarray | None = X_test_orig
        self.y_test_orig: np.ndarray | None = y_test_orig
        self.n_samples_orig: int = 0
        if X_train_orig is not None and X_test_orig is not None:
            self.n_samples_orig = len(X_train_orig) + len(X_test_orig)
        self.features: dict = features if features is not None else {}
        self.n_features: int = len(self.features)

        #  Safemodel report
        self.safemodel: list = []

    def add_processed_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Add a processed and split dataset."""
        self.X_train = X_train
        self.y_train = np.array(y_train, int)
        self.X_test = X_test
        self.y_test = np.array(y_test, int)
        self.n_samples = len(X_train) + len(X_test)

    def add_feature(self, name: str, indices: list[int], encoding: str) -> None:
        """Add a feature description to the data dictionary."""
        index: int = len(self.features)
        self.features[index] = {
            "name": name,
            "indices": indices,
            "encoding": encoding,
        }
        self.n_features = len(self.features)

    def add_raw_data(  # pylint: disable=too-many-arguments
        self,
        X_orig: np.ndarray,
        y_orig: np.ndarray,
        X_train_orig: np.ndarray,
        y_train_orig: np.ndarray,
        X_test_orig: np.ndarray,
        y_test_orig: np.ndarray,
    ) -> None:
        """Add original unprocessed dataset."""
        self.X_orig = X_orig
        self.y_orig = y_orig
        self.X_train_orig = X_train_orig
        self.y_train_orig = y_train_orig
        self.X_test_orig = X_test_orig
        self.y_test_orig = y_test_orig
        self.n_samples_orig = len(X_orig)

    def _save_model(self, path: str, ext: str, target: dict) -> None:
        """Save the target model.

        Parameters
        ----------
        path : str
            Path to write the model.
        ext : str
            File extension defining the model saved format, e.g., "pkl" or "sav".
        target : dict
            Target class as a dictionary for writing yaml.
        """
        if not self.model is None:
            target["model_type"] = self.model.model_type
            target["model_name"] = self.model.model_name
            target["model_params"] = self.model.get_params()

            if self.model_module_path != "":
                filename = os.path.normpath(f"{path}/model.py")
                shutil.copy2(self.model.model_module_path, filename)
                target["model_module_path"] = "model.py"

            if self.model.train_module_path != "":
                filename = os.path.normpath(f"{path}/train.py")
                shutil.copy2(self.model.train_module_path, filename)
                target["train_module_path"] = "train.py"
                target["train_params"] = self.model.train_params

            if not self.model is None:
                filename = os.path.normpath(f"{path}/model.{ext}")
                target["model_path"] = f"model.{ext}"
                self.model.save(filename)

    def _load_model(self, path: str, target: dict) -> None:
        """Load the target model.

        Parameters
        ----------
        path : str
            Path to a target directory.
        target : dict
            Target class as a dictionary for loading yaml.
        """
        #  Load attributes
        model_type: str = target.get("model_type", "")
        model_name: str = target.get("model_name", "")
        model_params: dict = target.get("model_params", {})
        model_path: str = target.get("model_path", "")
        model_module_path: str = target.get("model_module_path", "")
        train_module_path: str = target.get("train_module_path", "")
        train_params: dict = target.get("train_params", {})
        #  Normalise paths
        model_path = os.path.normpath(f"{path}/{model_path}")
        model_module_path = os.path.normpath(f"{path}/{model_module_path}")
        train_module_path = os.path.normpath(f"{path}/{train_module_path}")
        #  Load model
        if model_type in registry:
            model_class = registry[model_type]
            self.model = model_class.load(
                model_path=model_path,
                model_module_path=model_module_path,
                model_name=model_name,
                model_params=model_params,
                train_module_path=train_module_path,
                train_params=train_params,
            )
            logger.info("Loaded: %s : %s", model_type, model_name)
        else:  # pragma: no cover
            self.model = None
            logger.info("Can't load model: %s : %s", model_type, model_name)

    def _save_numpy(self, path: str, target: dict, name: str) -> None:
        """Save a numpy array variable as pickle.

        Parameters
        ----------
        path : str
            Path to save the data.
        target : dict
            Target class as a dictionary for writing yaml.
        name : str
            Name of the numpy array to save.
        """
        if getattr(self, name) is not None:
            np_path: str = os.path.normpath(f"{path}/{name}.pkl")
            target[f"{name}_path"] = f"{name}.pkl"
            with open(np_path, "wb") as fp:
                pickle.dump(getattr(self, name), fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            target[f"{name}_path"] = ""

    def load_array(self, arr_path: str, name: str) -> None:
        """Load a data array variable from file.

        Handles both .pkl and .csv files.

        Parameters
        ----------
        arr_path : str
            Filename of a data array.
        name : str
            Name of the data array to load.
        """
        path = os.path.normpath(arr_path)
        _, ext = os.path.splitext(path)
        if ext == ".pkl":
            arr = get_array_pkl(path, name)
        elif ext == ".csv":  # pragma: no cover
            arr = get_array_csv(path, name)
        else:  # pragma: no cover
            raise ValueError(f"Target cannot load {ext} files.") from None
        setattr(self, name, arr)

    def _load_array(self, arr_path: str, target: dict, name: str) -> None:
        """Load a data array variable contained in a yaml config.

        Parameters
        ----------
        arr_path : str
            Filename of a data array.
        target : dict
            Target class as a dictionary read from yaml.
        name : str
            Name of the data array to load.
        """
        key = f"{name}_path"
        if key in target and target[key] != "":
            path = f"{arr_path}/{target[key]}"
            self.load_array(path, name)

    def _save_data(self, path: str, target: dict) -> None:
        """Save the target model data.

        Parameters
        ----------
        path : str
            Path to save the data.
        target : dict
            Target class as a dictionary for writing yaml.
        """
        if self.dataset_module_path != "":  # pragma: no cover
            filename = os.path.normpath(f"{path}/dataset.py")
            shutil.copy2(self.dataset_module_path, filename)
            target["dataset_module_path"] = "dataset.py"

        self._save_numpy(path, target, "X_train")
        self._save_numpy(path, target, "y_train")
        self._save_numpy(path, target, "X_test")
        self._save_numpy(path, target, "y_test")
        self._save_numpy(path, target, "X_orig")
        self._save_numpy(path, target, "y_orig")
        self._save_numpy(path, target, "X_train_orig")
        self._save_numpy(path, target, "y_train_orig")
        self._save_numpy(path, target, "X_test_orig")
        self._save_numpy(path, target, "y_test_orig")
        self._save_numpy(path, target, "proba_train")
        self._save_numpy(path, target, "proba_test")

    def _load_data(self, path: str, target: dict) -> None:
        """Load the target model data.

        Parameters
        ----------
        path : str
            Path to load the data.
        target : dict
            Target class as a dictionary read from yaml.
        """
        self._load_array(path, target, "X_train")
        self._load_array(path, target, "y_train")
        self._load_array(path, target, "X_test")
        self._load_array(path, target, "y_test")
        self._load_array(path, target, "X_orig")
        self._load_array(path, target, "y_orig")
        self._load_array(path, target, "X_train_orig")
        self._load_array(path, target, "y_train_orig")
        self._load_array(path, target, "X_test_orig")
        self._load_array(path, target, "y_test_orig")
        self._load_array(path, target, "proba_train")
        self._load_array(path, target, "proba_test")

    def _ge(self) -> float:
        """Return the model generalisation error.

        Returns
        -------
        float
            Generalisation error.
        """
        if (
            self.model is not None
            and self.X_train is not None
            and self.y_train is not None
            and self.X_test is not None
            and self.y_test is not None
        ):
            return self.model.get_generalisation_error(
                self.X_train, self.y_train, self.X_test, self.y_test
            )
        return np.nan  # pragma: no cover

    def save(self, path: str = "target", ext: str = "pkl") -> None:
        """Save the target class to persistent storage.

        Parameters
        ----------
        path : str
            Name of the output folder to save target information.
        ext : str
            File extension defining the model saved format, e.g., "pkl" or "sav".
        """
        norm_path: str = os.path.normpath(path)
        filename: str = os.path.normpath(f"{norm_path}/target.yaml")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # convert Target to dict
        target: dict = {
            "dataset_name": self.dataset_name,
            "dataset_module_path": self.dataset_module_path,
            "n_samples": self.n_samples,
            "features": self.features,
            "n_features": self.n_features,
            "n_samples_orig": self.n_samples_orig,
            "generalisation_error": self._ge(),
            "safemodel": self.safemodel,
        }
        # write model details
        self._save_model(norm_path, ext, target)
        # write data arrays and add paths
        self._save_data(norm_path, target)

        # write yaml
        with open(filename, "w", encoding="utf-8") as fp:
            yaml.dump(target, fp, default_flow_style=False, sort_keys=False)

    def load(self, path: str = "target") -> None:
        """Load the target class from persistent storage.

        Parameters
        ----------
        path : str
            Name of the output folder containing a target yaml file.
        """
        target: dict = {}
        # load yaml
        filename: str = os.path.normpath(f"{path}/target.yaml")
        with open(filename, encoding="utf-8") as fp:
            target = yaml.safe_load(fp)
        # load modules
        if "dataset_module_path" in target:
            self.dataset_module_path = os.path.normpath(
                f"{path}/{target['dataset_module_path']}"
            )
        # load parameters
        if "dataset_name" in target:
            self.dataset_name = target["dataset_name"]
            logger.info("dataset_name: %s", self.dataset_name)
        if "n_samples" in target:
            self.n_samples = target["n_samples"]
        if "features" in target:
            features: dict = target["features"]
            # convert str keys to int
            self.features = {int(key): value for key, value in features.items()}
        if "n_features" in target:
            self.n_features = target["n_features"]
            logger.info("n_features: %d", self.n_features)
        if "n_samples_orig" in target:
            self.n_samples_orig = target["n_samples_orig"]
        if "safemodel" in target:
            self.safemodel = target["safemodel"]

        # load model
        self._load_model(path, target)
        # load data
        self._load_data(path, target)

    def add_safemodel_results(self, data: list) -> None:
        """Add the results of safemodel disclosure checking.

        Parameters
        ----------
        data : list
            The results of safemodel disclosure checking.
        """
        self.safemodel = data

    def has_model(self) -> bool:
        """Return whether the target has a loaded model."""
        return self.model is not None and self.model.model is not None

    def has_data(self) -> bool:
        """Return whether the target has all processed data."""
        return (
            self.X_train is not None
            and self.y_train is not None
            and self.X_test is not None
            and self.y_test is not None
        )

    def has_raw_data(self) -> bool:
        """Return whether the target has all raw data."""
        return (
            self.X_orig is not None
            and self.y_orig is not None
            and self.X_train_orig is not None
            and self.y_train_orig is not None
            and self.X_test_orig is not None
            and self.y_test_orig is not None
        )

    def has_probas(self) -> bool:
        """Return whether the target has all probability data."""
        return self.proba_train is not None and self.proba_test is not None


def get_array_pkl(path: str, name: str):  # pragma: no cover
    """Load a data array from pickle."""
    try:
        with open(path, "rb") as fp:
            arr = pickle.load(fp)
            try:
                logger.info("%s shape: %s", name, arr.shape)
            except AttributeError:
                logger.info("%s is a scalar value.", name)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Pickle file not found: {path}") from e
    except Exception as e:
        raise ValueError(f"Error loading pickle file {path}: {e}") from None
    return arr


def get_array_csv(path: str, name: str):  # pragma: no cover
    """Load a data array from csv."""
    try:
        arr = pd.read_csv(path, header=None).values
        logger.info("%s shape: %s", name, arr.shape)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {path}") from e
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"CSV file is empty: {path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error reading CSV file {path}: {e}") from None
    return arr
