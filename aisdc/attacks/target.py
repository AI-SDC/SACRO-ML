"""Target.py - information storage about the target model and data."""

from __future__ import annotations

import json
import logging
import os
import pickle

import numpy as np
import sklearn

from aisdc.attacks.report import NumpyArrayEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("target")


class Target:  # pylint: disable=too-many-instance-attributes
    """Stores information about the target model and data."""

    def __init__(self, model: sklearn.base.BaseEstimator | None = None) -> None:
        """Store information about a target model and associated data.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator | None
            Trained target model. Any class that implements the
            sklearn.base.BaseEstimator interface (i.e. has fit, predict and
            predict_proba methods)

        Attributes
        ----------
        name : str
            The name of the dataset.
        n_samples : int
            The total number of samples in the dataset.
        x_train : np.ndarray
            The (processed) training inputs.
        y_train : np.ndarray
            The (processed) training outputs.
        x_test : np.ndarray
            The (processed) testing inputs.
        y_test : np.ndarray
            The (processed) testing outputs.
        features : dict
            Dictionary describing the dataset features.
        n_features : int
            The total number of features.
        x_orig : np.ndarray
            The original (unprocessed) dataset inputs.
        y_orig : np.ndarray
            The original (unprocessed) dataset outputs.
        x_train_orig : np.ndarray
            The original (unprocessed) training inputs.
        y_train_orig : np.ndarray
            The original (unprocessed) training outputs.
        x_test_orig : np.ndarray
            The original (unprocessed) testing inputs.
        y_test_orig : np.ndarray
            The original (unprocessed) testing outputs.
        n_samples_orig : int
            The total number of samples in the original dataset.
        model : sklearn.base.BaseEstimator | None
            The trained model.
        safemodel : list
            The results of safemodel disclosure checks.
        """
        self.name: str = ""
        self.n_samples: int = 0
        self.x_train: np.ndarray
        self.y_train: np.ndarray
        self.x_test: np.ndarray
        self.y_test: np.ndarray
        self.features: dict = {}
        self.n_features: int = 0
        self.x_orig: np.ndarray
        self.y_orig: np.ndarray
        self.x_train_orig: np.ndarray
        self.y_train_orig: np.ndarray
        self.x_test_orig: np.ndarray
        self.y_test_orig: np.ndarray
        self.n_samples_orig: int = 0
        self.model: sklearn.base.BaseEstimator | None = model
        self.safemodel: list = []

    def add_processed_data(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Add a processed and split dataset."""
        self.x_train = x_train
        self.y_train = np.array(y_train, int)
        self.x_test = x_test
        self.y_test = np.array(y_test, int)
        self.n_samples = len(x_train) + len(x_test)

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
        x_orig: np.ndarray,
        y_orig: np.ndarray,
        x_train_orig: np.ndarray,
        y_train_orig: np.ndarray,
        x_test_orig: np.ndarray,
        y_test_orig: np.ndarray,
    ) -> None:
        """Add original unprocessed dataset."""
        self.x_orig = x_orig
        self.y_orig = y_orig
        self.x_train_orig = x_train_orig
        self.y_train_orig = y_train_orig
        self.x_test_orig = x_test_orig
        self.y_test_orig = y_test_orig
        self.n_samples_orig = len(x_orig)

    def __save_model(self, path: str, ext: str, target: dict) -> None:
        """Save the target model.

        Parameters
        ----------
        path : str
            Path to write the model.
        ext : str
            File extension defining the model saved format, e.g., "pkl" or "sav".
        target : dict
            Target class as a dictionary for writing JSON.
        """
        # write model
        filename: str = os.path.normpath(f"{path}/model.{ext}")
        if hasattr(self.model, "save"):
            self.model.save(filename)
        elif ext == "pkl":
            with open(filename, "wb") as fp:
                pickle.dump(self.model, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported file format for saving a model: {ext}")
        target["model_path"] = f"model.{ext}"
        # write hyperparameters
        try:
            target["model_name"] = type(self.model).__name__
            target["model_params"] = self.model.get_params()
        except Exception:  # pragma: no cover pylint: disable=broad-exception-caught
            pass

    def __load_model(self, path: str, target: dict) -> None:
        """Load the target model.

        Parameters
        ----------
        path : str
            Path to load the model.
        target : dict
            Target class as a dictionary read from JSON.
        """
        model_path = os.path.normpath(f"{path}/{target['model_path']}")
        _, ext = os.path.splitext(model_path)
        if ext == ".pkl":
            with open(model_path, "rb") as fp:
                self.model = pickle.load(fp)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported file format for loading a model: {ext}")

    def __save_numpy(self, path: str, target: dict, name: str) -> None:
        """Save a numpy array variable as pickle.

        Parameters
        ----------
        path : str
            Path to save the data.
        target : dict
            Target class as a dictionary for writing JSON.
        name : str
            Name of the numpy array to save.
        """
        if hasattr(self, name):
            np_path: str = os.path.normpath(f"{path}/{name}.pkl")
            target[f"{name}_path"] = f"{name}.pkl"
            with open(np_path, "wb") as fp:
                pickle.dump(getattr(self, name), fp, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_numpy(self, path: str, target: dict, name: str) -> None:
        """Load a numpy array variable from pickle.

        Parameters
        ----------
        path : str
            Path to load the data.
        target : dict
            Target class as a dictionary read from JSON.
        name : str
            Name of the numpy array to load.
        """
        key: str = f"{name}_path"
        if key in target:
            np_path: str = os.path.normpath(f"{path}/{target[key]}")
            with open(np_path, "rb") as fp:
                arr = pickle.load(fp)
                setattr(self, name, arr)

    def __save_data(self, path: str, target: dict) -> None:
        """Save the target model data.

        Parameters
        ----------
        path : str
            Path to save the data.
        target : dict
            Target class as a dictionary for writing JSON.
        """
        self.__save_numpy(path, target, "x_train")
        self.__save_numpy(path, target, "y_train")
        self.__save_numpy(path, target, "x_test")
        self.__save_numpy(path, target, "y_test")
        self.__save_numpy(path, target, "x_orig")
        self.__save_numpy(path, target, "y_orig")
        self.__save_numpy(path, target, "x_train_orig")
        self.__save_numpy(path, target, "y_train_orig")
        self.__save_numpy(path, target, "x_test_orig")
        self.__save_numpy(path, target, "y_test_orig")

    def __load_data(self, path: str, target: dict) -> None:
        """Load the target model data.

        Parameters
        ----------
        path : str
            Path to load the data.
        target : dict
            Target class as a dictionary read from JSON.
        """
        self.__load_numpy(path, target, "x_train")
        self.__load_numpy(path, target, "y_train")
        self.__load_numpy(path, target, "x_test")
        self.__load_numpy(path, target, "y_test")
        self.__load_numpy(path, target, "x_orig")
        self.__load_numpy(path, target, "y_orig")
        self.__load_numpy(path, target, "x_train_orig")
        self.__load_numpy(path, target, "y_train_orig")
        self.__load_numpy(path, target, "x_test_orig")
        self.__load_numpy(path, target, "y_test_orig")

    def __ge(self) -> str:
        """Returns the model generalisation error.

        Returns
        -------
        str
            Generalisation error.
        """
        if (
            hasattr(self.model, "score")
            and hasattr(self, "x_train")
            and hasattr(self, "y_train")
            and hasattr(self, "x_test")
            and hasattr(self, "y_test")
        ):
            try:
                train = self.model.score(self.x_train, self.y_train)
                test = self.model.score(self.x_test, self.y_test)
                return str(test - train)
            except sklearn.exceptions.NotFittedError:
                return "not fitted"
        return "unknown"

    def save(self, path: str = "target", ext: str = "pkl") -> None:
        """Saves the target class to persistent storage.

        Parameters
        ----------
        path : str
            Name of the output folder to save target information.
        ext : str
            File extension defining the model saved format, e.g., "pkl" or "sav".
        """
        path: str = os.path.normpath(path)
        filename: str = os.path.normpath(f"{path}/target.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # convert Target to JSON
        target: dict = {
            "data_name": self.name,
            "n_samples": self.n_samples,
            "features": self.features,
            "n_features": self.n_features,
            "n_samples_orig": self.n_samples_orig,
            "generalisation_error": self.__ge(),
            "safemodel": self.safemodel,
        }
        # write model and add path to JSON
        if self.model is not None:
            self.__save_model(path, ext, target)
        # write data arrays and add paths to JSON
        self.__save_data(path, target)
        # write JSON
        with open(filename, "w", newline="", encoding="utf-8") as fp:
            json.dump(target, fp, indent=4, cls=NumpyArrayEncoder)

    def load(self, path: str = "target") -> None:
        """Loads the target class from persistent storage.

        Parameters
        ----------
        path : str
            Name of the output folder containing a target JSON file.
        """
        target: dict = {}
        # load JSON
        filename: str = os.path.normpath(f"{path}/target.json")
        with open(filename, encoding="utf-8") as fp:
            target = json.load(fp)
        # load parameters
        if "data_name" in target:
            self.name = target["data_name"]
        if "n_samples" in target:
            self.n_samples = target["n_samples"]
        if "features" in target:
            features: dict = target["features"]
            # convert str keys to int
            self.features = {int(key): value for key, value in features.items()}
        if "n_features" in target:
            self.n_features = target["n_features"]
        if "n_samples_orig" in target:
            self.n_samples_orig = target["n_samples_orig"]
        if "safemodel" in target:
            self.safemodel = target["safemodel"]
        # load model
        if "model_path" in target:
            self.__load_model(path, target)
        # load data
        self.__load_data(path, target)

    def add_safemodel_results(self, data: list) -> None:
        """Adds the results of safemodel disclosure checking.

        Parameters
        ----------
        data : list
            The results of safemodel disclosure checking.
        """
        self.safemodel = data

    def __str__(self):
        return self.name
