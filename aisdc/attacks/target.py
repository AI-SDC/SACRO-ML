"""target.py - information storage about the target model and data"""

from __future__ import annotations

import json
import logging
import os
import pickle

import numpy as np
import sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("target")


class Target:  # pylint: disable=too-many-instance-attributes
    """Stores information about the target model and data"""

    def __init__(self, model: sklearn.base.BaseEstimator | None = None) -> None:
        """Store information about a target model and associated data.

        Parameters
        ----------
        model: sklearn.base.BaseEstimator | None
            Trained target model. Any class that implements the
            sklearn.base.BaseEstimator interface (i.e. has fit, predict and
            predict_proba methods)
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

    def add_processed_data(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Add a processed and split dataset"""
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
        """Add original unprocessed dataset"""
        self.x_orig = x_orig
        self.y_orig = y_orig
        self.x_train_orig = x_train_orig
        self.y_train_orig = y_train_orig
        self.x_test_orig = x_test_orig
        self.y_test_orig = y_test_orig
        self.n_samples_orig = len(x_orig)

    def __save_model(self, path: str, target: dict) -> None:
        """Save the target model.

        Parameters
        ----------
        path : str
            Path to write the model.

        target : dict
            Target class as a dictionary for writing JSON.
        """
        # write model
        filename: str = os.path.normpath(f"{path}/model.pkl")
        with open(filename, "wb") as fp:
            pickle.dump(self.model, fp, protocol=pickle.HIGHEST_PROTOCOL)
        target["model_path"] = filename
        # write hyperparameters
        try:
            target["model_name"] = type(self.model).__name__
            target["model_params"] = self.model.get_params()
        except Exception:  # pragma: no cover pylint: disable=broad-exception-caught
            pass

    def __load_model(self, target: dict) -> None:
        """Load the target model.

        Parameters
        ----------
        target : dict
            Target class as a dictionary read from JSON.
        """
        model_path = os.path.normpath(target["model_path"])
        with open(model_path, "rb") as fp:
            self.model = pickle.load(fp)

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
            target[f"{name}_path"] = np_path
            with open(np_path, "wb") as fp:
                pickle.dump(getattr(self, name), fp, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_numpy(self, target: dict, name: str) -> None:
        """Load a numpy array variable from pickle.

        Parameters
        ----------
        target : dict
            Target class as a dictionary read from JSON.

        name : str
            Name of the numpy array to load.
        """
        key: str = f"{name}_path"
        if key in target:
            np_path: str = os.path.normpath(target[key])
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

    def __load_data(self, target: dict) -> None:
        """Load the target model data.

        Parameters
        ----------
        target : dict
            Target class as a dictionary read from JSON.
        """
        self.__load_numpy(target, "x_train")
        self.__load_numpy(target, "y_train")
        self.__load_numpy(target, "x_test")
        self.__load_numpy(target, "y_test")
        self.__load_numpy(target, "x_orig")
        self.__load_numpy(target, "y_orig")
        self.__load_numpy(target, "x_train_orig")
        self.__load_numpy(target, "y_train_orig")
        self.__load_numpy(target, "x_test_orig")
        self.__load_numpy(target, "y_test_orig")

    def save(self, name: str = "target") -> None:
        """Saves the target class to persistent storage.

        Parameters
        ----------
        name : str
            Name of the output folder to save target information.
        """
        path: str = os.path.normpath(name)
        try:  # check if the directory was already created
            os.makedirs(path)
            logger.debug("Directory %s created successfully", path)
        except FileExistsError:  # pragma: no cover
            logger.debug("Directory %s already exists", path)
        # convert Target to JSON
        target: dict = {
            "data_name": self.name,
            "n_samples": self.n_samples,
            "features": self.features,
            "n_features": self.n_features,
            "n_samples_orig": self.n_samples_orig,
        }
        # write model and add path to JSON
        if self.model is not None:
            self.__save_model(path, target)
        # write data arrays and add paths to JSON
        self.__save_data(path, target)
        # write JSON
        filename: str = os.path.normpath(f"{path}/target.json")
        with open(filename, "w", newline="", encoding="utf-8") as fp:
            json.dump(target, fp, indent=4, sort_keys=False)

    def load(self, name: str = "target") -> None:
        """Loads the target class from persistent storage.

        Parameters
        ----------
        name : str
            Name of the output folder containing a target JSON file.
        """
        target: dict = {}
        # load JSON
        filename: str = os.path.normpath(f"{name}/target.json")
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
        # load model
        if "model_path" in target:
            self.__load_model(target)
        # load data
        self.__load_data(target)

    def __str__(self):
        return self.name
