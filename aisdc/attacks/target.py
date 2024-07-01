"""Store information about the target model and data."""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import sklearn
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Target:  # pylint: disable=too-many-instance-attributes
    """Store information about the target model and data."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        model: sklearn.base.BaseEstimator | None = None,
        dataset_name: str = "",
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
        model : sklearn.base.BaseEstimator | None, optional
            Trained target model. Any class that implements the
            sklearn.base.BaseEstimator interface (i.e. has fit, predict and
            predict_proba methods)
        dataset_name : str
            The name of the dataset.
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
        self.model: sklearn.base.BaseEstimator | None = model
        self.model_name: str = "unknown"
        self.model_params: dict = {}
        if self.model is not None:
            self.model_name = type(self.model).__name__
            self.model_params = self.model.get_params()
        # Model - predicted probabilities
        self.proba_train: np.ndarray | None = proba_train
        self.proba_test: np.ndarray | None = proba_test
        #  Dataset - details
        self.dataset_name: str = dataset_name
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
        target["model_name"] = self.model_name
        target["model_params"] = self.model_params

    def load_model(self, model_path: str) -> None:
        """Load the target model.

        Parameters
        ----------
        model_path : str
            Path to load the model.
        """
        path = os.path.normpath(model_path)
        _, ext = os.path.splitext(path)
        if ext == ".pkl":
            with open(path, "rb") as fp:
                self.model = pickle.load(fp)
                model_type = type(self.model)
                logger.info("Loaded: %s", model_type.__name__)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported file format for loading a model: {ext}")

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

        Parameters
        ----------
        arr_path : str
            Filename of a data array.
        name : str
            Name of the data array to load.
        """
        path = os.path.normpath(arr_path)
        with open(path, "rb") as fp:
            _, ext = os.path.splitext(path)
            if ext == ".pkl":
                arr = pickle.load(fp)
                setattr(self, name, arr)
                logger.info("%s shape: %s", name, arr.shape)
            else:
                raise ValueError(f"Target cannot load {ext} files.")

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

    def _ge(self) -> float:
        """Return the model generalisation error.

        Returns
        -------
        float
            Generalisation error.
        """
        if (
            hasattr(self.model, "score")
            and self.X_train is not None
            and self.y_train is not None
            and self.X_test is not None
            and self.y_test is not None
        ):
            try:
                train = self.model.score(self.X_train, self.y_train)
                test = self.model.score(self.X_test, self.y_test)
                return test - train
            except sklearn.exceptions.NotFittedError:
                return np.NaN
        return np.NaN

    def save(self, path: str = "target", ext: str = "pkl") -> None:
        """Save the target class to persistent storage.

        Parameters
        ----------
        path : str
            Name of the output folder to save target information.
        ext : str
            File extension defining the model saved format, e.g., "pkl" or "sav".
        """
        path: str = os.path.normpath(path)
        filename: str = os.path.normpath(f"{path}/target.yaml")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # convert Target to dict
        target: dict = {
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "features": self.features,
            "n_features": self.n_features,
            "n_samples_orig": self.n_samples_orig,
            "generalisation_error": self._ge(),
            "safemodel": self.safemodel,
        }
        # write model and add path
        if self.model is not None:
            self._save_model(path, ext, target)
        # write data arrays and add paths
        self._save_data(path, target)
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
        if "model_name" in target:
            self.model_name = target["model_name"]
        if "model_params" in target:
            self.model_params = target["model_params"]
        if "model_path" in target:
            model_path = os.path.normpath(f"{path}/{target['model_path']}")
            self.load_model(model_path)
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

    def __str__(self) -> str:
        """Return the name of the dataset used."""
        return self.dataset_name
