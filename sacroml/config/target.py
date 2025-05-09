"""Prompt to generate valid target config."""

from __future__ import annotations

import ast
import contextlib
import os
import pickle
import shutil
import sys
from typing import Any

from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter, WordCompleter

from sacroml.attacks.model import create_model
from sacroml.attacks.target import Target, registry
from sacroml.config import utils
from sacroml.version import __version__

arrays_pro = ["X_train", "y_train", "X_test", "y_test"]
arrays_raw = ["X", "y", "X_train_orig", "y_train_orig", "X_test_orig", "y_test_orig"]
arrays_proba = ["proba_train", "proba_test"]
encodings = ["onehot", "str", "int", "float"]

MAX_FEATURES = 64  # maximum features to prompt


def _get_arrays(target: Target, arrays: list[str]) -> None:
    """Prompt user for the paths to array data."""
    for arr in arrays:
        while True:
            msg = f"What is the path to {arr}?: "
            path = prompt(msg, completer=PathCompleter())
            try:
                target.load_array(path, arr)
                break
            except FileNotFoundError:
                print("File does not exist. Please try again.")
            except ValueError as e:
                print(f"WARNING: {e}")
                break


def _get_dataset_name(target: Target) -> None:
    """Prompt user for the name of a dataset."""
    target.dataset_name = prompt("What is the name of the dataset? ")


def _get_feature_encoding(feat: int) -> str:
    """Prompt user for feature encoding."""
    while True:
        encoding = prompt(f"What is the encoding of feature {feat}? ")
        if encoding in encodings:
            return encoding
        print("Invalid encoding. Please try again.")


def _get_feature_indices(feat: int) -> list[int]:
    """Prompt user for feature indices."""
    while True:
        indices = prompt(f"What are the indices for feature {feat}, e.g., [1,2,3]? ")
        try:
            indices = ast.literal_eval(indices)
            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                return indices
            print("Invalid input. Please enter a list of integers like [1,2,3].")
        except (ValueError, SyntaxError):
            print("Invalid input. Please enter a valid list of integers.")


def _get_features(target: Target) -> None:
    """Prompt user for dataset features."""
    print("To run attribute inference attacks the features must be described.")
    n_features = prompt("How many features does this dataset have? ")
    n_features = int(n_features)
    if n_features > MAX_FEATURES:
        print("There are too many features to add via prompt.")
        print("You can edit the 'target.yaml' to add features later.")
        print("Note: this is only necessary for attribute inference.")
        return
    print("The name, index, and encoding are needed for each feature.")
    print("For example: feature 0 = 'parents', '[0, 1, 2]', 'onehot'")
    if utils.get_bool("Do you want to add this information?"):
        print(f"Valid encodings: {', '.join(encodings)}")
        for i in range(n_features):
            name = prompt(f"What is the name of feature {i}? ")
            indices = _get_feature_indices(i)
            encoding = _get_feature_encoding(i)
            target.add_feature(name, indices, encoding)


def _get_model_type() -> str:
    """Prompt user for saved model type."""
    while True:
        models: list[str] = list(registry.keys())
        print(f"Model types natively supported: {models}")
        print("If it is a Pytorch model, type: 'PytorchModel'")
        print("If it is a scikit-learn model, type: 'SklearnModel'")
        completer = WordCompleter(models)
        model_type = prompt("What is the type of model? ", completer=completer)
        if model_type in registry:
            return model_type
        print(f"{model_type} is not loadable so some attacks will be unavailable")
        if utils.get_bool(f"Are you sure {model_type} is correct?"):
            return model_type


def _get_model_name() -> str:
    """Prompt user for saved model class name."""
    return prompt("What is the model class name? ")


def _get_model_module_path() -> str:
    """Prompt user for the Python module containing the model class."""
    while True:
        print("Please provide a Python module containing the model class")
        msg = "Enter the path including the full filename: "
        path = prompt(msg, completer=PathCompleter())
        if os.path.isfile(path):
            break
        print("File does not exist. Please try again.")
    return path


def _get_model_path() -> str:
    """Prompt user for path to a saved fitted model."""
    while True:
        msg = "Enter the saved model path including the full filename: "
        path = prompt(msg, completer=PathCompleter())
        if os.path.isfile(path):
            break
        print("File does not exist. Please try again.")
    return path


def _get_train_module_path() -> str:
    """Prompt user for the Python module containing the train function."""
    while True:
        print("Please provide a Python module containing the train function")
        msg = "Enter the path including the full filename: "
        path = prompt(msg, completer=PathCompleter())
        if os.path.isfile(path):
            break
        print("File does not exist. Please try again.")
    return path


def _get_params() -> dict[str, Any]:
    """Prompt user for hyperparameter names and their values."""
    params: dict[str, Any] = {}
    print("Enter hyperparameter names and values.")
    print("Type 'done' as the name when you're finished.")
    while True:
        name: str = prompt("Hyperparameter name (or 'done' to finish): ").strip()
        if name.lower() == "done":
            break
        if not name:
            print("Name cannot be empty.")
            continue
        value = prompt(f"Value for '{name}': ").strip()
        # Try to infer the type
        if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
        else:
            with contextlib.suppress(ValueError):
                value = float(value) if "." in value else int(value)
        params[name] = value
    return params


def _get_train_params() -> dict:
    """Prompt user for train function hyperparameters."""
    print("Please provide hyperparameters for the train function")
    return _get_params()


def _get_model_params() -> dict:
    """Prompt user for model hyperparameters."""
    print("Please provide hyperparameters for the model constructor")
    return _get_params()


def _get_proba(target: Target) -> None:
    """Prompt user for model predicted probabilities."""
    if not utils.get_bool("Do you have the model predicted probabilities?"):
        print("Cannot run any attacks without a supported model or probabilities.")
        sys.exit()
    _get_arrays(target, arrays_proba)


def _load_model(model_path: str) -> Any:
    """Load a model from a file.

    Currently only supports pickle.

    Parameters
    ----------
    model_path : str
        Path to load the model.

    Returns
    -------
    Any
        Loaded model.
    """
    path = os.path.normpath(model_path)
    _, ext = os.path.splitext(path)
    if ext == ".pkl":
        with open(path, "rb") as fp:
            model = pickle.load(fp)
            model_type = type(model)
            print(f"Successfully loaded: {model_type.__name__}")
            return model
    else:
        raise ValueError(f"Unsupported file format for loading a model: {ext}")


def prompt_for_target() -> None:
    """Prompt user for information to generate target config."""
    print(f"sacroml version {__version__}")

    if not utils.get_bool("Do you have a saved fitted model?"):
        print("Cannot generate a target config without a fitted model.")
        sys.exit()

    # Check target directory exists and create as necessary
    target_dir: str = "target"
    path: str = utils.check_dir(target_dir)

    # Get model and training information
    model_type: str = _get_model_type()
    model_path: str = _get_model_path()

    if model_type == "PytorchModel":
        model_module_path: str = _get_model_module_path()
        model_name: str = _get_model_name()
        model_params: dict = _get_model_params()
        train_module_path: str = _get_train_module_path()
        train_params: dict = _get_train_params()
        try:  # Create a new model
            model = create_model(model_module_path, model_name, model_params)
            print("Successfully created a new model using supplied class.")
        except ValueError as e:
            print(f"{str(e)}")
            print("Please try again.")
            sys.exit()
        # Create a new target object
        target = Target(
            model=model,
            model_path=model_path,
            model_module_path=model_module_path,
            model_name=model_name,
            model_params=model_params,
            train_module_path=train_module_path,
            train_params=train_params,
        )
    else:
        try:  # Load the saved model
            model = _load_model(model_path)
            target = Target(model=model)
        except ValueError:  # Unsupported model
            print("Unable to load model.")
            # Copy the model file to the target directory
            filename = os.path.basename(model_path)
            dest_path = os.path.join(target_dir, filename)
            shutil.copy2(model_path, dest_path)
            # Get predicted probabilities
            target = Target(model=None)
            _get_proba(target)

    # Get dataset information
    _get_dataset_name(target)
    if utils.get_bool("Do you know the paths to processed data?"):
        _get_arrays(target, arrays_pro)
        _get_features(target)
    if utils.get_bool("Do you know the paths to original raw data?"):
        _get_arrays(target, arrays_raw)

    # Save the target information to the target directory
    target.save(path)
    print(f"Target generated in directory: '{path}'")
