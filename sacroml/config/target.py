"""Prompt to generate valid target config."""

from __future__ import annotations

import ast
import os
import sys

from sacroml.attacks.target import Target
from sacroml.config import utils

arrays_pro = ["X_train", "y_train", "X_test", "y_test"]
arrays_raw = ["X", "y", "X_train_orig", "y_train_orig", "X_test_orig", "y_test_orig"]
arrays_proba = ["proba_train", "proba_test"]
encodings = ["onehot", "str", "int", "float"]

MAX_FEATURES = 64  # maximum features to prompt


def _get_arrays(target: Target, arrays: list[str]) -> None:
    """Prompt user for the paths to array data."""
    for arr in arrays:
        while True:
            path = input(f"What is the path to {arr}? ")
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
    target.dataset_name = input("What is the name of the dataset? ")


def _get_feature_encoding(feat: int) -> str:
    """Prompt user for feature encoding."""
    while True:
        encoding = input(f"What is the encoding of feature {feat}? ")
        if encoding in encodings:
            return encoding
        print("Invalid encoding. Please try again.")


def _get_feature_indices(feat: int) -> list[int]:
    """Prompt user for feature indices."""
    while True:
        indices = input(f"What are the indices for feature {feat}, e.g., [1,2,3]? ")
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
    n_features = input("How many features does this dataset have? ")
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
            name = input(f"What is the name of feature {i}? ")
            indices = _get_feature_indices(i)
            encoding = _get_feature_encoding(i)
            target.add_feature(name, indices, encoding)


def _get_model_path() -> str:
    """Prompt user for path to a saved fitted model."""
    if not utils.get_bool("Do you have a saved fitted model?"):
        print("Cannot generate a target config without a fitted model.")
        sys.exit()

    while True:
        path = input("Enter path including the full filename: ")
        if os.path.isfile(path):
            break
        print("File does not exist. Please try again.")
    return path


def _get_proba(target: Target) -> None:
    """Prompt user for model predicted probabilities."""
    if not utils.get_bool("Do you have the model predicted probabilities?"):
        print("Cannot run any attacks without a supported model or probabilities.")
        sys.exit()
    _get_arrays(target, arrays_proba)


def prompt_for_target() -> None:
    """Prompt user for information to generate target config."""
    target = Target()
    path: str = utils.check_dir("target")

    model_path: str = _get_model_path()
    try:  # attempt to load saved model
        target.load_model(model_path)
    except ValueError:  # unsupported model, require probas
        print("Unable to load model.")
        _get_proba(target)

    _get_dataset_name(target)
    if utils.get_bool("Do you know the paths to processed data?"):
        _get_arrays(target, arrays_pro)
        _get_features(target)
    if utils.get_bool("Do you know the paths to original raw data?"):
        _get_arrays(target, arrays_raw)
    target.save(path)
    print(f"Target generated in directory: '{path}'")
