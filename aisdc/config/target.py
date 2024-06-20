"""Prompt to generate valid target config."""

from __future__ import annotations

import os
import sys

from aisdc.attacks.target import Target
from aisdc.config import utils

arrays_pro = ["X_train", "y_train", "X_test", "y_test"]
arrays_raw = ["X", "y", "X_train_orig", "y_train_orig", "X_test_orig", "y_test_orig"]
arrays_proba = ["y_proba_train", "y_proba_test"]


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


def _get_n_features(target: Target) -> None:
    """Prompt user for the number of dataset features."""
    n_features = input("How many features were presented to the model: ")
    target.n_features = int(n_features) if n_features != "" else 0


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
    if not utils.get_bool("Do you have a the model predicted probabilities?"):
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
        _get_proba(target)

    _get_dataset_name(target)
    _get_n_features(target)
    if utils.get_bool("Do you know the paths to processed data?"):
        _get_arrays(target, arrays_pro)
    if utils.get_bool("Do you know the paths to original raw data?"):
        _get_arrays(target, arrays_raw)
    target.save(path)
    print(f"Target generated in directory: '{path}'")
