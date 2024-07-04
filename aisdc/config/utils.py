"""Utilities for config prompt generation."""

from __future__ import annotations

import os
import shutil

yes: list[str] = ["yes", "y", "yeah", "yep", "sure", "ok"]
no: list[str] = ["no", "n", "nope", "nah"]


def get_bool(prompt: str) -> bool:
    """Get a Boolean response to a prompt.

    Parameters
    ----------
    prompt : str
        Message to display to user.

    Returns
    -------
    bool
        Whether the user responded yes or no.
    """
    while True:
        response = input(prompt + " (yes/no): ").strip().lower()
        if response in yes:
            return True
        if response in no:
            return False
        print("Invalid input. Please enter 'yes' or 'no'.")


def check_dir(path: str) -> str:
    """Check directory exists and create as necessary.

    Parameters
    ----------
    path : str
        Directory to check exists.

    Returns
    -------
    str
        Directory, possibly changed by user if pre-existing.
    """
    if os.path.isdir(path):
        print(f"Directory '{path}' already exists.")
        resp = get_bool(
            "Continue using this directory and delete its current contents?"
        )
        if resp:
            shutil.rmtree(path)
        else:
            path = input("Specify an alternative directory: ")
            return check_dir(path)
    return path
