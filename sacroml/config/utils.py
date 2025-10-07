"""Utilities for config prompt generation."""

from __future__ import annotations

import ast
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
        resp = get_bool("Keep using this directory and delete its current contents?")
        if resp:
            shutil.rmtree(path)
        else:
            path = input("Specify an alternative directory: ")
            return check_dir(path)

    os.makedirs(path, exist_ok=True)
    return path


def get_class_names(path: str) -> list[str]:
    """Return a list of class names given a Python file.

    Parameters
    ----------
    path : str
        Path to a Python (.py) file.

    Returns
    -------
    list[str]
        List of class names.
    """
    classes: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            contents = f.read()
        tree = ast.parse(contents)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
    except FileNotFoundError:
        print(f"Error: File not found at '{path}'")
    return classes
