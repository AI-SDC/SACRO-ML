"""Configuration file for the Sphinx documentation builder."""

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(1, os.path.abspath("../../attacks/"))

# -- Project information -----------------------------------------------------

project = "AI-SDC"
copyright = "2024, GRAIMATTER and SACRO Project Team"
author = "GRAIMATTER and SACRO Project Team"
release = "1.2.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "numpydoc",
    "sphinx-prompt",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autopackagesummary",
    "sphinx_issues",
    "sphinx_rtd_theme",
]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 2}
html_static_path = ["_static"]

# -- -------------------------------------------------------------------------

numpydoc_class_members_toctree = False
