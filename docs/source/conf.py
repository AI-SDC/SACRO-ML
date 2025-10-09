"""Configuration file for the Sphinx documentation builder."""

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(1, os.path.abspath("../../attacks/"))

# -- Project information -----------------------------------------------------

project = "SACRO-ML"
copyright = "2025, GRAIMATTER and SACRO Project Team"
author = "GRAIMATTER and SACRO Project Team"
release = "1.4.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autopackagesummary",
    "sphinx_issues",
    "sphinx_prompt",
    "pydata_sphinx_theme",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "sphinx_design",
    "myst_parser",
]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {"navigation_depth": 2}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# -- References --------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# -- Notebook configuration --------------------------------------------------

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_timeout = 60

# -- -------------------------------------------------------------------------

numpydoc_class_members_toctree = False
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "groupwise",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
