"""Python setup script for installing AI-SDC."""

from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

ATTACKS_REQS = [
    "acro==0.4.5",
    "fpdf",
    "pypdf",
    "multiprocess",
    "scikit-learn",
    "xgboost",
]

SAFEMODEL_REQS = [
    "tensorflow>=2.14,<2.15",
    "tensorflow-privacy==0.9.0",
    "dictdiffer",
]

DOC_REQS = [
    "numpydoc",
    "sphinx-autopackagesummary",
    "sphinx-issues",
    "sphinx-prompt",
    "sphinx-rtd-theme",
    "sphinx",
]

TEST_REQS = [
    "pytest",
    "pytest-cov",
]

setup(
    name="aisdc",
    version="1.1.3",
    license="MIT",
    maintainer="Jim Smith",
    maintainer_email="james.smith@uwe.ac.uk",
    description="Tools for the statistical disclosure control of machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AI-SDC/AI-SDC",
    packages=find_packages(exclude=["tests"]),
    package_data={"aisdc.safemodel": ["rules.json"]},
    python_requires=">=3.9,<3.12",
    install_requires=ATTACKS_REQS,
    extras_require={
        "safemodel": SAFEMODEL_REQS,
        "doc": SAFEMODEL_REQS + DOC_REQS,
        "test": SAFEMODEL_REQS + TEST_REQS,
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "data-privacy",
        "data-protection",
        "machine-learning",
        "privacy",
        "privacy-tools",
        "statistical-disclosure-control",
    ],
)
