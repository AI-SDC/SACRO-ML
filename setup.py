"""Python setup script for installing AI-SDC."""

from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="aisdc",
    version="1.1.2",
    license="MIT",
    maintainer="Jim Smith",
    maintainer_email="james.smith@uwe.ac.uk",
    description="Tools for the statistical disclosure control of machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AI-SDC/AI-SDC",
    packages=find_packages(exclude=["tests"]),
    package_data={"aisdc.safemodel": ["rules.json"]},
    python_requires=">=3.8,<3.11",
    install_requires=[
        "dictdiffer~=0.9.0",
        "fpdf~=1.7.2",
        "joblib~=1.3",
        "multiprocess~=0.70.15",
        "numpy~=1.23.1",
        "pandas~=1.5.3",
        "scikit_learn~=1.1.3",
        "scipy~=1.10",
        "tensorflow-probability==0.15.0; python_version < '3.9'",
        "tensorflow~=2.10.0; python_version < '3.9'",
        "tensorflow~=2.12.0; python_version >= '3.9'",
        "tensorflow_privacy==0.7.3; python_version < '3.9'",
        "tensorflow_privacy==0.8.10; python_version >= '3.9'",
        "uuid~=1.30",
        "pypdf~=3.16",
        "xgboost~=1.7.6",
        "statsmodels==0.13.5",
        "acro==0.4.3",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
