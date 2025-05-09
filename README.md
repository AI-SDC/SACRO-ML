# SACRO-ML

[![DOI](https://zenodo.org/badge/518801511.svg)](https://zenodo.org/badge/latestdoi/518801511)
[![PyPI package](https://img.shields.io/pypi/v/sacroml.svg)](https://pypi.org/project/sacroml)
[![Conda](https://img.shields.io/conda/vn/conda-forge/sacroml.svg)](https://github.com/conda-forge/sacroml-feedstock)
[![Python versions](https://img.shields.io/pypi/pyversions/sacroml.svg)](https://pypi.org/project/sacroml)
[![codecov](https://codecov.io/gh/AI-SDC/SACRO-ML/branch/main/graph/badge.svg?token=AXX2XCXUNU)](https://codecov.io/gh/AI-SDC/SACRO-ML)

A collection of tools and resources for managing the [statistical disclosure control](https://en.wikipedia.org/wiki/Statistical_disclosure_control) of trained [machine learning](https://en.wikipedia.org/wiki/Machine_learning) models. For a brief introduction, see [Smith et al. (2022)](https://doi.org/10.48550/arXiv.2212.01233).

The `sacroml` package provides:
* A variety of privacy attacks for assessing machine learning models.
* The safemodel package: a suite of open source wrappers for common machine learning frameworks, including [scikit-learn](https://scikit-learn.org) and [Keras](https://keras.io). It is designed for use by researchers in Trusted Research Environments (TREs) where disclosure control methods must be implemented. Safemodel aims to give researchers greater confidence that their models are more compliant with disclosure control.

## Installation

### Python Package Index

```
$ pip install sacroml
```

Note: macOS users may need to install libomp due to a dependency on XGBoost:
```
$ brew install libomp
```

### Conda

```
$ conda install sacroml
```

## Usage

See the [examples](examples/).

## Documentation

See [API documentation](https://ai-sdc.github.io/SACRO-ML/).

## Contributing

See our [contributing](CONTRIBUTING.md) guide.

## Acknowledgement

This work was supported by UK Research and Innovation as part of the Data and Analytics Research Environments UK ([DARE UK](https://dareuk.org.uk)) programme, delivered in partnership with Health Data Research UK (HDR UK) and Administrative Data Research UK (ADR UK). The specific projects were Semi-Automated Checking of Research Outputs ([SACRO](https://gtr.ukri.org/projects?ref=MC_PC_23006); MC_PC_23006), Guidelines and Resources for AI Model Access from TrusTEd Research environments ([GRAIMATTER](https://gtr.ukri.org/projects?ref=MC_PC_21033); MC_PC_21033), and [TREvolution](https://dareuk.org.uk/trevolution) (MC_PC_24038). This project has also been supported by MRC and EPSRC ([PICTURES](https://gtr.ukri.org/projects?ref=MR%2FS010351%2F1); MR/S010351/1).

<img src="docs/source/images/UK_Research_and_Innovation_logo.svg" width="20%" height="20%" padding=20/> <img src="docs/source/images/health-data-research-uk-hdr-uk-logo-vector.png" width="10%" height="10%" padding=20/> <img src="docs/source/images/logo_print.png" width="15%" height="15%" padding=20/>
