[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://opensource.org/licenses/MIT)
[![Latest Version](https://img.shields.io/github/v/release/AI-SDC/AI-SDC?style=flat)](https://github.com/AI-SDC/AI-SDC/releases)
[![DOI](https://zenodo.org/badge/518801511.svg)](https://zenodo.org/badge/latestdoi/518801511)
[![codecov](https://codecov.io/gh/AI-SDC/AI-SDC/branch/main/graph/badge.svg?token=AXX2XCXUNU)](https://codecov.io/gh/AI-SDC/AI-SDC)
[![Python versions](https://img.shields.io/pypi/pyversions/aisdc.svg)](https://pypi.org/project/aisdc)

# AI-SDC

A collection of tools and resources for managing the [statistical disclosure control](https://en.wikipedia.org/wiki/Statistical_disclosure_control) of trained [machine learning](https://en.wikipedia.org/wiki/Machine_learning) models. For a brief introduction, see [Smith et al. (2022)](https://doi.org/10.48550/arXiv.2212.01233).

The `aisdc` package provides:
* A variety of privacy attacks for assessing machine learning models.
* The safemodel package: a suite of open source wrappers for common machine learning frameworks, including [scikit-learn](https://scikit-learn.org) and [Keras](https://keras.io). It is designed for use by researchers in Trusted Research Environments (TREs) where disclosure control methods must be implemented. Safemodel aims to give researchers greater confidence that their models are more compliant with disclosure control.

A collection of user guides can be found in the [`user_stories`](user_stories) folder of this repository. These guides include configurable examples from the perspective of both a researcher and a TRE, with separate scripts for each. Instructions on how to use each of these scripts and which scripts to use are included in the README located in the folder.

## Installation

[![PyPI package](https://img.shields.io/pypi/v/aisdc.svg)](https://pypi.org/project/aisdc)

Install `aisdc` and manually copy the [`examples`](examples/).

To install only the base package, which includes the attacks used for assessing privacy:

```
$ pip install aisdc
```

To additionally install the safemodel package:

```
$ pip install aisdc[safemodel]
```

## Running

To run an example, simply execute the desired script. For example, to run LiRA:

```
$ python -m lira_attack_example
```

## Acknowledgement

This work was funded by UK Research and Innovation under Grant Numbers MC_PC_21033 and MC_PC_23006 as part of Phase 1 of the [DARE UK](https://dareuk.org.uk) (Data and Analytics Research Environments UK) programme, delivered in partnership with Health Data Research UK (HDR UK) and Administrative Data Research UK (ADR UK). The specific projects were Semi-Automatic checking of Research Outputs (SACRO; MC_PC_23006) and Guidelines and Resources for AI Model Access from TrusTEd Research environments (GRAIMATTER; MC_PC_21033).­This project has also been supported by MRC and EPSRC [grant number MR/S010351/1]: PICTURES.

<img src="docs/source/images/UK_Research_and_Innovation_logo.svg" width="20%" height="20%" padding=20/> <img src="docs/source/images/health-data-research-uk-hdr-uk-logo-vector.png" width="10%" height="10%" padding=20/> <img src="docs/source/images/logo_print.png" width="15%" height="15%" padding=20/>
