[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://opensource.org/licenses/MIT)
[![Latest Version](https://img.shields.io/github/v/release/AI-SDC/AI-SDC?style=flat)](https://github.com/AI-SDC/AI-SDC/releases)
[![DOI](https://zenodo.org/badge/518801511.svg)](https://zenodo.org/badge/latestdoi/518801511)
[![codecov](https://codecov.io/gh/AI-SDC/AI-SDC/branch/development/graph/badge.svg?token=AXX2XCXUNU)](https://codecov.io/gh/AI-SDC/AI-SDC)
[![Python versions](https://img.shields.io/pypi/pyversions/aisdc.svg)](https://pypi.org/project/aisdc)

# AI-SDC

A collection of tools and resources for managing the statistical disclosure control of trained machine learning models. For a brief introduction, see [Smith et al. (2022)](https://doi.org/10.48550/arXiv.2212.01233).

## Content

* `aisdc`
    - `attacks` Contains a variety of privacy attacks on machine learning models, including membership and attribute inference.
    - `preprocessing` Contains preprocessing modules for test datasets.
    - `safemodel` The safemodel package is an open source wrapper for common machine learning models. It is designed for use by researchers in Trusted Research Environments (TREs) where disclosure control methods must be implemented. Safemodel aims to give researchers greater confidence that their models are more compliant with disclosure control.
* `docs` Contains Sphinx documentation files.
* `example_notebooks` Contains short tutorials on the basic concept of "safe_XX" versions of machine learning algorithms, and examples of some specific algorithms.
* `examples` Contains examples of how to run the code contained in this repository:
  - How to simulate attribute inference attacks `attribute_inference_example.py`.
  - How to simulate membership inference attacks:
    + Worst case scenario attack `worst_case_attack_example.py`.
    + LIRA scenario attack `lira_attack_example.py`.
  - Integration of attacks into safemodel classes `safemodel_attack_integration_bothcalls.py`.
* `risk_examples` Contains hypothetical examples of data leakage through machine learning models as described in the [Green Paper](https://doi.org/10.5281/zenodo.6896214).
* `tests` Contains unit tests.

## Documentation

Documentation is hosted here: https://ai-sdc.github.io/AI-SDC/

## Quick Start

### Development

Clone the repository and install the dependencies (safest in a virtual env):

```
$ git clone https://github.com/AI-SDC/AI-SDC.git
$ cd AI-SDC
$ pip install -r requirements.txt
```

Then run the tests:

```
$ pip install pytest
$ pytest .
```

Or run an example:

```
$ python -m examples.lira_attack_example
```

### Installation / End-user

[![PyPI package](https://img.shields.io/pypi/v/aisdc.svg)](https://pypi.org/project/aisdc)

Install `aisdc` (safest in a virtual env) and manually copy the `examples` and `example_notebooks`.

```
$ pip install aisdc
```

Then to run an example:

```
$ python attribute_inference_example.py
```

Or start up `jupyter notebook` and run an example.

Alternatively, you can clone the repo and install:

```
$ git clone https://github.com/AI-SDC/AI-SDC.git
$ cd AI-SDC
$ pip install .
```

---

This work was funded by UK Research and Innovation Grant Number MC_PC_21033 as part of Phase 1 of the DARE UK (Data and Analytics Research Environments UK) programme (https://dareuk.org.uk/), delivered in partnership with HDR UK and ADRUK. The specific project was Guidelines and Resources for AI Model Access from TrusTEd Research environments (GRAIMATTER).­ This project has also been supported by MRC and EPSRC [grant number MR/S010351/1]: PICTURES.

<img src="docs/source/images/UK_Research_and_Innovation_logo.svg" width="20%" height="20%" padding=20/> <img src="docs/source/images/health-data-research-uk-hdr-uk-logo-vector.png" width="10%" height="10%" padding=20/> <img src="docs/source/images/logo_print.png" width="15%" height="15%" padding=20/>
