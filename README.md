# AI-SDC
Collection of tools and resources for managing the statistical disclosure control of trained machine learning models

Documentation is hosted here: https://ai-sdc.github.io/AI-SDC/

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://opensource.org/licenses/MIT)
[![Latest Version](https://img.shields.io/github/v/release/AI-SDC/AI-SDC?style=flat)](https://github.com/AI-SDC/AI-SDC/releases)
[![DOI](https://zenodo.org/badge/518801511.svg)](https://zenodo.org/badge/latestdoi/518801511)

---
# Content

The two main elements are contained in "attacks" and "safemodels".

## example_notebooks

Contains short tutorials on  the basic concept of 'safe_XX' versions of Machine Learning algorithms, and examples of some specific algorithms.

## risk_examples

Contain hypothetical examples of data leakage through machine learning models as described in the Green Paper [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6896214.svg)](https://doi.org/10.5281/zenodo.6896214)

## Examples

Contain examples of code on how to run the code contained in this repository:
- How to simulate attribute inference attacks (attribute_inference_example.py).
- How to simulate membership inference attacks:
  - Worst case scenario attack (worst_case_attack_example.py)
  - LIRA scenario attack (lira_attack_example.py).
- Integration of attacks into safemodel classes (safemodel_attack_integration_bothcalls.py).


---

This work was funded by UK Research and Innovation Grant Number MC_PC_21033 as part of Phase 1 of the DARE UK (Data and Analytics Research Environments UK) programme (https://dareuk.org.uk/), delivered in partnership with HDR UK and ADRUK. The specific project was Guidelines and Resources for AI Model Access from TrusTEd Research environments (GRAIMATTER).­ This project has also been supported by MRC and EPSRC [grant number MR/S010351/1]: PICTURES.

<img src="docs/source/images/UK_Research_and_Innovation_logo.svg" width="20%" height="20%" padding=20/> <img src="docs/source/images/health-data-research-uk-hdr-uk-logo-vector.png" width="10%" height="10%" padding=20/> <img src="docs/source/images/logo_print.png" width="15%" height="15%" padding=20/>
