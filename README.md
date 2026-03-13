# SACRO-ML: Disclosure Control Tools for ML Models

[![arXiv](https://img.shields.io/badge/arXiv-2212.01233-b31b1b.svg)](https://arxiv.org/abs/2212.01233)
[![PyPI package](https://img.shields.io/pypi/v/sacroml.svg)](https://pypi.org/project/sacroml)
[![Conda](https://img.shields.io/conda/vn/conda-forge/sacroml.svg)](https://github.com/conda-forge/sacroml-feedstock)
[![Python versions](https://img.shields.io/pypi/pyversions/sacroml.svg)](https://pypi.org/project/sacroml)
[![codecov](https://codecov.io/gh/AI-SDC/SACRO-ML/branch/main/graph/badge.svg?token=AXX2XCXUNU)](https://codecov.io/gh/AI-SDC/SACRO-ML)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AI-SDC/SACRO-ML)

An increasing body of work has shown that [machine learning](https://en.wikipedia.org/wiki/Machine_learning) (ML) models may expose confidential properties of the data on which they are trained. This has resulted in a wide range of proposed attack methods with varying assumptions that exploit the model structure and/or behaviour to infer sensitive information.

The `sacroml` package is a collection of tools and resources for managing the [statistical disclosure control](https://en.wikipedia.org/wiki/Statistical_disclosure_control) (SDC) of trained ML models. In particular, it provides:

* A **safemodel** package that extends commonly used ML models to provide *ante-hoc* SDC by assessing the theoretical risk posed by the training regime (such as hyperparameter, dataset, and architecture combinations) *before* (potentially) costly model fitting is performed. In addition, it ensures that best practice is followed with respect to privacy, e.g., using [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) optimisers where available. For large models and datasets, *ante-hoc* analysis has the potential for significant time and cost savings by helping to avoid wasting resources training models that are likely to be found to be disclosive after running intensive *post-hoc* analysis.
* An **attacks** package that provides *post-hoc* SDC by assessing the empirical disclosure risk of a classification model through a variety of simulated attacks *after* training. It provides an integrated suite of attacks with a common application programming interface (API) and is designed to support the inclusion of additional state-of-the-art attacks as they become available. In addition to membership inference attacks (MIA) such as the likelihood ratio attack ([LiRA](https://doi.org/10.1109/SP46214.2022.9833649)) and attribute inference, the package provides novel [structural attacks](https://arxiv.org/abs/2502.09396) that report cheap-to-compute metrics, which can serve as indicators of model disclosiveness after model fitting, but before needing to run more computationally expensive MIAs.
* Summaries of the results are written in a simple human-readable report.

Classification models from [scikit-learn](https://scikit-learn.org) (including those implementing `sklearn.base.BaseEstimator`) and [PyTorch](https://pytorch.org) are broadly supported within the package. Some attacks can still be run if only [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files of the model predicted probabilities are supplied, e.g., if the model was produced in another language. See the [examples](examples) for further information.

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

Quick-start example:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.target import Target

# Load dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fit model
model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1)
model.fit(X_train, y_train)

# Wrap model and data
target = Target(
    model=model,
    dataset_name="breast cancer",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# Create an attack object and run the attack
attack = LIRAAttack(n_shadow_models=100, output_dir="output_example")
attack.attack(target)
```

For more information, see the [examples](examples/).

## QMIA vs LiRA benchmark (development stage)

This repository now includes a reproducible benchmark runner:

- Script: `examples/sklearn/benchmark_qmia_vs_lira.py`
- One-command make targets: `make qmia-bench`, `make qmia-bench-sklearn`, `make qmia-bench-strong`

Important: this benchmark is currently intended for **development-stage
validation** while we figure out tuning and behavior of the new QMIA path. It
helps answer:

- whether QMIA is functioning correctly,
- whether QMIA is computationally cheaper than shadow-model attacks (LiRA),
- how sensitive QMIA is to parameter choices.

It is **not** a final, publication-grade comparison yet.

### Quick run (synthetic scenarios)

```bash
make qmia-bench
```

Outputs are written to:

- `outputs/benchmarks/qmia_vs_lira_make.json`
- `outputs/benchmarks/qmia_vs_lira_make.csv`

### Run on sklearn dataset presets (second way)

This uses built-in sklearn datasets as a development-stage “real data” option.
Current presets:

- `breast_cancer`
- `wine_binary` (wine class 0 vs rest; staged as binary for QMIA v1)

```bash
make qmia-bench-sklearn
```

Outputs:

- `outputs/benchmarks/qmia_vs_lira_sklearn_make.json`
- `outputs/benchmarks/qmia_vs_lira_sklearn_make.csv`

### Fast smoke run

```bash
make qmia-bench-smoke
```

### Stronger tuning preset

Use this to test a stronger (and slower) QMIA/LiRA sweep during development:

```bash
make qmia-bench-strong
```

### Override tuning knobs from the command line

```bash
make qmia-bench LIRA_SHADOW_MODELS=20,40,100 QMIA_ALPHA=0.02 QMIA_ITERATIONS=100 QMIA_DEPTH=6 QMIA_LEARNING_RATE=0.03 QMIA_L2_LEAF_REG=5.0 QMIA_SUBSAMPLE=0.9
```

To force sklearn datasets with custom presets:

```bash
make qmia-bench DATASET_SOURCE=sklearn SKLEARN_DATASETS=breast_cancer,wine_binary
```

### Post-process benchmark results (winner summary)

Print per-scenario winners for:

- fastest runtime
- best AUC
- best AUC-per-second

```bash
make qmia-bench-summary
make qmia-bench-summary-sklearn
make qmia-bench-summary-strong
```

## Documentation

See [API documentation](https://ai-sdc.github.io/SACRO-ML/).

## Contributing

See our [contributing](CONTRIBUTING.md) guide.

## Acknowledgement

This work was supported by UK Research and Innovation as part of the Data and Analytics Research Environments UK ([DARE UK](https://dareuk.org.uk)) programme, delivered in partnership with Health Data Research UK (HDR UK) and Administrative Data Research UK (ADR UK). The specific projects were Semi-Automated Checking of Research Outputs ([SACRO](https://gtr.ukri.org/projects?ref=MC_PC_23006); MC_PC_23006), Guidelines and Resources for AI Model Access from TrusTEd Research environments ([GRAIMATTER](https://gtr.ukri.org/projects?ref=MC_PC_21033); MC_PC_21033), and [TREvolution](https://dareuk.org.uk/trevolution) (MC_PC_24038). This project has also been supported by MRC and EPSRC ([PICTURES](https://gtr.ukri.org/projects?ref=MR%2FS010351%2F1); MR/S010351/1).

<img src="docs/source/images/UK_Research_and_Innovation_logo.svg" width="20%" height="20%" padding=20/> <img src="docs/source/images/health-data-research-uk-hdr-uk-logo-vector.png" width="10%" height="10%" padding=20/> <img src="docs/source/images/logo_print.png" width="15%" height="15%" padding=20/>
