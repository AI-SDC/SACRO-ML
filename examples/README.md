# Examples

This folder contains examples of how to run the code contained in this repository.

## Contents

* Examples training a target model to be assessed:
    - `train_rf_breast_cancer.py` - Trains RF on breast cancer dataset.
    - `train_rf_nursery.py` - Trains RF on nursery dataset with one-hot encoding.
* Examples programmatically running attacks:
    - `attack_lira.py` - Simulated LiRA membership inference attack on breast cancer RF.
    - `attack_worstcase.py` - Simulated worst case membership inference attack on breast cancer RF.
    - `attack_attribute.py` - Simulated attribute inference attack on nursery RF.
* Examples of attack integration within safemodel classes:
    - `safemodel.py` - Simulated attacks on a safe RF trained on the nursery dataset.
* Examples training and attacking a [Pytorch model](pytorch/);

## Overview

The privacy attacks can be run in one of two different ways:
1. **Programmatic execution**, which involves running Python code that:
    1. Imports the desired attack(s);
    2. Instantiates the desired attack object(s);
    3. Calls an `attack()` function, passing in a `sacroml.target.Target` object containing a wrapped model and dataset.
2. **Command line interface (CLI) execution**, which involves:
    1. A saved `target_dir/` folder containing the model, data, and metadata needed to load/train the model.
    2. An `attack.yaml` configuration file specifying which attacks to run.
    3. Running the attacks via the command: `$ sacroml run target_dir/ attack.yaml`

The example training scripts show how to use the `Target` object to wrap and save a model to a `target_dir/` which can then be either reloaded to run attacks programmatically or passed as a CLI argument.

## Programmatic Execution

See the example Python scripts.

To run a programmatic example:
1. Run the relevant training script.
2. Run the desired attack script.

For example:
```
$ python -m examples.train_rf_breast_cancer
$ python -m examples.attack_lira
```

## CLI Execution

1. Run the relevant training script.
2. Generate an `attack.yaml` config.
3. Run the attack CLI tool.

For example:
```
$ python -m examples.train_rf_nursery
$ sacroml gen-attack
$ sacroml run target_rf_nursery attack.yaml
```

The `sacroml` package provides three basic commands:

| Command | Description |
| ------- | ----------- |
| `sacroml gen-attack` | Generate an `attack.yaml` configuration to specify which attacks to run. |
| `sacroml gen-target` | Generate a `target_dir/` containing a `target.yaml` with a saved model. |
| `sacroml run target_dir/ attack.yaml` | Runs the specified attacks on the target model. |

## User Stories

A collection of user guides can be found in the [`user_stories`](user_stories) folder of this repository. These guides include configurable examples from the perspective of both a researcher and a TRE, with separate scripts for each. Instructions on how to use each of these scripts and which scripts to use are included in the README located in the folder.

## Notebooks

The `notebooks` folder contains short tutorials on the basic concept of "safe" versions of machine learning algorithms, and examples of some specific algorithms.

## Risk Examples

The `risk_examples` contains hypothetical examples of data leakage through ML models as described by [Jefferson et al. (2022)](https://doi.org/10.5281/zenodo.6896214).
