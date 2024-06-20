# Examples

This folder contains examples of how to run the code contained in this repository.

## Scripts

### Contents

* Examples training a target model:
    - `train_rf_breast_cancer.py` - Trains RF on breast cancer dataset.
    - `train_rf_nursery.py` - Trains RF on nursery dataset with one-hot encoding.
* Examples programmatically running attacks:
    - `attack_lira.py` - Simulated LiRA membership inference attack on breast cancer RF.
    - `attack_worstcase.py` - Simulated worst case membership inference attack on breast cancer RF.
    - `attack_attribute.py` - Simulated attribute inference attack on nursery RF.
* Examples of attack integration within safemodel classes:
    - `safemodel.py` - Simulated attacks on a safe RF trained on the nursery dataset.

### Programmatic execution

To run a programmatic example:
1. Run the relevant training script.
2. Run the desired attack script.

For example:
```
$ python -m examples.train_rf_breast_cancer
$ python -m examples.attack_lira
```

### CLI execution

1. Run the relevant training script.
2. Generate an `attack.yaml` config.
3. Run the attack CLI tool.

For example:
```
$ python -m examples.train_rf_nursery
$ aisdc gen-attack
$ aisdc run target_rf_nursery attack.yaml
```

If you are unable to use the Python `Target` class to generate the `target.yaml` you can generate one using the CLI tool:

```
$ aisdc gen-target
```

## Notebooks

The `notebooks` folder contains short tutorials on the basic concept of "safe_XX" versions of machine learning algorithms, and examples of some specific algorithms.

## Risk Examples

The `risk_examples` contains hypothetical examples of data leakage through ML models as described by [Jefferson et al. (2022)](https://doi.org/10.5281/zenodo.6896214).
