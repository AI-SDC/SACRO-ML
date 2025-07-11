# Example Running Attacks Given Predicted Probabilities

For models where `sacroml` does not provide complete support (e.g., those created in R) a limited number of attacks can still be run if the model predicted probabilities for the train and test set are provided as csv files.

The `attack.py` example provided here shows how the csv files can be loaded and attacks run. The `train.py` trains a scikit-learn model and saves the probabilities, however it is only provided for a complete working example: the csv files can be generated with any language.

> [!WARNING]
> CSV files must not have headers.

> [!NOTE]
> The interactive prompt `sacroml gen-target` can also be used to create a loadable directory by supplying the paths to csv files.

## Contents

```md
unsupported/
├── train.py  [Trains a model and saves predicted probabilities as csv]
├── attack.py [Runs attacks using csv probabilities]
└── README.md
```

## Usage

1. Run the relevant training script to save csv probabilities.
2. Run the attack script.

```
$ python -m train
$ python -m attack
```
