## Example Pytorch Model Training and Assessment

### Contents

* `dataset.py` - Creates synthetic test data.
* `model.py` - Contains example Pytorch class.
* `train_pytorch.py` - Trains example Pytorch model and saves it with Target class.
* `attack_pytorch.py` - Loads the saved Pytorch model and runs attacks.

### Usage

```
$ python -m train_pytorch
$ python -m attack_pytorch
```

### Creating Your Own

1. The `model.py` must contain a Pytorch class with a constructor.
    * All parameters used by the constructor must be specified under
      `model_params` in the `target.yaml`.
2. The `train.py` must contain a `train` function.
    * The `train` function must have parameters:
        - `model` The Pytorch model to train.
        - `X` A numpy array containing features (NOTE: will change in a later version)
        - `y` A numpy array containing labels (NOTE: will change in a later version)
    * The `train` function may contain optional extra parameters.
        - These parameters must be specified under `train_params` in the `target.yaml`.

To create the target directory needed for running attacks, you may use the
`Target` object to wrap and save everything as in the `train_pytorch.py`
example. Alternatively, you may use the CLI interactive prompt. As a last
option you may manually create the folder and `target.yaml`. Once a suitable
target directory has been created `sacroml` attacks can be run using the CLI as
well as programmatically as in the `attack_pytorch.py` example.
