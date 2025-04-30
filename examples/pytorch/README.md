# Example Pytorch Model Training and Assessment

## Contents

```md
pytorch
├── dataset.py [Contains example dataset function]
├── model.py [Contains example model class]
├── train.py [Contains example training function]
├── train_pytorch.py [Trains example model and creates the required target_dir/]
├── attack_pytorch.py [Loads the saved model and runs attacks]
```

## Usage

```
$ python -m train_pytorch
$ python -m attack_pytorch
```

## Creating Your Own

1. The `model.py` must contain a Pytorch model class with a constructor.
    * All parameters used by the constructor must be specified under
      `model_params` in the `target.yaml`.
    * The file path to the Python model module must be specified under
      `model_module_path`.
2. The `train.py` must contain a `train` function.
    * The `train` function must have parameters:
        - `model : torch.nn.Module` Pytorch model to train.
        - `X : numpy.ndarray` Array containing features (NOTE: will change in a
          later version.)
        - `y : numpy.ndarray` Array containing labels (NOTE: will change in a
          later version.)
    * The `train` function may contain optional extra parameters.
        - These parameters must be specified under `train_params` in the `target.yaml`.
    * The file path to the Python train module must be specified under
      `train_module_path`.
3. The `dataset.py` is not yet required, but can be included in the target
   directory by setting the `dataset_module_path` parameter.
4. The file path to the saved Pytorch model must be specified under `model`.
5. Processed data as numpy arrays must be provided as usual: `X_train, y_train,
   X_test, y_test`. (NOTE: will change in a later version.)

To create the target directory needed for running attacks, you may use the
`Target` object to wrap and save everything as in the `train_pytorch.py`
example. Alternatively, you may use the CLI interactive prompt. As a last
option you may manually create the folder and `target.yaml`. Once a suitable
target directory has been created `sacroml` attacks can be run using the CLI as
well as programmatically as in the `attack_pytorch.py` example.
