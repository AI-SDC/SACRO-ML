# Example Pytorch Model Training and Assessment

## Contents

```md
pytorch
├── simple/ [Contains a very simple Pytorch example]
│   ├── dataset.py [Contains dataset module for loading CIFAR10]
│   ├── model.py [Contains example model class]
│   ├── train.py [Contains example training function]
│   ├── train_pytorch.py [Trains example model and creates the required target_dir/]
│   └── attack_pytorch.py [Loads the saved model from the target_dir/ and runs attacks]
├── cifar/ [Contains CIFAR10 example with the same structure/files as example above]
│   └── requirements.txt [Contains additional dependencies for CIFAR dataset]
└── README.md
```

## Usage

```
$ python -m train_pytorch
$ python -m attack_pytorch
```

> [!NOTE]
> For the CIFAR example, you may need to install extra dependencies:
>
> ```bash
> $ pip install -r requirements.txt
> ```

## Creating Your Own

To run the attacks you must create a `target_dir/` which contains a `target.yaml` that includes all information necessary for loading the model. The easiest way to do this is to use the `Target` class to wrap and save everything as in the `train_pytorch.py` example. If you have already trained and saved your model, you could produce a small script for loading your model and using the `Target` class to wrap and save it. Alternatively, you may use the CLI interactive prompt `$ sacroml gen-target` instead. As a last option, you may manually create the folder and `target.yaml` (you can start by copying the one produced from the example here).

Once a suitable target directory has been created `sacroml` attacks can be run using the CLI as well as programmatically as in the `attack_pytorch.py` example. To run the attacks on the CLI, first produce an `attack.yaml` with the interactive prompt `$ sacroml gen-attack` and then run the attacks with `$ sacroml run target_dir/ attack.yaml`

1. The `model.py` must contain a PyTorch model class with a constructor.
    * All parameters used by the constructor must be specified under `model_params`.
    * The file path to the Python model module must be specified under `model_module_path`.
2. The `train.py` must contain a `train` function.
    * The `train` function must have parameters:
        - `model : torch.nn.Module` PyTorch model to train.
        - `dataloader : torch.utils.data.DataLoader` training set DataLoader.
    * The `train` function may contain optional extra parameters.
        - These parameters must be specified under `train_params`.
    * The file path to the Python train module must be specified under `train_module_path`.
3. The `dataset.py` must contain a class for creating/loading the data.
    * The dataset class must implement the abstract `sacroml.attacks.data.DataHandler`:
    * The file path to the Python dataset module must be specified under `dataset_module_path`.
    * The parameter `dataset_name` must match the name of the dataset class.
    * If supplying a `dataset_module_path` the indices must be passed to `indices_train` and `indices_test`.
    * If you do not supply a `dataset_module_path`, you can pass numpy arrays to `X_train`, etc.
4. The file path to the saved PyTorch model must be specified under `model_path` in the `target.yaml` or the model should be passed to the `Target` object directly.
