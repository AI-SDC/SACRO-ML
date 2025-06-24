# CIFAR10 Example Pytorch Model Training and Assessment

## Contents

```md
pytorch
├── dataset.py [Contains dataset module for loading CIFAR10]
├── model.py [Contains example model class]
├── train.py [Contains example training function]
├── train_pytorch.py [Trains example model and creates the required target_dir/]
├── attack_pytorch.py [Loads the saved model from the target_dir/ and runs attacks]
├── requirements.txt [Contains dependencies needed to run this example]
```

## Usage

```
$ pip install -r requirements.txt
$ python -m train_pytorch
$ python -m attack_pytorch
```
