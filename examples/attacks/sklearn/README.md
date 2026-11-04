# Example Scikit-learn Model Training and Assessment

## Contents

```md
sklearn/
├── classification/ [Contains scikit-learn classification examples]
│   ├── cancer/ [Contains a scikit-learn example passing data arrays directly]
│   │   ├── main_train.py [Trains a model and creates the required target_dir/]
│   │   └── main_attack.py [Loads the model from the target_dir/ and runs attacks]
│   └── nursery/ [Contains a scikit-learn example passing a dataset class]
├── regression/ [Contains scikit-learn regression examples]
│   └── diabetes/ [Contains a scikit-learn example passing data arrays directly]
└── README.md
```

## Usage

1. Run the relevant training script.
2. Run the desired attack script.

For example, from the `cancer/` folder:
```
$ python main_train
$ python main_attack
```
