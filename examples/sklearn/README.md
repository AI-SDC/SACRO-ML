# Example Scikit-learn Model Training and Assessment

## Contents

```md
sklearn/
├── cancer/ [Contains a scikit-learn example passing data arrays directly]
│   ├── train_rf_cancer.py [Train a scikit-learn model]
│   └── attack_rf_cancer.py [Attack a scikit-learn model]
├── nursery/ [Contains a scikit-learn example with attribute attack]
│   ├── train_rf_nursery.py [Train a scikit-learn model]
│   └── attack_rf_nursery.py [Attack a scikit-learn model]
└── README.md
```

## Usage

1. Run the relevant training script.
2. Run the desired attack script.

For example:
```
$ python -m cancer.train_rf_cancer
$ python -m cancer.attack_rf_cancer
```
