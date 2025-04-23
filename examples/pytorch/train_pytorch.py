"""Train a Pytorch classifier on a synthetic dataset."""

import logging

import torch
from dataset import get_data
from model import OverfitNet

from sacroml.attacks.target import Target

target_dir = "target_pytorch"

random_state = 2
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)

if __name__ == "__main__":
    logging.info("Loading dataset")
    n_features = 4
    n_classes = 4
    X, y, X_train, y_train, X_test, y_test = get_data(
        n_features=n_features, n_classes=n_classes, random_state=random_state
    )

    logging.info("Defining the model")
    model = OverfitNet(x_dim=n_features, y_dim=n_classes)

    logging.info("Training the model")
    model.fit(X_train, y_train)

    logging.info("Wrapping the model and data in a Target object")
    target = Target(
        model=model,
        dataset_name="synthetic",
        model_module_path="model.py",
        dataset_module_path="dataset.py",
        # processed data
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        # original unprocessed data (for attribute attack)
        # in this example we just use the processed data since it's all floats
        X_orig=X,
        y_orig=y,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )

    logging.info("Wrapping feature details and encoding for attribute inference")
    for i in range(n_features):
        target.add_feature(
            name=f"A{i}",
            indices=[i],
            encoding="float",
        )

    logging.info("Writing Target object to directory: '%s'", target_dir)
    target.save(target_dir)

    acc_train = target.model.score(X_train, y_train)
    acc_test = target.model.score(X_test, y_test)
    logging.info("Base model train accuracy: %.4f", acc_train)
    logging.info("Base model test accuracy: %.4f", acc_test)
