"""Train a Pytorch classifier on a synthetic dataset."""

import logging

import torch
from dataset import get_data
from model import OverfitNet
from train import train

from sacroml.attacks.target import Target

target_dir = "target_pytorch"

random_state = 2
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)

if __name__ == "__main__":
    #############################################################################
    # Dataset loading and model training
    #############################################################################

    model_params = {  # These must match all required in the model constructor.
        "x_dim": 4,
        "y_dim": 4,
        "n_units": 1000,
    }
    train_params = {  # These must match all required in the train function.
        "epochs": 1000,
        "learning_rate": 0.001,
        "momentum": 0.9,
    }

    logging.info("Loading dataset")
    X, y, X_train, y_train, X_test, y_test = get_data(
        n_features=model_params["x_dim"],
        n_classes=model_params["y_dim"],
        random_state=random_state,
    )

    logging.info("Defining the model")
    model = OverfitNet(**model_params)

    logging.info("Training the model")
    train(model, X_train, y_train, **train_params)

    #############################################################################
    # Below shows the use of the Target class to help generate the target_dir/
    # If you have already saved your model, you can use the CLI target generator.
    #############################################################################

    logging.info("Wrapping the model and data in a Target object")
    target = Target(
        model=model,
        model_module_path="model.py",
        model_params=model_params,
        train_module_path="train.py",
        train_params=train_params,
        dataset_module_path="dataset.py",
        dataset_name="synthetic",
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
    for i in range(model_params["x_dim"]):
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
