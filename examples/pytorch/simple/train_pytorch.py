"""Train a Pytorch classifier on a synthetic dataset."""

import logging

import torch
from dataset import Synthetic
from model import OverfitNet
from train import test, train

from sacroml.attacks.target import Target

target_dir = "target_pytorch"
random_state = 2

if __name__ == "__main__":
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        logging.info("Found no NVIDIA driver on your system")

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
    dataset = Synthetic()
    train_loader = dataset.get_train_loader(batch_size=32)
    test_loader = dataset.get_test_loader(batch_size=32)

    logging.info("Defining the model")
    model = OverfitNet(**model_params)

    logging.info("Training the model")
    train(model, train_loader, **train_params)

    logging.info("Testing the model")
    test(model, test_loader)

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
        dataset_name="Synthetic",  # Must match the class name in the dataset module
    )

    logging.info("Writing Target object to directory: '%s'", target_dir)
    target.save(target_dir)
