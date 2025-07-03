"""Train a Pytorch classifier on CIFAR10."""

import logging

import torch
from dataset import Cifar10
from model import Net
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

    logging.info("Loading dataset")

    # Access dataset
    data_handler = Cifar10()

    # Get the (preprocessed) dataset
    dataset = data_handler.get_dataset()

    # Create data splits
    indices_train, indices_test = data_handler.get_train_test_indices()

    # Get dataloaders
    train_loader = data_handler.get_dataloader(dataset, indices_train, shuffle=True)
    test_loader = data_handler.get_dataloader(dataset, indices_test, shuffle=False)

    logging.info("Defining the model")

    model_params = {
        "n_kernal": 5,
    }
    train_params = {
        "epochs": 100,
        "learning_rate": 0.001,
        "momentum": 0.9,
    }
    model = Net(**model_params)

    logging.info("Training the model")
    train(model, train_loader, **train_params)

    logging.info("Testing the model")
    test(model, test_loader, data_handler.classes)

    #############################################################################
    # Below shows the use of the Target class to help generate the target_dir/
    # If you have already saved your model, you can use the CLI target generator.
    #############################################################################

    logging.info("Wrapping the model and data in a Target object")
    target = Target(
        model=model,
        model_module_path="model.py",
        model_params=model_params,  # Must match all required in model constructor
        train_module_path="train.py",
        train_params=train_params,  # Must match all required in the train function
        dataset_module_path="dataset.py",
        dataset_name="Cifar10",  # Must match the class name in dataset module
        indices_train=indices_train,
        indices_test=indices_test,
    )

    logging.info("Writing Target object to directory: '%s'", target_dir)
    target.save(target_dir)
