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

    dataset = Cifar10()
    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()

    logging.info("Defining the model")

    model_params = {  # These must match all required in the model constructor.
        "n_kernal": 5,
    }

    train_params = {  # These must match all required in the train function.
        "epochs": 2,
        "learning_rate": 0.001,
        "momentum": 0.9,
    }

    model = Net(**model_params)

    logging.info("Training the model")
    train(model, train_loader, **train_params)

    logging.info("Testing the model")
    test(model, test_loader, dataset.classes)

    #############################################################################
    # Below shows the use of the Target class to help generate the target_dir/
    # If you have already saved your model, you can use the CLI target generator.
    #############################################################################

    X_train, y_train = dataset.dataloader_to_numpy(train_loader)
    X_test, y_test = dataset.dataloader_to_numpy(test_loader)

    logging.info("Wrapping the model and data in a Target object")
    target = Target(
        model=model,
        model_module_path="model.py",
        model_params=model_params,
        train_module_path="train.py",
        train_params=train_params,
        dataset_module_path="dataset.py",
        dataset_name="Cifar10",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    logging.info("Writing Target object to directory: '%s'", target_dir)
    target.save(target_dir)

    acc_train = target.model.score(X_train, y_train)
    acc_test = target.model.score(X_test, y_test)
    logging.info("Base model train accuracy: %.4f", acc_train)
    logging.info("Base model test accuracy: %.4f", acc_test)
