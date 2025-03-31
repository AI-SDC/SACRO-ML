"""Train a Pytorch classifier on a synthetic dataset."""

import logging

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from sacroml.attacks import (
    attribute_attack,
    likelihood_attack,
    structural_attack,
    worst_case_attack,
)
from sacroml.attacks.target import Target

output_dir = "output_pytorch"
target_dir = "target_pytorch"

random_state = 2
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)


class OverfitNet(nn.Module):
    """A Pytorch classification model designed to overfit.

    To work with sacroml the class must have attributes:

    - self.layers : The model architecture.
    - self.epochs : How many training epochs are performed.
    - self.criterion : The Pytorch loss function.
    - self.optimizer : The Pytorch optimiser.

    It must also implement a fit function that takes in two numpy
    arrays and performs training.
    """

    def __init__(self, x_dim: int, y_dim: int) -> None:
        """Construct a simple Pytorch model."""
        super().__init__()
        n_units = 1000
        self.layers = nn.Sequential(
            nn.Linear(x_dim, n_units),
            nn.ReLU(),
            nn.Linear(n_units, y_dim),
        )
        self.epochs = 1000
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate input."""
        return self.layers(x)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit model to data."""
        x_tensor = torch.FloatTensor(features)
        y_tensor = torch.LongTensor(labels)

        for _ in range(self.epochs):
            # Forward
            logits = self(x_tensor)
            loss = self.criterion(logits, y_tensor)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    logging.info("Loading dataset")
    n_features = 4
    n_classes = 4
    X_orig, y_orig = make_classification(
        n_samples=50,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    X_orig = np.asarray(X_orig)
    y_orig = np.asarray(y_orig)

    logging.info("Preprocessing dataset")
    input_encoder = StandardScaler()
    X = input_encoder.fit_transform(X_orig)
    y = y_orig  # leave as labels

    logging.info("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=random_state
    )

    X = np.asarray(X)
    y = np.asarray(y)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    logging.info("Defining the model")
    model = OverfitNet(x_dim=n_features, y_dim=n_classes)

    logging.info("Training the model")
    model.fit(X_train, y_train)

    logging.info("Wrapping the model and data in a Target object")
    target = Target(
        model=model,
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

    ####################################################################
    # Model assessment
    ####################################################################

    logging.info("Loading Target object from '%s'", target_dir)
    target = Target()
    target.load(target_dir)

    logging.info("Running attacks...")

    attack = worst_case_attack.WorstCaseAttack(n_reps=10, output_dir=output_dir)
    output = attack.attack(target)

    attack = structural_attack.StructuralAttack(output_dir=output_dir)
    output = attack.attack(target)

    attack = attribute_attack.AttributeAttack(n_cpu=2, output_dir=output_dir)
    output = attack.attack(target)

    attack = likelihood_attack.LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    output = attack.attack(target)
