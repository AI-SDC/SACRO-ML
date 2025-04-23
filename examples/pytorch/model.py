"""An example Pytorch classifier."""

import numpy as np
import torch
from torch import nn, optim


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
