"""Pytorch train function for testing."""

import numpy as np
import torch
from torch import nn, optim


def train(  # pylint: disable=too-many-arguments
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    momentum: float,
) -> None:
    """Train Pytorch model."""
    x_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for _ in range(epochs):
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
