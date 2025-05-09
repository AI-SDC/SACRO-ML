"""An example Pytorch training."""

import numpy as np
import torch
from torch import nn, optim


def train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    momentum: float,
) -> None:
    """Train Pytorch model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for _ in range(epochs):
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
