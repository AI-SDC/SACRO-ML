"""An example PyTorch training module.

This module must contain a `train` function with parameters:
* model : torch.nn.Module
* dataloader : torch.utils.data.DataLoader
* Optional extra parameters may be included, as shown here.
    - These must be passed to the wrapper Target `train_params` as a dict.
"""

import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
) -> None:
    """Train model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    model.train()
    for _ in range(epochs):
        for inputs, targets in dataloader:
            X = inputs.to(device, non_blocking=True)
            y = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = mse_loss(y_pred, y.reshape(y_pred.shape))
            loss.backward()
            optimizer.step()


def score(model: nn.Module, dataloader: DataLoader) -> float:
    """Return the MSE for a given dataloader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            X = inputs.to(device, non_blocking=True)
            y = targets.to(device, non_blocking=True).float()

            y_pred = model(X)
            loss = mse_loss(y_pred, y.reshape(y_pred.shape))
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples
