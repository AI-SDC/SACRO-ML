"""An example PyTorch training module.

This module must contain a `train` function with parameters:
* model : torch.nn.Module
* dataloader : torch.utils.data.DataLoader
* Optional extra parameters may be included, as shown here.
    - These must be passed to the wrapper Target `train_params` as a dict.
"""

import torch
from torch import nn, optim
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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    model.train()
    for _ in range(epochs):
        for inputs, labels in dataloader:
            X = inputs.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()


def test(model: nn.Module, dataloader: DataLoader) -> None:
    """Test model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            X, y = inputs.to(device), labels.to(device)
            logits = model(X)
            _, predicted = torch.max(logits, 1)

            correct += (predicted == y).sum().item()
            total += y.size(0)

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
