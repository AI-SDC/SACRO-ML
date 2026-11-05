"""An example Pytorch training module."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
    task: str = "classifier",
) -> None:
    """Train model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss() if task == "regressor" else nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for inputs, labels in dataloader:
            X = inputs.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_pred = model(X)

            if task == "regressor":
                y = y.float().reshape(y_pred.shape)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
