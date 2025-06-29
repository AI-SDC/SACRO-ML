"""An example Pytorch classifier."""

import torch
from torch import nn


class Net(nn.Module):
    """A Pytorch classification model for cifar10."""

    def __init__(self, n_kernal: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, n_kernal)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, n_kernal)
        self.fc1 = nn.Linear(16 * n_kernal * n_kernal, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward propagate input."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
