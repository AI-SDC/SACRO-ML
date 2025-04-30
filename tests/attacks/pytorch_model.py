"""Pytorch model for testing."""

from __future__ import annotations

import torch
from torch import nn


class SimpleNet(nn.Module):
    """A simple Pytorch classification model."""

    def __init__(self, x_dim: int, y_dim: int) -> None:
        """Construct a simple Pytorch model."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(x_dim, 50),
            nn.ReLU(),
            nn.Linear(50, y_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate input."""
        return self.layers(x)
