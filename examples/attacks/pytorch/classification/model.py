"""An example Pytorch classifier."""

import torch


class OverfitNet(torch.nn.Module):
    """An example Pytorch classification model."""

    def __init__(self, x_dim: int, y_dim: int, n_units: int) -> None:
        """Construct a simple Pytorch model."""
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(x_dim, n_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_units, y_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate input."""
        return self.layers(x)
