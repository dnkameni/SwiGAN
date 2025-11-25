"""Module containing Spatial and Channel wise Squeeze Excitation Network."""

import torch
from torch import nn


class SCSEModule(nn.Module):
    """Implement a Spatial and Channel Squeeze Excitation (scSE) Module.

    Incorporates Attention mechanism in the decoding phase.
    See <https://arxiv.org/abs/1803.02579> for more details.
    """

    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        """Initialize the input class.

        Args:
        ----
            in_channels: Number of input channels.
            reduction: Reduction factor to apply. Reduces the number of input_channels
                to input_channels // reduction.

        """
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            x: The input tensor. Should be of shape (batch_size, channels, height, width).

        Returns:
        -------
            A tensor of shape (batch_size, channels, height, width)

        """
        return x * self.cSE(x) + x * self.sSE(x)
