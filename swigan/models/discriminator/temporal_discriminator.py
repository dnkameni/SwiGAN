"""Module containing the temporal level discriminator."""

from __future__ import annotations

import torch
from torch import nn

from swigan.models.utils import single_conv3d_block


class TemporalDiscriminator(nn.Module):
    """The temporal level discriminator."""

    def __init__(self, input_channels: int) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_channels: Number of input channels.

        """
        super().__init__()
        self.conv1 = single_conv3d_block(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=2,
            dropout=0.0,
            normalization="instancenorm",
            padding=0,
            stride=1,
        )
        self.conv2 = single_conv3d_block(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=2,
            dropout=0.0,
            normalization="instancenorm",
            padding=1,
            stride=2,
        )
        self.fc_layer = nn.Linear(input_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            inputs: The input frames. Must be of shape
                (batch_size, seq_len, channels, height, width).

        Returns:
        -------
            A tensor of shape (batch_size, seq_len, 1)

        """
        out = self.conv1(inputs.transpose(1, 2))
        out = self.conv2(out)
        out = self.fc_layer(out.mean(dim=(-3, -2, -1)))
        return out
