"""Module containing the frame level discriminator."""

from __future__ import annotations

import torch
from torch import nn

from modules.base_conv_blocks import single_conv_block


class FrameDiscriminator(nn.Module):
    """The frame level discriminator."""

    def __init__(self, input_channels: int) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_channels: Number of input channels.

        """
        super().__init__()
        self.conv1 = single_conv_block(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=2,
            dropout=0.0,
            normalization="instancenorm",
            padding=0,
            stride=2,
        )
        self.conv2 = single_conv_block(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=2,
            dropout=0.0,
            normalization="instancenorm",
            padding=0,
            stride=1,
        )
        self.final_conv = single_conv_block(
            in_channels=input_channels,
            out_channels=1,
            kernel_size=1,
            dropout=0.0,
            normalization=None,
            padding=0,
            stride=1,
        )

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
        batch_size, length, channels, height, width = inputs.shape
        out = inputs.reshape(batch_size * length, channels, height, width)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.final_conv(out)
        out = out[..., 0, 0]
        return out
