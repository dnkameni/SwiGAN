"""Module containing the base discriminator."""

from __future__ import annotations

import torch
from torch import nn

from modules.base_conv_blocks import single_conv_block


class BaseDiscriminator(nn.Module):
    """The network for the final discriminators."""

    def __init__(self, input_channels: int) -> None:
        """Initialize the input arguments.

        Args:
        ----
            input_channels: The number of input channels.
            temporal_dim: The dimension of the input temporal vector.
                The temporal vector will be appended to the channels
                on the input frames.

        """
        super().__init__()
        self.conv1 = single_conv_block(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            dropout=0.0,
            normalization="instancenorm",
            padding=1,
            stride=2,
        )
        self.conv2 = single_conv_block(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            dropout=0.0,
            normalization="instancenorm",
            padding=1,
            stride=2,
        )
        self.conv3 = single_conv_block(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            dropout=0.0,
            normalization="instancenorm",
            padding=1,
            stride=2,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
          inputs: Input tensor of shape (batch_size, seq_len, channels, height, width)

        Returns:
        -------
          torch.Tensor: Output tensor of shape(batch_size, seq_len, hidden_dim).

        """
        batch_size, length, channels, height, width = inputs.shape

        out = inputs.reshape(batch_size * length, channels, height, width)

        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)

        if out.shape[-2:] != (
            5,
            6,
        ):  # Unfortunately this is hard coded due to the uneven input shapes of (37, 44)
            raise RuntimeError(f"Output shape should be (B, 64, 5, 6), but got {out.shape}")
        return out.reshape(batch_size, length, *out.shape[-3:])
