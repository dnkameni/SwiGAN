"""Module containing the patchGAN discriminator."""

from __future__ import annotations

import torch
from torch import nn

from modules.base_conv_blocks import single_conv_block


class PatchGANDiscriminator(nn.Module):
    """The patchGAN discriminator. Patches of (4, 5) are considered."""

    def __init__(self, input_channels: int) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_channels: Number of input channels.

        """
        super().__init__()
        self.conv = single_conv_block(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=1,
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
            inputs: Input tensor of shape (batch_size, seq_len, channels, height, width).

        Returns:
        -------
            A tensor of shape (batch_size, seq_len, 1).

        """
        batch_size, length, channels, height, width = inputs.shape
        out = inputs.reshape(batch_size * length, channels, height, width)
        out = self.conv(out)
        out = self.final_conv(out)  # (B, 64, 5, 6)
        out = out.reshape(batch_size, length, *out.shape[-3:])
        return out.mean(dim=(-2, -1))
