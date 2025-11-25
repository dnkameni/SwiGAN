"""Module containing the patchGAN discriminator."""

from __future__ import annotations

import torch
from torch import nn

from modules.base_conv_blocks import single_discriminator_conv_block
from modules.utils import glorot_init


class PatchGANDiscriminator(nn.Module):
    """The patchGAN discriminator. Patches of (5, 6) are considered."""

    def __init__(self, input_channels: int) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_channels: Number of input channels.

        """
        super().__init__()

        self.conv = single_discriminator_conv_block(
            in_channels=input_channels,
            out_channels=input_channels * 2,
            kernel_size=2,
            dropout=0.0,
            normalization="instancenorm",
            padding=0,
            stride=1,
        )

        self.final_conv = single_discriminator_conv_block(
            in_channels=input_channels * 2,
            out_channels=1,
            kernel_size=1,
            dropout=0.0,
            normalization=None,
            padding=0,
            stride=1,
            activation=False,
        )
        self.apply(glorot_init)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
        ----
            inputs: Input tensor of shape (batch_size, seq_len, channels, height, width).

        Returns:
        -------
            A tensor of shape (batch_size, seq_len, 1).

        """
        features = []
        out = self.conv(inputs)
        features.append(out)
        out = self.final_conv(out)  # (B, 64, 4, 5)
        return out, features
