"""Module containing the discriminator modules."""

from __future__ import annotations

import torch
from torch import nn

from swigan.models.utils import single_conv2d_block


class BaseDiscriminator(nn.Module):
    """The network for the final discriminators."""

    def __init__(self, input_channels: int, temporal_dim: int) -> None:
        """Initialize the input arguments.

        Args:
        ----
            input_channels: The number of input channels.
            temporal_dim: The dimension of the input temporal vector.
                The temporal vector will be appended to the channels
                on the input frames.

        """
        super().__init__()
        self.conv1 = single_conv2d_block(
            in_channels=input_channels + temporal_dim,
            out_channels=16,
            kernel_size=3,
            dropout=0.0,
            normalization="instancenorm",
            padding=1,
            stride=2,
        )
        self.conv2 = single_conv2d_block(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            dropout=0.0,
            normalization="instancenorm",
            padding=1,
            stride=2,
        )
        self.conv3 = single_conv2d_block(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            dropout=0.0,
            normalization="instancenorm",
            padding=1,
            stride=2,
        )

    def forward(self, inputs: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
          inputs: Input tensor of shape (batch_size, seq_len, channels, height, width)
          timestamps: Timestamp tensor of shape (batch_size, seq_len, temporal_dim)

        Returns:
        -------
          torch.Tensor: Output tensor of shape(batch_size, seq_len, hidden_dim).

        """
        out = torch.concat(
            [inputs, timestamps[..., None, None].repeat(1, 1, 1, *inputs.shape[-2:])],
            dim=-3,
        )
        batch_size, length, channels, height, width = out.shape

        out = out.reshape(batch_size * length, channels, height, width)

        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)

        if out.shape[-2:] != (5, 6):
            raise RuntimeError(f"Output shape should be (B, 64, 4, 5), but got {out.shape}")
        return out.reshape(batch_size, length, *out.shape[-3:])


class PatchGANDiscriminator(nn.Module):
    """The patchGAN discriminator. Patches of (4, 5) are considered."""

    def __init__(self, input_channels: int) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_channels: Number of input channels.

        """
        super().__init__()
        self.conv = single_conv2d_block(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=1,
            dropout=0.0,
            normalization="instancenorm",
            padding=0,
            stride=1,
        )
        self.final_conv = single_conv2d_block(
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


class FrameDiscriminator(nn.Module):
    """The frame level discriminator."""

    def __init__(self, input_channels: int) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_channels: Number of input channels.

        """
        super().__init__()
        self.conv1 = single_conv2d_block(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=2,
            dropout=0.0,
            normalization="instancenorm",
            padding=0,
            stride=2,
        )
        self.conv2 = single_conv2d_block(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=1,
            dropout=0.0,
            normalization="instancenorm",
            padding=0,
            stride=2,
        )
        self.final_conv = single_conv2d_block(
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
        out = out.reshape(batch_size * length, channels, height, width)
        return out.squeeze(dim=(-2, -1))
