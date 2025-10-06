"""Module containing useful building blocks utils."""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import torch
from torch import nn

Channels: TypeAlias = list[int]


class ModelFlavour(Enum):
    """Class containing the spatial encoder's/decoder's input/output channels.

    Returns a tuple of lists of ints With the first and second element being
    respectively the encoder and decoder channels.
    """

    Small: tuple[Channels, Channels] = ([8, 16, 32, 64], [64, 32, 16, 8])
    Medium: tuple[Channels, Channels] = ([16, 32, 64, 128], [128, 64, 32, 16])
    Large: tuple[Channels, Channels] = ([32, 64, 128, 256], [256, 128, 64, 32])


def single_conv2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    normalization: str | None,
    dropout: float = 0.3,
    stride: int = 1,
    padding: str | int = 1,
    activation: bool = True,
) -> nn.Module:
    """Single convolutional building block.

    A single 2D convolutional building block with normalization,
    dropout and leakyrelu activation.

    Args:
    ----
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the convolution.
        normalization: The type of normalization to apply.
            If None, no normalization is applied. Supported normalization are
            "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
        dropout: Dropout rate.
        stride: Stride of the convolutions.
        padding: Padding to add.
        activation: Whether to use LeakyReLu activation with 0.2 negative slope.

    Returns:
    -------
        The convolution block.

    """
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
        ),
    ]
    if normalization is None:
        pass
    elif normalization == "batchnorm":
        layers.append(nn.BatchNorm2d(out_channels))
    elif normalization == "instancenorm":
        layers.append(nn.InstanceNorm2d(out_channels))
    else:
        raise NotImplementedError(f"Unknown normalization: {normalization}")

    if activation:
        layers.append(nn.LeakyReLU(0.2))

    layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def single_conv3d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    normalization: str | None,
    dropout: float = 0.3,
    stride: int = 1,
    padding: str | int = 1,
    activation: bool = True,
) -> nn.Module:
    """Single convolutional building block.

    A single 3D convolutional building block with normalization,
    dropout and leakyrelu activation.

    Args:
    ----
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the convolution.
        normalization: The type of normalization to apply.
            If None, no normalization is applied. Supported normalization are
            "instancenorm" for InstanceNorm3D, "batchnorm" for BatchNorm3D.
        dropout: Dropout rate.
        stride: Stride of the convolutions.
        padding: Padding to add.
        activation: Whether to use LeakyReLu activation with 0.2 negative slope.

    Returns:
    -------
        The convolution block.

    """
    layers: list[nn.Module] = [
        nn.Conv3d(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
        ),
    ]
    if normalization is None:
        pass
    elif normalization == "batchnorm":
        layers.append(nn.BatchNorm3d(out_channels))
    elif normalization == "instancenorm":
        layers.append(nn.InstanceNorm3d(out_channels))
    else:
        raise NotImplementedError(f"Unknown normalization: {normalization}")

    if activation:
        layers.append(nn.LeakyReLU(0.2))

    layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


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
