"""Module containing basic conv blocks for building the models."""

from __future__ import annotations

from torch import nn


def single_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    normalization: str | None,
    dropout: float = 0.3,
    stride: int = 1,
    padding: str | int = 1,
    activation: bool | None = True,
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


def single_discriminator_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    normalization: str | None,
    dropout: float = 0.3,
    stride: int = 1,
    padding: str | int = 1,
    activation: bool = True,
) -> nn.Module:
    """Single convolutional building block for the discriminator.

    Applies spectral normalization to avoid mregularize the Discriminator.
    See https://arxiv.org/abs/1802.05957 .

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
        nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
            )
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


def single_discriminator_conv3d_block(
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

    Applies spectral normalization to avoid mregularize the Discriminator.
    See https://arxiv.org/abs/1802.05957 .

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
        nn.utils.spectral_norm(
            nn.Conv3d(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
            )
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
