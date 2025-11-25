"""Module containing the frame decoder."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms.functional import center_crop

from modules.base_conv_blocks import single_conv_block
from modules.scse import SCSEModule


class DecoderBlock(nn.Module):
    """A single upscale decoding block.

    Uses skip connections, transpose convolutions and scSE attention modules.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        normalization: str | None,
    ) -> None:
        """Initialize the input parameters.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: The number of output channels.
            dropout: Dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.

        """
        super().__init__()
        self.inner_upscale = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            output_padding=1,
        )
        self.upscale = nn.Sequential(
            nn.Dropout(dropout),
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
                output_padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv1 = single_conv_block(
            in_channels,
            out_channels,
            dropout=dropout,
            normalization=normalization,
            kernel_size=3,
            padding=1,
        )
        self.attention1 = SCSEModule(in_channels=in_channels)
        self.conv2 = single_conv_block(
            out_channels,
            out_channels,
            dropout=dropout,
            normalization=normalization,
            kernel_size=3,
            padding=1,
        )
        self.attention2 = SCSEModule(in_channels=out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            inputs: The input Tensor. Should be of shape
                (batch_size, channels, height, width).

        Returns:
        -------
            A Tensor of shape (batch_size, out_channels, 2 * height, 2 * width).

        """
        x = self.inner_upscale(inputs)
        out = self.upscale(inputs)
        out = self.attention1(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.attention2(out)
        return out + x


class FrameDecoder(nn.Module):
    """The frame decoder applied to each map individually."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        decoder_channels: list[int],
        input_map_dims: list[int],
        dropout: float = 0.3,
        normalization: str | None = "batchnorm",
    ) -> None:
        """Initialize input parameters.

        Args:
        ----
            input_dim: Number of input channels.
            output_dim: Number of output channels.
            decoder_channels: A list of output channels of the decoding stage.
            input_map_dims: A tuple (H, W) specifying the height and width
                of the input maps. This is used to crop the final output of
                the decoder to the same size as the input of the encoder.
            dropout: The dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.

        """
        super().__init__()
        # computing blocks input and output channels
        in_channels = [input_dim] + decoder_channels

        # combine decoder keyword arguments
        blocks: list[nn.Module] = [
            DecoderBlock(
                in_channels=in_channels[i],
                out_channels=in_channels[i + 1],
                dropout=dropout,
                normalization=normalization,
            )
            for i in range(len(in_channels) - 1)
        ]
        blocks.append(
            single_conv_block(
                in_channels=in_channels[-1],
                out_channels=output_dim,
                kernel_size=1,
                dropout=dropout,
                normalization=None,
                padding=0,
            )
        )
        self.decoder = nn.Sequential(*blocks)
        self.input_map_dims = input_map_dims

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            inputs: The input Tensor.

        Returns:
        -------
            A Tensor of shape (batch_size, output_dim, H, W).

        """
        # to match the final size of the frame encoder output
        # Unfortunately this is hard coded sinc the input maps have a fixed
        # uneven size (37, 44).
        inputs = inputs.repeat(1, 1, 2, 2)
        out = self.decoder(inputs)
        # Crop the final output to match the input maps width and height
        out = center_crop(out, self.input_map_dims)
        return out
