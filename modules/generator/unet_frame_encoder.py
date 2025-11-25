"""Encoder for the UNET SWIGAN variant."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.ops import stochastic_depth

from modules.base_conv_blocks import single_conv_block
from modules.generator.center_block import CenterBlock
from modules.scse import SCSEModule
from modules.utils import glorot_init


class UNetEncoderBlock(nn.Module):
    """A single upscale decoding block.

    Uses skip connections, transpose convolutions and scSE attention modules.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        normalization: str | None,
        prob: float = 1.0,
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
            prob: Survival probability for stochastic depth.

        """
        super().__init__()
        self.proj_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1 = single_conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dropout=dropout,
            normalization=normalization,
            padding=1,
        )
        self.conv2 = single_conv_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dropout=dropout,
            normalization=normalization,
            padding=1,
        )
        self.attention = SCSEModule(in_channels=out_channels)
        self.noise_weights1 = nn.Parameter(torch.randn(out_channels))
        self.noise_weights2 = nn.Parameter(torch.randn(out_channels))
        self.prob = prob

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
        noise = torch.randn(
            (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]), device=inputs.device
        )
        out = inputs
        x = self.proj_layer(out)
        out = self.conv1(out)
        out = out + noise * self.noise_weights1[None, :, None, None]

        out = self.conv2(out)
        out = out + noise * self.noise_weights2[None, :, None, None]

        out = self.attention(out)
        out = stochastic_depth(out, 1 - self.prob, mode="batch", training=self.training)
        out = out + x
        return out


class UNetFrameEncoder(nn.Module):
    """The frame encoder applied to each map individually."""

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        dropout: float,
        normalization: str | None = "batchnorm",
        apply_center_block: bool = False,
    ) -> None:
        """Initialize the module.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: Number of output_channels.
            dropout: Dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2d.
            apply_center_block: Whether to apply the center block at the end of the encoding.

        """
        super().__init__()
        in_channel = in_channels
        layers = []
        downsample_layers = []
        self.probs = list(np.arange(1.0, 0.5, -0.5 / (len(out_channels) - 1))) + [0.5]
        for idx, out in enumerate(out_channels):
            layers.append(
                UNetEncoderBlock(
                    in_channels=in_channel,
                    out_channels=out,
                    dropout=dropout,
                    normalization=normalization,
                    prob=self.probs[idx],
                )
            )

            downsample_layers.append(
                single_conv_block(
                    in_channels=out,
                    out_channels=out,
                    kernel_size=2,
                    stride=2,
                    dropout=dropout,
                    normalization=normalization,
                    padding=0,
                )
            )
            in_channel = out
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        if apply_center_block:
            self.center = CenterBlock(
                out_channels[-1],
                out_channels[-1],
                dropout,
                normalization,
            )
        else:
            self.center = nn.Identity()
        self.apply(glorot_init)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
        ----
            inputs: Input tensor. Should be of size
                (batch_size, channels, height, width)

        Returns:
        -------
            A Tensor of shape (batch_size, out_channels)

        """
        out = inputs
        outputs = []
        for _, (block, downsample) in enumerate(
            zip(self.layers, self.downsample_layers, strict=True)
        ):
            out = block(out)
            outputs.append(out)
            out = downsample(out)
        out = self.center(out)
        outputs.append(out)
        return outputs
