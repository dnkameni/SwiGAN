"""Contains the modules for the image encoder."""

from __future__ import annotations

import torch
from torch import nn

from modules.utils import SCSEModule, single_conv2d_block


class CenterBlock(nn.Sequential):
    """Center convolutional block to apply at the end of the encoding stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        normalization: str | None,
    ) -> None:
        """Initialize the class.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: Number of output_channels.
            dropout: Dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.

        """
        conv1 = single_conv2d_block(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            normalization=normalization,
            activation=True,
            dropout=dropout,
        )
        conv2 = single_conv2d_block(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            normalization=normalization,
            activation=False,
            dropout=dropout,
        )
        super().__init__(conv1, conv2)


class FrameEncoder(nn.Module):
    """The frame encoder applied to each map individually."""

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        output_dim: int,
        dropout: float,
        normalization: str = "batchnorm",
        apply_center_block: bool = False,
    ) -> None:
        """Initialize the module.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: Number of output_channels.
            output_dim: Dimension of the output feature after the FC layer.
            dropout: Dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
            apply_center_block: Whether to apply the center block at the end of the encoding.

        """
        super().__init__()
        in_channel = in_channels
        layers = []
        downsample_layers = []
        proj_layers = []
        for out in out_channels:
            layers.append(
                nn.Sequential(
                    SCSEModule(in_channels=in_channel),
                    single_conv2d_block(
                        in_channels=in_channel,
                        out_channels=out,
                        kernel_size=3,
                        dropout=dropout,
                        normalization=normalization,
                        padding=1,
                    ),
                    single_conv2d_block(
                        in_channels=out,
                        out_channels=out,
                        kernel_size=3,
                        dropout=dropout,
                        normalization=normalization,
                        padding=1,
                    ),
                    SCSEModule(in_channels=out),
                )
            )
            downsample_layers.append(
                single_conv2d_block(
                    in_channels=out,
                    out_channels=out,
                    kernel_size=2,
                    stride=2,
                    dropout=dropout,
                    normalization=normalization,
                    padding=0,
                ),
            )
            proj_layers.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            in_channel = out
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.proj_layers = nn.ModuleList(proj_layers)
        self.final_layer = nn.Linear(out_channels[-1], output_dim)
        if apply_center_block:
            self.center = CenterBlock(
                out_channels[-1],
                out_channels[-1],
                dropout,
                normalization,
            )
        else:
            self.center = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            inputs: Input tensor. Should be of size
                (batch_size, channels, height, width)

        Returns:
        -------
            A Tensor of shape (batch_size, output_dim)

        """
        out = inputs
        for block, downsample, proj in zip(
            self.layers, self.downsample_layers, self.proj_layers, strict=True
        ):
            x = proj(out)
            out = block(out)
            out = out + x
            out = downsample(out)
        out = self.center(out)
        out = out.mean(dim=(-2, -1))
        out = self.final_layer(out)
        return out
