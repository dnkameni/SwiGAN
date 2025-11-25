"""Module containing the base discriminator."""

from __future__ import annotations

import torch
from torch import nn

from modules.base_conv_blocks import single_discriminator_conv_block
from modules.discriminator.minibatch_discrimination import MiniBatchDiscrimination
from modules.utils import glorot_init


class BaseDiscriminator(nn.Module):
    """The network for the common parts of the discriminators."""

    def __init__(
        self,
        input_channels: int,
        output_channels: list[int],
        mbd_output_channels: int | None = None,
    ) -> None:
        """Initialize the input arguments.

        Args:
        ----
            input_channels: The number of input channels.
            output_channels: The number of output channels.
            mbd_output_channels: The number of output channels of the MiniBatch
                Discrimination module. If None, no Minibatch Discrimination is used.

        """
        super().__init__()
        channels = [input_channels] + output_channels
        self.blocks = nn.ModuleList(
            [
                single_discriminator_conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dropout=0.0,
                    normalization="instancenorm",
                    padding=1,
                    stride=2,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:], strict=False)
            ]
        )
        if mbd_output_channels is not None:
            self.mbd = MiniBatchDiscrimination(
                in_features=output_channels[-1], out_features=mbd_output_channels, kernel_dims=10
            )
        else:
            self.mbd = nn.Identity()
        self.apply(glorot_init)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
        ----
          inputs: Input tensor of shape (batch_size, seq_len, channels, height, width)
          timestamps: Timestamp tensor of shape (batch_size, seq_len, temporal_dim)

        Returns:
        -------
          torch.Tensor: Output tensor of shape(batch_size, seq_len, hidden_dim).

        """
        features = []
        out = inputs
        for block in self.blocks:
            out = block(out)
            features.append(out)
        out = self.mbd(out)

        if out.shape[-2:] != (5, 6):
            raise RuntimeError(f"Output shape should be (B, 64, 5, 6), but got {out.shape}")
        return out.contiguous(), features
