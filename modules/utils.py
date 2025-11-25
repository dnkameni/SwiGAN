"""Module containing useful building blocks utils."""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

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


def glorot_init(m: nn.Module) -> None:
    """Util function to initialize all Conv and linear blocks with Glorot."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv3d)):  # noqa
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def glorot_gru_init(gru_layer: nn.Module) -> None:
    """Util function to initialize all GRU cells with Glorot an orthogonal initialization."""
    for name, param in gru_layer.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param.data)
        elif "bias" in name:
            nn.init.zeros_(param.data)
            # Optional: bias for update gate (helps training)
            n = param.size(0)
            param.data[n // 3 : n // 3 * 2].fill_(1.0)
