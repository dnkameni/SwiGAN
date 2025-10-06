"""Module containing an LSTM-based temporal frame decoder."""

from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """An LSTM/GRU-based encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        temporal_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        cell_type: str = "lstm",
    ) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_dim: Number of dimensions of the input features.
            hidden_dim: Number of dimensions of the LSTM/GRU hidden features.
            num_layers: The number of LSTM/GRU layers.
            dropout: The dropout rate.
            temporal_dim: The dimensions of the temporal vector. This
                temporal vector will be appended to the input features.
            bidirectional: If True, becomes a bidirectional LSTM/GRU.
            cell_type: Either 'gru' or 'lstm'.

        """
        super().__init__()
        if cell_type not in {"gru", "lstm"}:
            raise ValueError(f"cell_type must be either 'gru' or 'lstm'. Found {cell_type}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.num_layers = num_layers
        if cell_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim + temporal_dim,
                hidden_size=hidden_dim,
                dropout=dropout if num_layers > 1 else 0.0,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=input_dim + temporal_dim,
                hidden_size=hidden_dim,
                dropout=dropout if num_layers > 1 else 0.0,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )

    def forward(self, input_features: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Forward pas of the Temporal module.

        Args:
        ----
          input_features: Input tensor of shape (batch_size, seq_len, input_dim)
          timestamps: Timestamp tensor of shape (batch_size, seq_len, temporal_dim)

        Returns:
        -------
          torch.Tensor: Output tensor of shape (batch_size, hidden_dim).

        """
        inputs = torch.concat([input_features, timestamps], dim=-1)
        if self.cell_type == "gru":
            _, final_hidden_state = self.encoder(inputs)
        else:
            _, (final_hidden_state, _) = self.encoder(inputs)

        final_hidden_state = final_hidden_state.mean(axis=0)
        return final_hidden_state
