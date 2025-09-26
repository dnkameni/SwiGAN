"""Module containing an LSTM-based temporal frame decoder."""

from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """An LSTM-based encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        temporal_dim: int = 1,
        bidirectional: bool = True,
    ) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_dim: Number of dimensions of the input features.
            hidden_dim: Number of dimensions of the LSTM hidden features.
            temporal_dim: The dimensions of the temporal vector. This
                temporal vector will be appended to the input features.
            bidirectional: If True, becomes a bidirectional LSTM.

        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm_encoder = nn.LSTM(
            input_size=input_dim + temporal_dim,
            hidden_size=hidden_dim,
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
          torch.Tensor: Output tensor of shape(batch_size, seq_len, hidden_dim).

        """
        inputs = torch.concat([input_features, timestamps], dim=-1)
        _, (final_hidden_state, _) = self.lstm_encoder(inputs)
        if self.bidirectional:
            final_hidden_state = (
                final_hidden_state[0] + final_hidden_state[1]
            )  # (batch_size, hidden_dim)
        return final_hidden_state
