"""Module containing an LSTM-based temporal decoder."""

from __future__ import annotations

import torch
from torch import nn


class TemporalDecoder(nn.Module):
    """An LSTM-based decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        temporal_dim: int,
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
        self.lstm_decoder = nn.LSTM(
            input_size=input_dim + temporal_dim,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        timestamps: torch.Tensor,
        noise_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pas of the Temporal module.

        Args:
        ----
          inputs: Input tensor of shape (batch_size, input_dim)
          timestamps: Timestamp tensor of shape (batch_size, output_length, temporal_dim)
          noise_vector: Noise vector to add to the input must be of shape (batch_size, input_dim).
            Defaults to None.

        Returns:
        -------
          torch.Tensor: Output tensor of shape(batch_size, seq_len, hidden_dim

        """
        output_length = timestamps.shape[1]
        input_sequence = inputs[:, None, :].repeat(1, output_length, 1)

        if noise_vector is None:
            input_sequence += torch.randn_like(input_sequence)
        else:
            if noise_vector.shape != inputs.shape:
                raise RuntimeError(
                    f"'noise_vector' should have same shape as 'inputs'. "
                    f"Found {noise_vector.shape} and {inputs.shape}."
                )
            input_sequence += noise_vector[:, None, :]
        input_sequence = torch.concat([input_sequence, timestamps], dim=-1)
        outputs, (_, _) = self.lstm_decoder(input_sequence)
        if self.bidirectional:
            outputs = outputs[:, :, : self.hidden_dim] + outputs[:, :, self.hidden_dim :]
        return outputs
