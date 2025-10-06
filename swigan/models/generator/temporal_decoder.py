"""Module containing an LSTM-based temporal decoder."""

from __future__ import annotations

import torch
from torch import nn


class TemporalDecoder(nn.Module):
    """An LSTM/GRU-based decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        noise_dim: int,
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
            noise_dim: Number of dimensions of the noise vector.
            num_layers: The number of LSTM/GRU layers.
            dropout: The dropout rate.
            temporal_dim: The dimensions of the temporal vector. This
                temporal vector will be appended to the input features.
            bidirectional: If True, becomes a bidirectional LSTM.
            cell_type: Either 'gru' or 'lstm'.

        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.noise_dim = noise_dim
        self.cell_type = cell_type
        if cell_type == "gru":
            self.decoder = nn.GRU(
                input_size=input_dim + noise_dim + temporal_dim,
                hidden_size=hidden_dim,
                dropout=dropout if num_layers > 1 else 0.0,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.decoder = nn.LSTM(
                input_size=input_dim + noise_dim + temporal_dim,
                hidden_size=hidden_dim,
                dropout=dropout if num_layers > 1 else 0.0,
                num_layers=num_layers,
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
          torch.Tensor: Output tensor of shape(batch_size, seq_len, hidden_dim)

        """
        batch_size, output_length = timestamps.shape[:2]
        input_sequence = inputs[:, None, :].repeat(1, output_length, 1)

        if noise_vector is None:
            input_sequence = torch.cat(
                [
                    input_sequence,
                    torch.randn(
                        (batch_size, output_length, self.noise_dim), device=input_sequence.device
                    ),
                ],
                dim=-1,
            )
        else:
            if noise_vector.shape[0] != inputs.shape[0] or noise_vector.ndim != inputs.ndim:
                raise RuntimeError(
                    f"'noise_vector'and 'inputs' should have the same number of dimensions"
                    " with the same dimension at axis=0. "
                    f"Found {noise_vector.shape} and {inputs.shape}."
                )
            input_sequence = torch.cat(
                [input_sequence, noise_vector[:, None, :].repeat(1, output_length, 1)], dim=-1
            )
        input_sequence = torch.concat([input_sequence, timestamps], dim=-1)
        if self.cell_type == "gru":
            outputs, _ = self.decoder(input_sequence)
        else:
            outputs, (_, _) = self.decoder(input_sequence)

        if self.bidirectional:
            outputs = outputs.reshape(
                outputs.shape[0], outputs.shape[1], self.num_layers, self.hidden_dim
            ).mean(axis=-2)
        return outputs
