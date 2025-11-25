"""Module containing the InfoGAN network."""

import torch
from torch import nn


class InfoGANNetwork(nn.Module):
    """The InfoGAN network for mutual information maximization.

    Enforces interpretable latent space representations.
    Check InfoGAN: Interpretable Representation Learning by
    Information Maximizing Generative Adversarial Nets (https://arxiv.org/pdf/1606.03657).

    This module takes the output from the base discriminator and out puts the latent_code
    (output of the temporal encoder) and the temporal vector (month).
    """

    def __init__(self, input_channels: int, latent_code_dim: int, temporal_dim: int) -> None:
        """Initialize the parameters.

        Args:
        ----
            input_channels: Number of input channels.
            latent_code_dim: Dimension of the laten code from the temporal encoder.
            temporal_dim: The dimension of the temporal vector. Should be 12 for the
                number of months.

        """
        super().__init__()
        self.latent_code_encoder = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.ReLU(),
            nn.Linear(input_channels, latent_code_dim),
        )
        self.timestamps_encoder = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.ReLU(),
            nn.Linear(input_channels, temporal_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
        ----
          inputs: The input frames. Must be of shape
              (batch_size, seq_len, channels, height, width).

        Returns:
        -------
          A tensor of shape (batch_size, seq_len, 12) for the
          timestamps encoder and (batch_size, seq_len, latent_code_dim) for the
          latent code encoder.

        """
        recovered_timestamps = self.timestamps_encoder(inputs.mean(dim=(-2, -1)))
        recovered_latent_code = self.latent_code_encoder(inputs.mean(dim=(-4, -2, -1)))
        return recovered_timestamps, recovered_latent_code
