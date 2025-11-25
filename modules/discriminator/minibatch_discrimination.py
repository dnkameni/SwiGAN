"""Module containing a minibatch discrimination module."""

import torch
from torch import nn


class MiniBatchDiscrimination(nn.Module):
    """Minibatch discrimination module."""

    def __init__(self, in_features: int, out_features: int, kernel_dims: int) -> None:
        """Minibatch discrimination module.

        See https://arxiv.org/pdf/1606.03498.

        Args:
        ----
            in_features: number of input features
            out_features: number of kernels (output features)
            kernel_dims: dimension for each kernel

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            x: Input tensor of shape (batch_size, channels, H, W).

        """
        batch_size = x.size(0)
        H, W = x.size(2), x.size(3)  # noqa
        x_in = x.contiguous().transpose(3, 1)

        M = x_in @ self.T.view(self.in_features, -1)  # noqa
        M = M.view(  # noqa
            batch_size, W, H, self.out_features, self.kernel_dims
        )  # [batch_size, w, h, out_features, kernel_dims]

        # Compute L1 distance between each pair of samples
        M_expanded_1 = M.unsqueeze(0)  # [1, batch, w, h, out_features, kernel] # noqa
        M_expanded_2 = M.unsqueeze(1)  # [batch, 1, w, h, out_features, kernel] # noqa
        diff = torch.abs(M_expanded_1 - M_expanded_2).sum(5)  # [batch, batch, w, h, out_features]

        exp_neg_diff = torch.exp(-diff)

        mask = 1 - torch.eye(batch_size, device=x.device)
        out = (
            (exp_neg_diff * mask[..., None, None, None]).sum(1).contiguous()
        )  # [batch, w, h, out_features]

        # Concatenate with original features
        return torch.cat([x, out.transpose(3, 1)], dim=1)
