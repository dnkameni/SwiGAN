"""Differentiable augmentation for Discriminator regularization.

See https://arxiv.org/abs/2006.10738
"""

import torch
import torch.nn.functional as F  # noqa


def DiffAugment(x: torch.Tensor, policy: str = "", channels_first: bool = True) -> torch.Tensor:  # noqa
    """Apply differentiable augmentation to the input given a policy.

    Copied from https://github.com/mit-han-lab/data-efficient-gans

    Args:
    ----
        x: Input Tensor to apply diff augment.
        policy: A comma separated string specifying the transformations to apply.
        channels_first: If true expects a tensor of shape (batch, channels, H, W),
            else (batch, H, W, channels).

    Returns:
    -------
        The augmented input.

    """
    if policy:
        if not channels_first:
            raise NotImplementedError("Only support channels_first=True")
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        x = x.contiguous()
    return x


def rand_translation(x: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    """Slight modification of the Vanilla DiffAugment translation.

    Handles spatio-temporal data by applying the same transformation to
    all timesteps and all channels of a single batch element.

    Args:
    ----
        x: Input Tensor to apply diff augment.
        ratio: Ratio of the input maps along which the translation will be performed.

    Returns:
    -------
        The augmented input.

    """
    shift_x, shift_y = int(x.size(-2) * ratio + 0.5), int(x.size(-1) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(-2), dtype=torch.long, device=x.device),
        torch.arange(x.size(-1), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(-2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(-1) + 1)
    if x.ndim == 4:  # (batch_size, channels, H, W)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = (
            x_pad.permute(0, 2, 3, 1)
            .contiguous()[grid_batch, grid_x, grid_y]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
    else:  # (batch_size, timesteps, channels, H, W)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        x = (
            x_pad.permute(0, 3, 4, 1, 2)
            .contiguous()[grid_batch, grid_x, grid_y]
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    return x


def rand_cutout(x: torch.Tensor, ratio: float = 0.3) -> torch.Tensor:
    """Slight modification of the Vanilla DiffAugment cutout.

    Handles spatio-temporal data by applying the same transformation to
    all timesteps and all channels of a single batch element.

    Args:
    ----
        x: Input Tensor to apply diff augment.
        ratio: Ratio of the input maps along which the translation will be performed.

    Returns:
    -------
        The augmented input.

    """
    cutout_size = int(x.size(-2) * ratio + 0.5), int(x.size(-1) * ratio + 0.5)
    offset_x = torch.randint(
        0, x.size(-2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    offset_y = torch.randint(
        0, x.size(-1) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(-2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(-1) - 1)
    mask = torch.ones(x.size(0), x.size(-2), x.size(-1), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    if x.ndim == 4:  # (batch_size, channels, H, W)
        x = x * mask.unsqueeze(1)
    else:  # (batch_size, timesteps, channels, H, W)
        x = x * mask.unsqueeze(1).unsqueeze(1)
    return x


AUGMENT_FNS = {
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
