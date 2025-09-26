"""The dataset module for training the GAN."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SWIAutoRegressiveDataset(Dataset):
    """The dataset for the training the GAN.

    Concatenates the input_maps to the target_values at each input step.
    """

    def __init__(
        self,
        maps_feats: np.ndarray,
        target_maps: np.ndarray,
        mask: np.ndarray,
        timestamps: np.ndarray,
        num_input_steps: int,
        num_target_steps: int,
    ) -> None:
        """Dataset for the maps.

        Args:
        ----
            maps_feats: The input maps.
            target_maps: The maps to predict.
            mask: A 2D boolean mask representing the region of interest.
                Of shape (1, H, W).
            timestamps: The time vectors corresponding to each map.
            num_input_steps: Number of input timesteps for the data generator.
            num_target_steps: Number of timesteps in the future to predict.

        """
        self.num_input_steps = num_input_steps
        self.num_target_steps = num_target_steps
        self.mask = torch.tensor(mask)
        self.samples = self._build_samples(
            torch.tensor(maps_feats) * self.mask[None, ...],
            torch.tensor(target_maps) * self.mask[None, ...],
            torch.tensor(timestamps),
        )

    def __len__(self) -> int:
        """Size of the dataset.

        Since a sequence of 'num_input_maps' is sent to the model,
        """
        return len(self.samples)

    def _build_samples(
        self, input_maps: torch.Tensor, target_maps: torch.Tensor, timestamps: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Build samples from the dataset."""
        samples = [
            {
                "input_maps": torch.concat(
                    [
                        input_maps[i - self.num_input_steps : i],
                        target_maps[i - self.num_input_steps : i],
                    ],
                    dim=1,
                ),
                "target_maps": target_maps[i : i + self.num_target_steps],
                "input_timestamps": timestamps[i - self.num_input_steps : i],
                "output_timestamps": timestamps[i : i + self.num_target_steps],
            }
            for i in range(self.num_input_steps, len(input_maps) - self.num_target_steps)
        ]
        return samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        return {
            "input_maps": self.samples[idx]["input_maps"],
            "input_timestamps": self.samples[idx]["input_timestamps"],
            "output_timestamps": self.samples[idx]["output_timestamps"],
            "target_maps": self.samples[idx]["target_maps"],
        }
