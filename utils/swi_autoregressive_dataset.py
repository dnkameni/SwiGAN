"""Module containing dataset utils for the forecast SWIGAN model."""

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
            torch.tensor(timestamps).long(),
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
            "mask": self.mask[None, ...],
        }


def build_train_val_test_datasets(
    input_maps: np.ndarray,
    target_maps: np.ndarray,
    timesteps: np.ndarray,
    mask: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    num_input_steps: int,
    num_output_steps_train: int,
    num_output_steps_val: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """Build tge train validation and test datasets.

    Args:
    ----
        input_maps: The input rasters.
        target_maps: The output rasters with the target variable as channels.
        timesteps: The time vectors containing the indices of the months.
        mask: A boolean mask representing the region of interest.
        train_ratio: Ratio of training split.
        val_ratio: Ratio of the validation split.
        num_input_steps: Number of timesteps to consider as input.
            Used for training, validation and test datasets.
        num_output_steps_train: Number of output timesteps in the training dataset.
        num_output_steps_val: Number of output timesteps in the validation and test dataset.

    Returns:
    -------
        Train, validation and test datasets.

    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            "Please provide 'train_ratio' and 'val_ratio' such that "
            "train_ratio + val_ratio < 1.0."
        )
    train_length = int(train_ratio * len(input_maps))
    val_length = int(val_ratio * len(input_maps))

    (
        maps_train,
        maps_val,
        maps_test,
        targets_train,
        targets_val,
        targets_test,
        timestamps_train,
        timestamps_val,
        timestamps_test,
    ) = (
        input_maps[:train_length],
        input_maps[train_length : train_length + val_length],
        input_maps[train_length + val_length :],
        target_maps[:train_length],
        target_maps[train_length : train_length + val_length],
        target_maps[train_length + val_length :],
        timesteps[:train_length],
        timesteps[train_length : train_length + val_length],
        timesteps[train_length + val_length :],
    )

    # Normalize train data
    mean_value, std_value = maps_train.mean(axis=(0, 1, 2)), maps_train.std(axis=(0, 1, 2))
    maps_train = (maps_train - mean_value) / (std_value + 1e-8)
    maps_val = (maps_val - mean_value) / (std_value + 1e-8)
    maps_test = (maps_test - mean_value) / (std_value + 1e-8)

    # Split the dataset
    train_dataset = SWIAutoRegressiveDataset(
        maps_feats=maps_train,
        target_maps=targets_train,
        timestamps=timestamps_train,
        mask=mask,
        num_input_steps=num_input_steps,
        num_target_steps=num_output_steps_train,
    )
    val_dataset = SWIAutoRegressiveDataset(
        maps_feats=maps_val,
        target_maps=targets_val,
        timestamps=timestamps_val,
        mask=mask,
        num_input_steps=num_input_steps,
        num_target_steps=num_output_steps_val,
    )
    test_dataset = SWIAutoRegressiveDataset(
        maps_feats=maps_test,
        target_maps=targets_test,
        timestamps=timestamps_test,
        mask=mask,
        num_input_steps=num_input_steps,
        num_target_steps=num_output_steps_val,
    )
    return train_dataset, val_dataset, test_dataset
