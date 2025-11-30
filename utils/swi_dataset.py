"""The dataset module for training the GAN."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SWIDataset(Dataset):
    """The dataset for the training the SWIGAN UNet variant.

    Concatenates the input_maps to the target_values at each input step.
    """

    def __init__(
        self,
        maps_feats: np.ndarray,
        target_maps: np.ndarray,
        mask: np.ndarray,
        timestamps: np.ndarray,
        num_input_steps: int,
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

        """
        self.num_input_steps = num_input_steps
        self.mask = torch.tensor(mask).float()
        self.samples = self._build_samples(
            torch.tensor(maps_feats) * self.mask[None, ...],
            torch.tensor(target_maps) * self.mask[None, ...],
            torch.tensor(timestamps.squeeze()).long(),
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
                        input_maps[i],
                        target_maps[i - self.num_input_steps : i, 0],
                    ],
                    dim=0,
                ).float(),
                "target_maps": target_maps[i].float(),
                "timestamps": timestamps[i],
            }
            for i in range(self.num_input_steps, len(input_maps))
        ]
        return samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        return {
            "input_maps": self.samples[idx]["input_maps"],
            "timestamps": self.samples[idx]["timestamps"],
            "target_maps": self.samples[idx]["target_maps"],
            "mask": self.mask,
        }


def build_train_val_test_datasets(
    input_maps: np.ndarray,
    target_maps: np.ndarray,
    timesteps: np.ndarray,
    mask: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    num_input_steps: int,
) -> tuple[dict[str, SWIDataset], dict[str, np.ndarray]]:
    """Build the train validation and test datasets.

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
        Two dictionaries:
            The first one containing Train, validation and test datasets, with keys
            "train", "val", "test.
            The second one containing standardization metrics for the feature maps and
            the target maps.

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

    # Standardize train data
    # First let's remove the mask
    maps_train_without_mask = np.where(mask.squeeze(), maps_train, np.nan)
    maps_val_without_mask = np.where(mask.squeeze(), maps_val, np.nan)
    maps_test_without_mask = np.where(mask.squeeze(), maps_test, np.nan)

    mean_value, std_value = (
        np.nanmean(maps_train_without_mask, axis=(0, -2, -1), keepdims=True),
        np.nanstd(maps_train_without_mask, axis=(0, -2, -1), keepdims=True),
    )
    maps_train = (maps_train_without_mask - mean_value) / std_value
    maps_val = (maps_val_without_mask - mean_value) / std_value
    maps_test = (maps_test_without_mask - mean_value) / std_value

    maps_train = np.where(mask.squeeze(), maps_train, 0.0)
    maps_val = np.where(mask.squeeze(), maps_val, 0.0)
    maps_test = np.where(mask.squeeze(), maps_test, 0.0)

    # # Standardize target data
    targets_train_without_mask = np.where(mask.squeeze(), targets_train, np.nan)
    targets_val_without_mask = np.where(mask.squeeze(), targets_val, np.nan)
    targets_test_without_mask = np.where(mask.squeeze(), targets_test, np.nan)

    targets_mean_value, targets_std_value = (
        np.nanmean(targets_train_without_mask, axis=(0, -2, -1), keepdims=True),
        np.nanstd(targets_train_without_mask, axis=(0, -2, -1), keepdims=True),
    )
    targets_train = (targets_train_without_mask - targets_mean_value) / targets_std_value
    targets_val = (targets_val_without_mask - targets_mean_value) / targets_std_value
    targets_test = (targets_test_without_mask - targets_mean_value) / targets_std_value

    targets_train = np.where(mask.squeeze(), targets_train, 0.0)
    targets_val = np.where(mask.squeeze(), targets_val, 0.0)
    targets_test = np.where(mask.squeeze(), targets_test, 0.0)

    train_dataset = SWIDataset(
        maps_feats=maps_train,
        target_maps=targets_train,
        timestamps=timestamps_train,
        mask=mask,
        num_input_steps=num_input_steps,
    )
    val_dataset = SWIDataset(
        maps_feats=maps_val,
        target_maps=targets_val,
        timestamps=timestamps_val,
        mask=mask,
        num_input_steps=num_input_steps,
    )
    test_dataset = SWIDataset(
        maps_feats=maps_test,
        target_maps=targets_test,
        timestamps=timestamps_test,
        mask=mask,
        num_input_steps=num_input_steps,
    )

    return (
        {"train": train_dataset, "val": val_dataset, "test": test_dataset},
        {
            "feats_mean": mean_value,
            "feats_std": std_value,
            "targets_mean": targets_mean_value,
            "targets_std": targets_std_value,
        },
    )
