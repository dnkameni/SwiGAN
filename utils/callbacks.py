"""Module containing custom training callbacks."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, Trainer
from matplotlib import colors
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


class RandomTimeStepMapCallback(Callback):
    """Callback class for plotting randomly generated maps during training.

    This class will randomly select k maps from the train dataset and k
    maps from the validation dataset and plot the model predictions.
    """

    def __init__(
        self,
        save_dir: str,
        input_statistics: dict[str, np.ndarray],
        train_dataset: Dataset,
        val_dataset: Dataset,
        print_every_n_epochs: int = 20,
        num_months_per_plot: int = 6,
    ) -> None:
        """Initialize the callback.

        Args:
        ----
            save_dir: Folder where the figures will be saved.
            input_statistics: Dictionary containing the train targets mean and std
                for inference.
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            print_every_n_epochs: Plotting frequency.
            num_months_per_plot: Number of months to show in the same plot.

        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.input_statistics = input_statistics
        self.print_frequency = print_every_n_epochs
        self.num_months_per_plot = num_months_per_plot

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Print the maps at the end of the validation epoch."""
        plt.close("all")
        if (pl_module.current_epoch + 1) % self.print_frequency == 0:
            targets_mean_value, targets_std_value = (
                self.input_statistics["targets_mean"],
                self.input_statistics["targets_std"],
            )
            with torch.no_grad():
                for [name, dataset_p] in [
                    ("Train", self.train_dataset),
                    ("Validation", self.val_dataset),
                ]:
                    fig, axes = plt.subplots(
                        2, self.num_months_per_plot, figsize=(5 * self.num_months_per_plot, 6)
                    )
                    plt.subplots_adjust(wspace=0.5)
                    for k, idx in enumerate(np.random.choice(len(dataset_p), 6, replace=False)):
                        inputs = dataset_p[idx]

                        targets = inputs["target_maps"].squeeze().detach().cpu().numpy()
                        res = pl_module(
                            inputs["input_maps"].unsqueeze(0).to(pl_module.device),
                            inputs["timestamps"].unsqueeze(0).to(pl_module.device),
                            inputs["mask"].unsqueeze(0).to(pl_module.device),
                        )
                        timestamps = inputs["timestamps"].detach().cpu().numpy()
                        mask = inputs["mask"].squeeze().detach().cpu().numpy()

                        month_name = MONTHS[int(timestamps)]
                        targets_k = (
                            targets * targets_std_value.squeeze() + targets_mean_value.squeeze()
                        ) * mask
                        res_k = (
                            res.squeeze().detach().cpu().numpy() * targets_std_value.squeeze()
                            + targets_mean_value.squeeze()
                        ) * mask
                        merged = np.stack([targets_k, res_k], axis=0)
                        norm = colors.Normalize(vmin=np.min(merged), vmax=np.max(merged))
                        axes[0, k].set_title(f"GT Month {month_name}")
                        p = axes[0, k].imshow(
                            np.where(mask, targets_k, np.nan), cmap="RdYlGn", norm=norm
                        )
                        axes[1, k].set_title(f"Prediction Month {month_name}")
                        p = axes[1, k].imshow(
                            np.where(mask, res_k, np.nan), cmap="RdYlGn", norm=norm
                        )
                        plt.colorbar(p, ax=axes[:, k], shrink=0.4)
                    fig.suptitle(f"{name} maps epoch {pl_module.current_epoch}")
                    plt.savefig(
                        os.path.join(
                            self.save_dir,
                            f"{name}_predictions_epoch{pl_module.current_epoch:04d}.png",
                        ),
                        dpi=200,
                    )
