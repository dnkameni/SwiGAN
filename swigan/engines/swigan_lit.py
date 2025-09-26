"""The lightning module for the GAN."""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule
from torch import autograd, nn

from swigan.models.discriminator.discriminator import (
    BaseDiscriminator,
    FrameDiscriminator,
    PatchGANDiscriminator,
)
from swigan.models.generator.decoder import FrameDecoder
from swigan.models.generator.encoder import FrameEncoder
from swigan.models.generator.lstm_decoder import TemporalDecoder
from swigan.models.generator.lstm_encoder import TemporalEncoder


class SWIGAN(LightningModule):
    """Lightning module for training the GAN.

    The model is trained with a Wasserstein loss with gradient penalty.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_map_dims: list[int],
        temporal_dims: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        dropout: float,
        apply_center_block: bool,
        z_dim: int,
        bidirectional: bool,
        lr: float,
        weight_decay: float,
        loss_fn: nn.Module | str,
        optim: torch.optim.Optimizer,
        normalization: str,
        num_critic_iterations_per_epoch: int = 5,
        lambda_penalty: float = 10,
        image_distance_weight: float = 1.5,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        scheduler_threshold: float = 0.001,
    ) -> None:
        """Initialize the input arguments.

        Args:
        ----
            input_channels: Number of input channels of the input maps.
            output_channels: Number of output channels of the predicted maps.
            input_map_dims: A tuple (H, W) specifying the height and width of the
                input maps.
            temporal_dims: The dimension of the time vectors.
            encoder_channels: The output channels of the convolution blocks
                for the frame encoder.
            decoder_channels: The output channels of the convolution blocks
                for the frame decoder.
            dropout: The dropout rate to apply after each convolutional block.
            apply_center_block: Whether to apply the center block at the end of the frame
                encoding.
            z_dim: The input dimension of the noise vector and the output dimension of the
                temporal decoder.
            bidirectional: Whether to use a bidirectional LSTM for the temporal encoder/decoder.
            lr: Learning rate.
            weight_decay: The weight decay.
            loss_fn: The loss function to apply as an added pixel reconstruction term.
            optim: The optimizer.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
            num_critic_iterations_per_epoch: Number of critic updates per epoch.
            lambda_penalty: The weight to apply to the gradient penalty of the
                critic loss.
            image_distance_weight: The weight to apply to the pixel reconstruction
                loss term.
            scheduler_factor: Factor of the scheduler.
            scheduler_patience: How often to update the learning rate through
                the scheduler.
            scheduler_threshold: The minimum threshold on the monitored metric over
                which the learning rate is updated.

        """
        super().__init__()
        self.frame_encoder = FrameEncoder(
            in_channels=input_channels + output_channels,
            out_channels=encoder_channels,
            dropout=dropout,
            normalization=normalization,
            apply_center_block=apply_center_block,
        )

        self.frame_decoder = FrameDecoder(
            input_dim=z_dim,
            input_map_dims=input_map_dims,
            output_dim=output_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
            normalization=normalization,
        )

        self.temporal_encoder = TemporalEncoder(
            input_dim=encoder_channels[-1],
            temporal_dim=temporal_dims,
            hidden_dim=z_dim,
            bidirectional=bidirectional,
        )

        self.temporal_decoder = TemporalDecoder(
            input_dim=z_dim,
            temporal_dim=temporal_dims,
            hidden_dim=z_dim,
            bidirectional=bidirectional,
        )

        self.base_critic = BaseDiscriminator(
            input_channels=output_channels, temporal_dim=temporal_dims
        )
        self.patch_critic = PatchGANDiscriminator(input_channels=64)
        self.frame_critic = FrameDiscriminator(input_channels=64)

        if isinstance(loss_fn, str):
            if loss_fn == "l1":
                self.loss_fn = nn.L1Loss(reduction="mean")
            elif loss_fn == "mse":
                self.loss_fn = nn.MSELoss(reduction="mean")
            else:
                raise NotImplementedError(f"Unknown loss function: {loss_fn}")
        else:
            self.loss_fn = loss_fn()

        self.automatic_optimization = False
        self.save_hyperparameters(ignore="loss_fn")

    def gradient_penalty(
        self,
        real_maps: torch.Tensor,
        fake_maps: torch.Tensor,
        timesteps: torch.Tensor,
        is_patch_critic: bool,
    ) -> torch.Tensor:
        """Gradient penalty to apply to the Wasserstein loss.

        Args:
        ----
            real_maps: The target maps. A tensor of shape
                (batch_size, output_length, output_channels, H, W).
            fake_maps: The maps generated by the generator.
                Of shape (batch_size, output_length, output_channels, H, W).
            timesteps: The output timestamps vectors. Of shape
                (batch_size, output_length, temporal_dims).
            is_patch_critic: Whether the gradient penalty is applied to the
                patchGAN critic or the frame critic.

        Returns:
        -------
            The gradient penalty.

        """
        batch_size, output_length = real_maps.shape[:2]
        device = real_maps.device

        alpha = torch.rand(batch_size, output_length, 1, 1, 1, device=device)
        interpolated = alpha * real_maps + (1 - alpha) * fake_maps
        interpolated = (
            interpolated.clone().detach().requires_grad_(True)
        )  # Make it a leaf tensor with grad

        if is_patch_critic:
            d_interpolated = self.patch_critic(self.base_critic(interpolated, timesteps))
        else:
            d_interpolated = self.frame_critic(self.base_critic(interpolated, timesteps))
        d_interpolated = d_interpolated.requires_grad_(True)
        # Use scalar output for autograd
        gradients = autograd.grad(
            outputs=d_interpolated.sum(),  # Ensure scalar output
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, output_length, -1)
        gradient_penalty = (
            self.hparams.lambda_penalty * ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        )
        return gradient_penalty

    def generator_step(
        self, batch: dict[str, torch.Tensor], z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single generator step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            z: Noise vector. Of shape (batch_size, input_channels).

        Returns:
        -------
            The generator loss and the pixel distance loss.

        """
        (input_maps, target_maps, input_timestamps, output_timestamps) = (
            batch["input_maps"],
            batch["target_maps"],
            batch["input_timestamps"],
            batch["output_timestamps"],
        )

        fake_maps = self(
            input_maps=input_maps,
            input_timestamps=input_timestamps,
            output_timestamps=output_timestamps,
            noise_vector=z,
        )
        base_critic_fake = self.base_critic(fake_maps, output_timestamps)
        patch_critic_fake = self.patch_critic(base_critic_fake)
        frame_critic_fake = self.frame_critic(base_critic_fake)
        pixel_distance_loss = self.loss_fn(fake_maps, target_maps)
        loss_generator = (
            -torch.mean(patch_critic_fake)
            - torch.mean(frame_critic_fake)
            + self.hparams.image_distance_weight * pixel_distance_loss
        )
        return loss_generator, pixel_distance_loss

    def critic_step(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """Run a single critic step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            z: Noise vector. Of shape (batch_size, input_channels).

        Returns:
        -------
            The Wasserstein loss.

        """
        input_maps, target_maps, input_timestamps, output_timestamps = (
            batch["input_maps"],
            batch["target_maps"],
            batch["input_timestamps"],
            batch["output_timestamps"],
        )
        # Generate fake images

        fake_imgs = self(
            input_maps=input_maps,
            input_timestamps=input_timestamps,
            output_timestamps=output_timestamps,
            noise_vector=z,
        ).detach()
        base_critic_real = self.base_critic(target_maps, output_timestamps)
        base_critic_fake = self.base_critic(fake_imgs, output_timestamps)

        patch_critic_real = self.patch_critic(base_critic_real)
        patch_critic_fake = self.patch_critic(base_critic_fake)
        frame_critic_real = self.frame_critic(base_critic_real)
        frame_critic_fake = self.frame_critic(base_critic_fake)

        if self.training:
            gradient_penalty = self.gradient_penalty(
                target_maps, fake_imgs, output_timestamps, is_patch_critic=True
            )
            gradient_penalty += self.gradient_penalty(
                target_maps, fake_imgs, output_timestamps, is_patch_critic=False
            )
            loss_critic = (
                (torch.mean(patch_critic_fake) - torch.mean(patch_critic_real))
                + (torch.mean(frame_critic_fake) - torch.mean(frame_critic_real))
                + gradient_penalty
            )
        else:
            loss_critic = (torch.mean(patch_critic_fake) - torch.mean(patch_critic_real)) + (
                torch.mean(frame_critic_fake) - torch.mean(frame_critic_real)
            )

        return loss_critic

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a single training step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            batch_idx: Index of the batch.

        Returns:
        -------
            The generator loss.

        """
        batch_size = batch["target_maps"].shape[0]
        opt_generator, opt_critic = self.optimizers()
        z = torch.randn(batch_size, self.hparams.z_dim, device=batch["target_maps"].device)

        # Train Critic multiple times
        for _ in range(self.hparams.num_critic_iterations_per_epoch):
            opt_critic.zero_grad()
            loss_critic = self.critic_step(batch, z)
            self.manual_backward(loss_critic)
            opt_critic.step()

        # Train Generator
        opt_generator.zero_grad()
        loss_generator, pixel_distance_loss = self.generator_step(batch, z)
        self.manual_backward(loss_generator)
        opt_generator.step()

        # Logging
        self.log_dict(
            {
                "train/critic_loss": loss_critic,
                "train/generator_loss": loss_generator,
                "train/generator_pixel_distance_loss": pixel_distance_loss,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return loss_generator

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a single validation step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            batch_idx: Index of the batch.

        Returns:
        -------
            The generator loss.

        """
        batch_size = batch["input_maps"].shape[0]
        z = torch.randn(batch_size, self.hparams.z_dim, device=batch["input_maps"].device)
        loss_critic = self.critic_step(batch, z)
        loss_generator, pixel_distance_loss = self.generator_step(batch, z)
        # Logging
        self.log_dict(
            {
                "val/critic_loss": loss_critic,
                "val/generator_loss": loss_generator,
                "val/generator_pixel_distance_loss": pixel_distance_loss,
            },
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss_generator

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Runa single test step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            batch_idx: Index of the batch.

        Returns:
        -------
            The generator loss.

        """
        batch_size = batch["input_maps"].shape[0]
        z = torch.randn(batch_size, self.hparams.z_dim, device=batch["input_maps"].device)
        loss_critic = self.critic_step(batch, z)
        loss_generator, pixel_distance_loss = self.generator_step(batch, z)
        # Logging
        self.log_dict(
            {
                "test/critic_loss": loss_critic,
                "test/generator_loss": loss_generator,
                "test/generator_pixel_distance_loss": pixel_distance_loss,
            },
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss_generator

    def forward(
        self,
        input_maps: torch.Tensor,
        input_timestamps: torch.Tensor,
        output_timestamps: torch.Tensor,
        noise_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forward pass.

        Args:
        ----
            input_maps: The input maps as torch Tensors. Of shape
                (batch_size, input_length, input_channels, height, width)
            input_timestamps: The timesteps vectors corresponding to each input map.
                Must be of shape (batch_size, input_length, temporal_dim).
            output_timestamps: The timesteps vectors corresponding to outputs.
                Must be of shape (batch_size, output_length, temporal_dim).
            noise_vector: A noise vector for stochasticity in the predictions. If
                None a vector will be generated from the standard normal distribution.
                Must be of shape (batch_size, z_dim).

        Returns:
        -------
            The generated output maps of shape
            (batch_size, output_length, output_channels, height, width).


        """
        outputs = self.encode_frames(
            input_maps
        )  # (batch_size, num_timesteps, encoder_output_channels)
        outputs = self.generate_trajectory(
            outputs, input_timestamps, output_timestamps, noise_vector
        )  # (batch_size, output_length, decoder_input_channels)
        outputs = self.decode_frames(outputs)  # (batch_size, output_length, output_channels, H, W)
        return outputs

    def encode_frames(self, input_maps: torch.Tensor) -> torch.Tensor:
        """Pass the input maps through the frame encoder.

        Args:
        ----
            input_maps: The input maps as torch Tensors. Of shape
                (batch_size, input_length, input_channels, height, width)

        Returns:
        -------
            The encoded frames. Of shape (batch_size, input_length, out_channels).

        """
        outputs = self.frame_encoder(
            input_maps.view(-1, *input_maps.shape[2:])
        )  # (batch_size*num_timesteps, encoder_output_channels)
        outputs = outputs.view(
            *input_maps.shape[:2], -1
        )  # (batch_size, num_timesteps, encoder_output_channels)
        return outputs

    def decode_frames(self, input_features: torch.Tensor) -> torch.Tensor:
        """Pass the input maps through the frame decoder.

        Args:
        ----
            input_features: Input features coming from the temporal decoder.
                Of shape (batch_size, output_length, z_dim)

        Returns:
        -------
            The decoded frames. Of shape (batch_size, output_length, output_channels, height, width)

        """
        outputs = input_features.reshape(-1, input_features.shape[-1])[
            ..., None, None
        ]  # (batch_size*num_timesteps, decoder_input_channels, 1, 1)
        outputs = self.frame_decoder(
            outputs
        )  # (batch_size*num_timesteps, decoder_output_channels, H, W)
        outputs = outputs.reshape(
            *input_features.shape[:2], *outputs.shape[1:]
        )  # (batch_size, num_timesteps, encoder_output_channels, H, W)
        return outputs

    def generate_trajectory(
        self,
        feature_maps: torch.Tensor,
        input_timestamps: torch.Tensor,
        output_timestamps: torch.Tensor,
        noise_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate output features to pass through the frame decoder.

        Args:
        ----
            feature_maps: The feature maps from the frame decoder. Of shape
                (batch_size, input_length, encoder_output_channels).
            input_timestamps: The timesteps vectors corresponding to each input map.
                Must be of shape (batch_size, input_length, temporal_dim).
            output_timestamps: The timesteps vectors corresponding to outputs.
                Must be of shape (batch_size, output_length, temporal_dim).
            noise_vector: A noise vector for stochasticity in the predictions. If
                None a vector will be generated from the standard normal distribution.
                Must be of shape (batch_size, z_dim).

        Returns:
        -------
            A tensor of shape (batch_size, output_length, z_dim).

        """
        outputs = self.temporal_encoder(feature_maps, input_timestamps)  # (batch_size, z_dim)
        outputs = self.temporal_decoder(
            outputs, output_timestamps, noise_vector
        )  # (batch_size, output_length, z_dim)
        return outputs

    def on_validation_epoch_end(self) -> None:
        """To run at the end of the validation epoch."""
        # Update scheduler
        val_pixel_distance_loss = self.trainer.callback_metrics["val/generator_pixel_distance_loss"]
        # val_loss_critic = self.trainer.callback_metrics["val/critic_loss"]

        scheduler_generator, scheduler_critic = self.lr_schedulers()
        scheduler_generator.step(val_pixel_distance_loss)
        # scheduler_critic.step(val_loss_critic)

    # def on_train_epoch_end(self):
    #   if (self.current_epoch + 1) % 2 == 0:
    #     print(f"Epoch: {self.current_epoch}")
    #     train_idx = np.random.choice(len(train_dataset))
    #     val_idx = np.random.choice(len(val_dataset))
    #     with torch.no_grad():
    #       fig, axes = plt.subplots(2, 3)
    #       to_plot = [train_dataset[train_idx], val_dataset[val_idx]]
    #       for i, image_label in enumerate(to_plot):
    #         image = image_label["image"].to(device)
    #         label = image_label["label"].to(device)
    #         res = F.sigmoid(unet_model(image.unsqueeze(0)))
    #         axes[i,0].imshow(image.squeeze().detach().cpu().numpy(), cmap="Greys")
    #         axes[i,1].imshow((res.squeeze().detach().cpu().numpy()), cmap="Greys")
    #         axes[i,2].imshow(label.squeeze().detach().cpu().numpy(), cmap="Greys")
    #       plt.show()

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[Any]]:
        """Configure the optimizers."""
        optimizer_generator = self.hparams.optim(
            list(self.frame_encoder.parameters())
            + list(self.frame_decoder.parameters())
            + list(self.temporal_encoder.parameters())
            + list(self.temporal_decoder.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        optimizer_critic = self.hparams.optim(
            list(self.base_critic.parameters())
            + list(self.patch_critic.parameters())
            + list(self.frame_critic.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_generator,
            "min",
            patience=self.hparams.scheduler_patience,
            factor=self.hparams.scheduler_factor,
            threshold=self.hparams.scheduler_threshold,
        )
        scheduler_critic = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_critic,
            "min",
            patience=self.hparams.scheduler_patience,
            factor=self.hparams.scheduler_factor,
            threshold=self.hparams.scheduler_threshold,
        )

        return [optimizer_generator, optimizer_critic], [scheduler_generator, scheduler_critic]
