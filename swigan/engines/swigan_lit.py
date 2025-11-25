"""The lightning module for the GAN."""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule
from torch import autograd, nn

from modules.discriminator import (
    BaseDiscriminator,
    FrameDiscriminator,
    PatchGANDiscriminator,
    TemporalDiscriminator,
)
from modules.discriminator.info_gan_network import InfoGANNetwork
from modules.generator import FrameDecoder, FrameEncoder, TemporalDecoder, TemporalEncoder
from modules.utils import ModelFlavour


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
        latent_code_dim: int,
        spatial_dropout: float,
        temporal_dropout: float,
        apply_center_block: bool,
        z_dim: int,
        num_temporal_layers: int,
        rnn_cell_type: str,
        bidirectional: bool,
        lr: float,
        weight_decay: float,
        loss_fn: nn.Module | str,
        optim: torch.optim.Optimizer,
        normalization: str,
        num_critic_iterations_per_epoch: int = 5,
        lambda_penalty: float = 10,
        image_distance_weight: float = 1.5,
        max_epochs: int = 100,
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
            latent_code_dim: The output dimension of the frame and temporal encoders.
            spatial_dropout: The dropout rate to apply after each convolutional block
                in the frame level modules.
            temporal_dropout: The dropout rate in the rnn encoder and decoder.
            apply_center_block: Whether to apply the center block at the end of the frame
                encoding.
            z_dim: The input dimension of the noise vector and the output dimension of the
                temporal decoder.
            num_temporal_layers: The number of layers in the temporal encoder/decoder.
            rnn_cell_type: Either 'gru' or 'lstm'.
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
            max_epochs: Number of training epochs. Used to set up the Tmax parameter of the
                CosineAnnealingLR scheculer.

        """
        super().__init__()
        self.time_embedding = nn.Embedding(
            num_embeddings=12,  # number of months
            embedding_dim=temporal_dims,
        )
        self.frame_encoder = FrameEncoder(
            in_channels=input_channels + output_channels,
            out_channels=encoder_channels,
            output_dim=latent_code_dim,
            dropout=spatial_dropout,
            normalization=normalization,
            apply_center_block=apply_center_block,
        )

        self.frame_decoder = FrameDecoder(
            input_dim=latent_code_dim,
            input_map_dims=input_map_dims,
            output_dim=output_channels,
            decoder_channels=decoder_channels,
            dropout=spatial_dropout,
            normalization=normalization,
        )

        self.temporal_encoder = TemporalEncoder(
            input_dim=latent_code_dim,
            temporal_dim=temporal_dims,
            hidden_dim=latent_code_dim,
            dropout=temporal_dropout,
            num_layers=num_temporal_layers,
            cell_type=rnn_cell_type,
            bidirectional=bidirectional,
        )

        self.temporal_decoder = TemporalDecoder(
            input_dim=latent_code_dim,
            temporal_dim=temporal_dims,
            noise_dim=z_dim,
            hidden_dim=latent_code_dim,
            dropout=temporal_dropout,
            num_layers=num_temporal_layers,
            cell_type=rnn_cell_type,
            bidirectional=bidirectional,
        )

        self.base_critic = BaseDiscriminator(input_channels=output_channels)
        self.patch_critic = PatchGANDiscriminator(input_channels=64)
        self.frame_critic = FrameDiscriminator(input_channels=64)
        self.temporal_critic = TemporalDiscriminator(input_channels=64)

        self.Q_net = InfoGANNetwork(
            input_channels=64,
            latent_code_dim=latent_code_dim,
            temporal_dim=12,
        )

        if isinstance(loss_fn, str):
            if loss_fn == "l1":
                self.loss_fn = nn.SmoothL1Loss(reduction="sum")
            elif loss_fn == "mse":
                self.loss_fn = nn.MSELoss(reduction="sum")
            else:
                raise NotImplementedError(f"Unknown loss function: {loss_fn}")
        else:
            self.loss_fn = loss_fn()

        self.automatic_optimization = False
        self.save_hyperparameters(ignore="loss_fn")

    @property
    def description(self) -> str:
        """A string summarizing the main characteristics of the model."""
        if self.hparams.encoder_channels[0] == ModelFlavour.Small.value[0][0]:
            model_flavour = "small"
        elif self.hparams.encoder_channels[0] == ModelFlavour.Medium.value[0][0]:
            model_flavour = "medium"
        else:
            model_flavour = "large"

        bidirectional = "bidirectional" if self.hparams.bidirectional else "unidirectional"

        return (
            f"{model_flavour}_{bidirectional}_{self.hparams.rnn_cell_type}"
            f"_noise_dim{self.hparams.z_dim}_latent_dim{self.hparams.latent_code_dim}_"
            "input_6_target_12_no_temporal_disc"
        )

    def gradient_penalty(
        self,
        real_maps: torch.Tensor,
        fake_maps: torch.Tensor,
        critic_type: str,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient penalty to apply to the Wasserstein loss.

        Args:
        ----
            real_maps: The target maps. A tensor of shape
                (batch_size, output_length, output_channels, H, W).
            fake_maps: The maps generated by the generator.
                Of shape (batch_size, output_length, output_channels, H, W).
            critic_type: Whether the gradient penalty is applied to the
                patchGAN critic or the frame critic or the temporal critic.
                Must be one of "patch", "frame", "temporal"
            alpha: The interpolation factor.

        Returns:
        -------
            The gradient penalty.

        """
        batch_size, output_length = real_maps.shape[:2]
        interpolated = alpha * real_maps + (1 - alpha) * fake_maps
        interpolated = (
            interpolated.clone().detach().requires_grad_(True)
        )  # Make it a leaf tensor with grad

        if critic_type == "patch":
            d_interpolated = self.patch_critic(self.base_critic(interpolated))
        elif critic_type == "frame":
            d_interpolated = self.frame_critic(self.base_critic(interpolated))
        elif critic_type == "temporal":
            d_interpolated = self.temporal_critic(self.base_critic(interpolated))
        else:
            raise RuntimeError(
                f"Unknown critic type: {critic_type}. "
                f"Please provide one of 'patch', 'frame', 'temporal'."
            )
        d_interpolated = d_interpolated.requires_grad_(True)
        # Use scalar output for autograd
        gradients = autograd.grad(
            outputs=d_interpolated.sum(),
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
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
        (input_maps, target_maps, input_timestamps, output_timestamps, mask) = (
            batch["input_maps"],
            batch["target_maps"],
            batch["input_timestamps"],
            batch["output_timestamps"],
            batch["mask"],
        )

        fake_maps, latent_code = self(
            input_maps=input_maps,
            input_timestamps=input_timestamps,
            output_timestamps=output_timestamps,
            mask=mask,
            noise_vector=z,
        )

        base_critic_fake = self.base_critic(fake_maps)
        patch_critic_fake = self.patch_critic(base_critic_fake)
        frame_critic_fake = self.frame_critic(base_critic_fake)
        temporal_critic_fake = self.temporal_critic(base_critic_fake)
        fake_timestamps, fake_latent_code = self.Q_net(base_critic_fake)
        # Pixel distance loss
        pixel_distance_loss = self.hparams.image_distance_weight * self.loss_fn(
            fake_maps, target_maps
        )

        # Penalty on the reconstructed timestamps
        timestamps_penalty = 0.1 * torch.nn.functional.cross_entropy(
            fake_timestamps.view(-1, fake_timestamps.shape[-1]), output_timestamps.view(-1)
        )

        # Penalty on the reconstructed latent code
        latent_code_penalty = 0.1 * torch.nn.functional.mse_loss(fake_latent_code, latent_code)

        loss_generator = (
            -torch.mean(patch_critic_fake)
            - torch.mean(frame_critic_fake)
            - torch.mean(temporal_critic_fake)
            + pixel_distance_loss
            + timestamps_penalty
            + latent_code_penalty
        )

        return loss_generator, pixel_distance_loss

    def critic_step(
        self, batch: dict[str, torch.Tensor], z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single critic step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            z: Noise vector. Of shape (batch_size, input_channels).

        Returns:
        -------
            The Wasserstein loss with gradient penalty.

        """
        input_maps, target_maps, input_timestamps, output_timestamps, mask = (
            batch["input_maps"],
            batch["target_maps"],
            batch["input_timestamps"],
            batch["output_timestamps"],
            batch["mask"],
        )
        device = input_maps.device
        # Generate fake images

        fake_imgs, _ = self(
            input_maps=input_maps,
            input_timestamps=input_timestamps,
            output_timestamps=output_timestamps,
            mask=mask,
            noise_vector=z,
        )
        fake_imgs = fake_imgs.detach()
        base_critic_real = self.base_critic(target_maps)
        base_critic_fake = self.base_critic(fake_imgs)

        patch_critic_real = self.patch_critic(base_critic_real)
        patch_critic_fake = self.patch_critic(base_critic_fake)
        frame_critic_real = self.frame_critic(base_critic_real)
        frame_critic_fake = self.frame_critic(base_critic_fake)
        temporal_critic_real = self.temporal_critic(base_critic_real)
        temporal_critic_fake = self.temporal_critic(base_critic_fake)

        # Compute the losses for each discriminator and aggregate them
        loss_critic_patch = torch.mean(patch_critic_fake) - torch.mean(patch_critic_real)
        loss_critic_frame = torch.mean(frame_critic_fake) - torch.mean(frame_critic_real)
        loss_critic_temporal = torch.mean(temporal_critic_fake) - torch.mean(temporal_critic_real)

        total_loss_critic = loss_critic_patch + loss_critic_frame + loss_critic_temporal

        if self.training:
            # add gradient penalty
            batch_size, output_length = target_maps.shape[:2]
            alpha = torch.rand(batch_size, output_length, 1, 1, 1, device=device)
            gradient_penalty = self.gradient_penalty(
                target_maps, fake_imgs, critic_type="patch", alpha=alpha
            )
            gradient_penalty += self.gradient_penalty(
                target_maps,
                fake_imgs,
                critic_type="frame",
                alpha=alpha,
            )
            gradient_penalty += self.gradient_penalty(
                target_maps,
                fake_imgs,
                critic_type="temporal",
                alpha=alpha,
            )
            total_loss_critic += gradient_penalty

        return total_loss_critic, loss_critic_patch, loss_critic_frame, loss_critic_temporal

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
            loss_critic, loss_patch_critic, loss_frame_critic, loss_temporal_critic = (
                self.critic_step(batch, z)
            )
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
                "train/total_critic_loss": loss_critic,
                "train/critic_patch_loss": loss_patch_critic,
                "train/critic_frame_loss": loss_frame_critic,
                "train/critic_temporal_loss": loss_temporal_critic,
                "train/gradient_penalty": loss_critic
                - (loss_patch_critic + loss_frame_critic + loss_temporal_critic),
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
        loss_critic, loss_patch_critic, loss_frame_critic, loss_temporal_critic = self.critic_step(
            batch, z
        )
        loss_generator, pixel_distance_loss = self.generator_step(batch, z)
        # Logging
        self.log_dict(
            {
                "val/total_critic_loss": loss_critic,
                "val/critic_patch_loss": loss_patch_critic,
                "val/critic_frame_loss": loss_frame_critic,
                "val/critic_temporal_loss": loss_temporal_critic,
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
        loss_critic, loss_patch_critic, loss_frame_critic, loss_temporal_critic = self.critic_step(
            batch, z
        )
        loss_generator, pixel_distance_loss = self.generator_step(batch, z)
        # Logging
        self.log_dict(
            {
                "test/total_critic_loss": loss_critic,
                "test/critic_patch_loss": loss_patch_critic,
                "test/critic_frame_loss": loss_frame_critic,
                "test/critic_temporal_loss": loss_temporal_critic,
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
        mask: torch.Tensor,
        noise_vector: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass.

        Args:
        ----
            input_maps: The input maps as torch Tensors. Of shape
                (batch_size, input_length, input_channels, height, width)
            input_timestamps: The timesteps vectors corresponding to each input map.
                Must be of shape (batch_size, input_length, temporal_dim).
            output_timestamps: The timesteps vectors corresponding to outputs.
                Must be of shape (batch_size, output_length, temporal_dim).
            mask: The mask of shape (1, 1, 1, height, width).
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
        outputs, latent_code = self.generate_trajectory(
            outputs, input_timestamps, output_timestamps, noise_vector
        )  # (batch_size, output_length, decoder_input_channels)
        outputs = self.decode_frames(outputs)  # (batch_size, output_length, output_channels, H, W)
        return outputs * mask, latent_code

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        input_timestamps_embeddings = self.time_embedding(input_timestamps).squeeze(dim=-2)
        output_timestamps_embeddings = self.time_embedding(output_timestamps).squeeze(dim=-2)
        latent_code = self.temporal_encoder(
            feature_maps, input_timestamps_embeddings
        )  # (batch_size, latent_code_dim)
        outputs = self.temporal_decoder(
            latent_code, output_timestamps_embeddings, noise_vector
        )  # (batch_size, output_length, latent_code_dim)
        return outputs, latent_code

    def on_validation_epoch_end(self) -> None:
        """To run at the end of the validation epoch."""
        # Update scheduler
        scheduler_generator, scheduler_critic = self.lr_schedulers()
        scheduler_generator.step()
        scheduler_critic.step()

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[Any]]:
        """Configure the optimizers."""
        optimizer_generator = self.hparams.optim(
            list(self.frame_encoder.parameters())
            + list(self.frame_decoder.parameters())
            + list(self.temporal_encoder.parameters())
            + list(self.temporal_decoder.parameters())
            + list(self.Q_net.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        optimizer_critic = self.hparams.optim(
            list(self.base_critic.parameters())
            + list(self.patch_critic.parameters())
            + list(self.frame_critic.parameters())
            + list(self.temporal_critic.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_generator,
            T_max=self.hparams.max_epochs,
            eta_min=0,
        )
        scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_critic,
            T_max=self.hparams.max_epochs,
            eta_min=0,
        )

        return [optimizer_generator, optimizer_critic], [scheduler_generator, scheduler_critic]
