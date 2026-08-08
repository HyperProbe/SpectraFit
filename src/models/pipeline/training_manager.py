"""
Training components for the model pipeline.

This module contains training steps, loss setup, and optimizer configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loguru import logger
from typing import Callable, Optional

from src.models.networks.discriminator import PatchDiscriminator
from src.models.networks.losses import (
    ChannelProjector,
    CompositeLossWithTV,
    PerceptualLoss,
    VGG16Features,
    WeightedMSELossWithSignalPenalty,
    WeightedReconstructionLoss,
    SSIMLoss,
    WeightedSSIMWithL1Loss,
    WeightedSSIMWithMSELoss,
    PerceptualLossWrapper,
)
from .config import ModelConfig


class TrainingManager:
    """Manages training components including loss functions, optimizers, and training steps."""

    def __init__(self, config: ModelConfig, model: nn.Module, ddmp_scheduler=None):
        """
        Initialize the training manager.

        Parameters
        ----------
        config : ModelConfig
            Configuration object.
        model : nn.Module
            The model to train.
        ddmp_scheduler : optional
            DDPM scheduler for diffusion models.
        """
        self.config = config
        self.model = model
        self.ddmp_scheduler = ddmp_scheduler

        # Training components
        self.criterion = None
        self.criterion_hsi = None
        self.optimizer = None
        self.scheduler = None
        self.discriminator = None
        self.optimizer_discriminator = None
        self.channel_projector = None

        # Training step function
        self.train_step = None

    def setup_training(self):
        """Setup all training components."""
        self.setup_loss()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_train_step()

        logger.info(
            f"Training setup complete - Optimizer: {self.config.optimizer}, "
            f"Scheduler: {self.config.scheduler}, Loss: {self.config.loss_function}"
        )

    def get_loss_fn(self, loss_function: str) -> Callable:
        """Get loss function by name (instantiate only the selected one)."""

        def maybe_wrap_tv(loss):
            if self.config.lambda_tv is not None:
                return CompositeLossWithTV(
                    base_loss=loss, lambda_tv=self.config.lambda_tv
                )
            return loss

        lw = self.config.loss_weights
        device = self.config.device

        match loss_function:
            case "MSE":
                loss = nn.MSELoss()
            case "L1":
                loss = nn.L1Loss()
            case "Huber":
                loss = nn.HuberLoss()
            case "weighted":
                loss = WeightedReconstructionLoss(alpha=lw.get("alpha", 0.5))
            case "weighted_with_penalty":
                loss = WeightedMSELossWithSignalPenalty(
                    alpha=lw.get("alpha", 0.5),
                    lambda_weighted_mse=lw.get("lambda_weighted_mse", 1.0),
                    lambda_hbt=lw.get("lambda_hbt", 1.0),
                    lambda_diffCCO=lw.get("lambda_diffCCO", 1.0),
                )
            case "SSIM":
                loss = SSIMLoss(
                    gaussian_kernel=lw.get("gaussian_kernel", True),
                    kernel_size=lw.get("kernel_size", 11),
                    reduction=lw.get("reduction", "elementwise_mean"),
                    device=device,
                )
            case "weighted_ssim_with_mse":
                loss = WeightedSSIMWithMSELoss(
                    alpha=lw.get("alpha", 0.5),
                    gaussian_kernel=lw.get("gaussian_kernel", True),
                    kernel_size=lw.get("kernel_size", 11),
                    reduction=lw.get("reduction", "elementwise_mean"),
                    device=device,
                )
            case "weighted_ssim_with_l1":
                loss = WeightedSSIMWithL1Loss(
                    alpha=lw.get("alpha", 0.5),
                    gaussian_kernel=lw.get("gaussian_kernel", True),
                    kernel_size=lw.get("kernel_size", 11),
                    reduction=lw.get("reduction", "elementwise_mean"),
                    device=device,
                )
            case "perceptual":
                loss = self.get_perceptual_loss()
            case "adversarial":
                loss = self.get_adversarial_loss()
            case "perceptualMonai":
                loss = self.get_perceptual_monai_loss()
            case _:
                raise ValueError(f"Unsupported loss function: {loss_function}")

        return maybe_wrap_tv(loss)

    def setup_loss(self):
        """Setup loss function based on configuration."""
        self.criterion = self.get_loss_fn(self.config.loss_function)
        if self.config.autoencoder_type == "DualEncoder":
            self.criterion_hsi = self.get_loss_fn(
                self.config.dual_encoder_config.get("hsi_loss", "MSE")
            )

    def get_perceptual_loss(self):
        self.channel_projector = ChannelProjector(in_ch=10, out_ch=3).to(
            self.config.device
        )
        self.channel_projector.train()
        vgg_features = VGG16Features(layers=("3", "8", "17")).to(self.config.device)
        criterion = PerceptualLoss(
            projector=self.channel_projector,
            feature_extractor=vgg_features,
            layer_weights=self.config.loss_weights.get("layer_weights", None),
            lambda_mse=self.config.loss_weights.get("lambda_mse", 1.0),
            lambda_perceptual=self.config.loss_weights.get("lambda_perceptual", 0.1),
        )
        return criterion

    def get_adversarial_loss(self):
        self.discriminator = PatchDiscriminator(
            in_ch=self.config.in_channels,
            ndf=self.config.loss_weights.get("ndf", 64),
            n_layers=self.config.loss_weights.get("n_layers", 3),
        ).to(self.config.device)
        self.discriminator.train()

        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.loss_weights.get("lr_disc", 1e-4),
        )

        criterion = WeightedSSIMWithMSELoss(
            alpha=self.config.loss_weights.get("alpha", 0.5),
            gaussian_kernel=self.config.loss_weights.get("gaussian_kernel", True),
            sigma=self.config.loss_weights.get("sigma", 1.5),
            kernel_size=self.config.loss_weights.get("kernel_size", 11),
            reduction=self.config.loss_weights.get("reduction", "elementwise_mean"),
            device=self.config.device,
        )
        return criterion

    def get_perceptual_monai_loss(self):
        self.channel_projector = ChannelProjector(in_ch=10, out_ch=3).to(
            self.config.device
        )
        self.channel_projector.train()
        criterion = PerceptualLossWrapper(
            spatial_dims=2,
            channel_projector=self.channel_projector,
            network_type=self.config.loss_weights.get("network_type", "vgg16"),
            is_fake_3d=False,
            base_loss=WeightedSSIMWithMSELoss(),
            lambda_base=self.config.loss_weights.get("lambda_base", 1.0),
            lambda_perceptual=self.config.loss_weights.get("lambda_perceptual", 0.1),
        ).to(self.config.device)

        logger.info(
            f"Using MONAI's perceptual loss with {self.config.loss_weights.get('network_type', 'vgg16')} features."
        )

        return criterion

    def setup_optimizer(self):
        """Setup optimizer based on configuration."""
        # Determine trainable parameters
        trainable_params = []
        if self.config.autoencoder_type == "Diffusion":
            trainable_params.append(
                {
                    "params": self.model.unet.parameters(),
                    "lr": self.config.learning_rate,
                }
            )
        elif self.config.autoencoder_type == "DualEncoder":
            base = (
                self.model.module
                if isinstance(
                    self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
                )
                else self.model
            )
            if self.config.dual_encoder_config.get("cross_modal", False):
                trainable_params.extend(
                    [
                        {
                            "params": base.encoder.molecule_encoder.parameters(),
                            "lr": self.config.dual_encoder_config.get(
                                "molecule_encoder_lr", self.config.learning_rate
                            ),
                        },
                        {
                            "params": base.encoder.hsi_encoder.parameters(),
                            "lr": self.config.dual_encoder_config.get(
                                "hsi_encoder_lr", self.config.learning_rate
                            ),
                        },
                        {
                            "params": base.molecule_decoder.parameters(),
                            "lr": self.config.dual_encoder_config.get(
                                "molecule_decoder_lr", self.config.learning_rate
                            ),
                        },
                        {
                            "params": base.hsi_decoder.parameters(),
                            "lr": self.config.dual_encoder_config.get(
                                "hsi_decoder_lr", self.config.learning_rate
                            ),
                        },
                    ]
                )
            else:
                trainable_params.extend(
                    [
                        {
                            "params": base.encoder.molecule_encoder.parameters(),
                            "lr": self.config.dual_encoder_config.get(
                                "molecule_encoder_lr", self.config.learning_rate
                            ),
                        },
                        {
                            "params": base.encoder.hsi_encoder.parameters(),
                            "lr": self.config.dual_encoder_config.get(
                                "hsi_encoder_lr", self.config.learning_rate
                            ),
                        },
                        {
                            "params": base.decoder.parameters(),
                            "lr": self.config.dual_encoder_config.get(
                                "decoder_lr", self.config.learning_rate
                            ),
                        },
                    ]
                )
        else:
            trainable_params.append(
                {"params": self.model.parameters(), "lr": self.config.learning_rate}
            )

        # Add channel projector parameters if using perceptual loss
        if self.channel_projector is not None:
            trainable_params.append(
                {
                    "params": self.channel_projector.parameters(),
                    "lr": self.config.learning_rate,
                }
            )

        # Create optimizer
        if self.config.optimizer == "Adam":
            self.optimizer = optim.Adam(trainable_params)
        elif self.config.optimizer == "SGD":
            self.optimizer = optim.SGD(trainable_params)
        elif self.config.optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(trainable_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def setup_scheduler(self):
        """Setup learning rate scheduler based on configuration."""
        if self.config.scheduler == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **self.config.scheduler_params
            )
        elif self.config.scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **self.config.scheduler_params
            )
        elif self.config.scheduler is None:
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

    def setup_train_step(self):
        """Setup train step function based on autoencoder type."""
        if self.config.noise_std > 0:
            logger.info(
                f"Adding Gaussian noise with std {self.config.noise_std} to input data."
            )

        if self.config.autoencoder_type == "Diffusion":
            self.train_step = self._train_step_diffusion
            logger.info("Using diffusion training step.")
        elif self.config.multires_training:
            self.train_step = self._train_step_multires
            logger.info("Using multi-resolution training step.")
        elif self.config.loss_function == "adversarial":
            self.train_step = self._train_step_adversarial
            logger.info("Using adversarial loss training step.")
        elif self.config.autoencoder_type == "DualEncoder":
            if self.config.dual_encoder_config.get("concentration_map_only", False):
                self.train_step = self._train_step_dual_encoder_concentration_map_only
                logger.info(
                    "Using DualEncoder training step with concentration map only."
                )
            else:
                self.train_step = self._train_step_dual_encoder
                logger.info("Using dual encoder training step.")
        elif self.config.ssr:
            self.train_step = self._train_step_ssr
            logger.info("Using spectral super-resolution training step.")
        else:
            self.train_step = self._train_step_ae
            logger.info("Using standard autoencoder training step.")

    def _add_noise(self, input_data) -> torch.Tensor:
        """Add Gaussian noise to the input data if configured."""
        if self.config.noise_std > 0:
            noise = torch.randn_like(input_data) * self.config.noise_std
            return input_data + noise
        return input_data

    def _train_step_ae(self, batch) -> torch.Tensor:
        """Standard autoencoder training step."""
        # Get ground truth concentration maps (target for reconstruction)
        gt_coef = batch["gt"]["coef_list"]  # Shape: (B, H, W, C)
        gt_coef = gt_coef.to(self.config.device).float()

        # For this autoencoder, we use the GT as both input and target
        # In practice, you might want to use reduced_wl data as input
        input_data = batch["reduced_wl"]["coef_list"]  # Shape: (B, H, W, C)
        input_data = input_data.to(self.config.device).float()
        input_data = self._add_noise(input_data)  # Add noise if specified
        target_data = gt_coef

        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(input_data)
        loss = self.criterion(output, target_data)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss
    
    def _train_step_ssr(self, batch) -> torch.Tensor:
        """Spectral super-resolution training step."""
        # Get ground truth high-res HSI cubes (target for reconstruction)
        gt_hsi = batch["hsi_cube_original"]  # Shape: (B,C,H,W)
        gt_hsi = gt_hsi.to(self.config.device).float()

        # Input is the low-res HSI cube
        input_hsi = batch["hsi_cube"]  # Shape: (B,C,H,W)
        input_hsi = input_hsi.to(self.config.device).float()
        input_hsi = self._add_noise(input_hsi)  # Add noise if specified
        target_hsi = gt_hsi

        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(input_hsi)
        loss = self.criterion(output, target_hsi)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss

    def _train_step_adversarial(self, batch) -> torch.Tensor:
        """Adversarial training step: update generator then discriminator."""
        gt_coef = batch["gt"]["coef_list"].to(self.config.device).float()
        input_data = batch["reduced_wl"]["coef_list"].to(self.config.device).float()
        input_data = self._add_noise(input_data)  # Add noise if specified

        # ——— 1) Generator step ———
        self.optimizer.zero_grad()
        fake = self.model(input_data)  # (B,H,W,C)
        # weighted ssim + mse loss
        loss_g = self.criterion(fake, gt_coef)
        # adv loss
        pred_fake = self.discriminator(fake)
        valid = torch.ones_like(pred_fake)
        loss_g_adv = F.binary_cross_entropy_with_logits(pred_fake, valid)
        total_g_loss = (
            loss_g + self.config.loss_weights.get("lambda_adv", 0.001) * loss_g_adv
        )
        total_g_loss.backward()
        self.optimizer.step()

        # ——— 2) Discriminator step ———
        self.optimizer_discriminator.zero_grad()
        pred_real = self.discriminator(gt_coef)
        loss_d_real = F.binary_cross_entropy_with_logits(pred_real, valid)

        pred_fake_det = self.discriminator(fake.detach())
        fake_label = torch.zeros_like(pred_fake_det)
        loss_d_fake = F.binary_cross_entropy_with_logits(pred_fake_det, fake_label)

        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        loss_d.backward()
        self.optimizer_discriminator.step()

        return total_g_loss

    def _train_step_diffusion(self, batch) -> torch.Tensor:
        """Train step for diffusion-based autoencoder."""
        self.model.train()  # Set wrapper to training mode
        gt_coef = batch["gt"]["coef_list"].to(self.config.device).float()
        input_data = batch["reduced_wl"]["coef_list"].to(self.config.device).float()
        B = input_data.shape[0]

        t = torch.randint(
            low=0,
            high=self.ddmp_scheduler.config.num_train_timesteps,
            size=(B,),
            device=self.config.device,
        ).long()

        # Convert to BCHW format
        gt_bchw = gt_coef.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        input_data_bchw = input_data.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Generate noise and create noisy version of ground truth
        noise = torch.randn_like(gt_bchw)
        x_t = self.ddmp_scheduler.add_noise(gt_bchw, noise, t)

        # Use the predict_noise method from our wrapper
        noise_pred = self.model.predict_noise(x_t, input_data_bchw, t)

        # Calculate loss between predicted and actual noise
        loss = self.criterion(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _train_step_multires(self, batch) -> torch.Tensor:
        """Multi-resolution training step."""
        self.model.train()

        # batch is a tuple: (inputs, targets)
        # inputs = [full_tensor_CHW, crop128_tensor_CHW, crop64_tensor_CHW, ...]
        inputs, targets = batch

        self.optimizer.zero_grad()
        total_loss = 0.0

        # Train on each resolution level
        for resolution_idx, (input_tensor, target_tensor) in enumerate(
            zip(inputs, targets)
        ):
            # Convert tensors to device and ensure proper format
            input_tensor = input_tensor.to(self.config.device).float()  # (B, C, H, W)
            target_tensor = target_tensor.to(self.config.device).float()  # (B, C, H, W)

            # Convert from CHW to HWC format for the model (B, C, H, W) -> (B, H, W, C)
            input_hwc = input_tensor.permute(0, 2, 3, 1)
            target_hwc = target_tensor.permute(0, 2, 3, 1)

            # Add noise if specified
            input_hwc = self._add_noise(input_hwc)

            # Forward pass
            output = self.model(input_hwc)  # Output is (B, H, W, C)

            # Calculate loss for this resolution
            loss = self.criterion(output, target_hwc)
            total_loss += loss

        # Backward pass on accumulated loss
        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def _train_step_dual_encoder_concentration_map_only(self, batch) -> torch.Tensor:
        """DualEncoderModel training step using only concentration map input."""
        # Get ground truth concentration maps (target for reconstruction)
        gt_coef = batch["concentration_data"]["gt"]["coef_list"]  # Shape: (B, C, H, W)
        gt_coef = gt_coef.to(self.config.device).float()

        input_concentration_map = batch["concentration_data"]["reduced_wl"]["coef_list"]
        input_concentration_map = input_concentration_map.to(self.config.device).float()
        input_concentration_map = self._add_noise(
            input_concentration_map
        )  # Add noise if specified

        input_hsi = batch["hsi_cube"].to(self.config.device).float()
        self.optimizer.zero_grad()

        output_concentration_map, _ = self.model(input_concentration_map, input_hsi)
        loss_concentration = self.criterion(output_concentration_map, gt_coef)

        loss_concentration.backward()
        self.optimizer.step()

        return loss_concentration

    def _train_step_dual_encoder(self, batch) -> torch.Tensor:
        """DualEncoderModel training step."""
        # Get ground truth concentration maps (target for reconstruction)
        gt_coef = batch["concentration_data"]["gt"]["coef_list"]  # Shape: (B, C, H, W)
        gt_coef = gt_coef.to(self.config.device).float()

        # For this autoencoder, we use the GT as both input and target
        # In practice, you might want to use reduced_wl data as input
        input_concentration_map = batch["concentration_data"]["reduced_wl"]["coef_list"]

        input_concentration_map = input_concentration_map.to(self.config.device).float()
        input_concentration_map = self._add_noise(
            input_concentration_map
        )  # Add noise if specified

        input_hsi = batch["hsi_cube"].to(self.config.device).float()

        target_concentration_map = gt_coef
        target_hsi = batch["hsi_cube_original"].to(self.config.device).float()

        # Forward pass
        self.optimizer.zero_grad()
        output_concentration_map, output_hsi = self.model(
            input_concentration_map, input_hsi
        )
        loss_concentration = self.criterion(
            output_concentration_map, target_concentration_map
        )
        loss_hsi = self.criterion_hsi(output_hsi, target_hsi)

        total_loss = (
            loss_concentration
            + self.config.dual_encoder_config.get("loss_hsi_weight", 0.1) * loss_hsi
        )
        total_loss.backward()
        self.optimizer.step()

        return total_loss
