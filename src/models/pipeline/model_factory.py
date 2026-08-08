"""
Model factory for creating autoencoder models.

This module handles the creation and initialization of different autoencoder
architectures based on configuration parameters.
"""

import torch
import torch.nn as nn
from loguru import logger
from diffusers import DDPMScheduler

from src.models.networks.autoencoder import (
    ConvAutoencoder,
    ResidualAutoencoder,
    SmoothResidualAutoencoder,
    UnetAutoencoder,
    UnetWithEncoderWrapper,
    LinknetWithEncoderWrapper,
    UnetEnsemble,
    SwinUNETRWrapper,
    MonaiAutoencoderWrapper,
)
from monai.networks.nets import UNet
from src.models.networks.dual_encoder_model import (
    DualEncoderModel,
    CrossModalDualEncoderModel,
)
from src.models.networks.mlp import PatchMLP, SmoothPatchMLP
from src.models.networks.diffusion_models import (
    DiffusionReconstructor,
    DiffusionReconstructorWrapper,
)

from .config import ModelConfig


class ModelFactory:
    """Factory class for creating autoencoder models."""

    def __init__(self, config: ModelConfig):
        """
        Initialize the model factory.

        Parameters
        ----------
        config : ModelConfig
            Configuration object containing model parameters.
        """
        self.config = config
        self.ddpm_scheduler = None

    def create_model(self) -> nn.Module:
        """
        Create and initialize the autoencoder model based on configuration.

        Returns
        -------
        nn.Module
            Initialized autoencoder model.
        """
        # Get activation functions
        activation_fn = getattr(nn, self.config.activation)
        final_activation_fn = None
        if self.config.final_activation:
            final_activation_fn = getattr(nn, self.config.final_activation)

        # Create base model
        model = self._create_base_model(activation_fn, final_activation_fn)

        # Apply residual wrappers if configured
        model = self._apply_residual_wrappers(model)

        # Apply DataParallel for multi-GPU (except for diffusion models)
        if self.config.autoencoder_type != "Diffusion":
            model = torch.nn.DataParallel(model, device_ids=self.config.device_ids)

        # Move to device
        model.to(self.config.device)

        # Log model information
        self._log_model_info(model)

        return model

    def _create_base_model(self, activation_fn, final_activation_fn) -> nn.Module:
        """Create the base model without wrappers."""
        if self.config.autoencoder_type == "Conv":
            return self._create_conv_autoencoder(activation_fn, final_activation_fn)
        elif self.config.autoencoder_type == "Unet":
            return self._create_unet_autoencoder(activation_fn, final_activation_fn)
        elif self.config.autoencoder_type == "Linknet":
            return self._create_linknet_autoencoder(final_activation_fn)
        elif self.config.autoencoder_type == "PatchMLP":
            return self._create_patch_mlp()
        elif self.config.autoencoder_type == "SmoothPatchMLP":
            return self._create_smooth_patch_mlp()
        elif self.config.autoencoder_type == "UnetEnsemble":
            return self._create_unet_ensemble(activation_fn, final_activation_fn)
        elif self.config.autoencoder_type == "Diffusion":
            return self._create_diffusion_model()
        elif self.config.autoencoder_type == "SwinUNETR":
            return self._create_swin_unetr()
        elif self.config.autoencoder_type == "DualEncoder":
            return self._create_dual_encoder(activation_fn)
        elif self.config.autoencoder_type == "MonaiAutoencoder":
            return self._create_monai_autoencoder()
        elif self.config.autoencoder_type == "MonaiUnet":
            model = self._create_monai_unet()
            logger.info("Initialized MONAI UNet for autoencoder.")
            return model
        else:
            raise ValueError(
                f"Unsupported autoencoder type: {self.config.autoencoder_type}"
            )

    def _create_conv_autoencoder(
        self, activation_fn, final_activation_fn
    ) -> ConvAutoencoder:
        """Create ConvAutoencoder model."""
        model = ConvAutoencoder(
            in_channels=self.config.in_channels,
            encoder_channels=self.config.encoder_channels,
            decoder_channels=self.config.decoder_channels,
            kernel_size=self.config.kernel_size,
            stride=self.config.stride,
            padding=self.config.padding,
            activation=activation_fn,
            final_activation=final_activation_fn,
        )
        logger.info(
            f"Initialized ConvAutoencoder with {len(model.encoder)} encoder layers"
        )
        return model

    def _create_monai_autoencoder(self) -> nn.Module:
        """Create MONAI Autoencoder model."""
        model = MonaiAutoencoderWrapper(
            spatial_dims=2,
            in_channels=self.config.in_channels,
            out_channels=self.config.in_channels,
            channels=self.config.encoder_channels,
            strides=self.config.stride,
            kernel_size=self.config.kernel_size,
            up_kernel_size=self.config.kernel_size,
            num_res_units=self.config.num_res_units,
            act=self.config.activation,
            dropout=self.config.dropout,
            norm=self.config.norm,
            bias=True,
        )
        logger.info("Initialized MONAI Autoencoder.")
        return model
    
    def _create_monai_unet(self) -> nn.Module:
        """Create MONAI UNet model."""
        model = UNet(
            spatial_dims=2,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            channels=self.config.encoder_channels,
            strides=self.config.stride,
            kernel_size=self.config.kernel_size,
            num_res_units=self.config.num_res_units,
            act=self.config.activation,
            dropout=self.config.dropout,
            norm=self.config.norm,
        )
        logger.info("Initialized MONAI UNet for autoencoder.")
        return model

    def _create_unet_autoencoder(self, activation_fn, final_activation_fn) -> nn.Module:
        """Create UNet autoencoder model."""
        if self.config.encoder_name is not None:
            model = UnetWithEncoderWrapper(
                encoder_name=self.config.encoder_name,
                encoder_depth=self.config.unet_depth,
                in_channels=self.config.in_channels,
                final_activation_fn=final_activation_fn,
            )
            logger.info(f"Initialized Unet with encoder {self.config.encoder_name}")
        else:
            model = UnetAutoencoder(
                in_channels=self.config.in_channels,
                base_channels=self.config.base_channels,
                depth=self.config.unet_depth,
                kernel_size=self.config.kernel_size,
                activation=activation_fn,
                final_activation=final_activation_fn,
                use_batchnorm=self.config.use_batchnorm,
            )
            logger.info(
                f"Initialized UnetAutoencoder with base channels {self.config.base_channels}"
            )
        return model

    def _create_linknet_autoencoder(
        self, final_activation_fn
    ) -> LinknetWithEncoderWrapper:
        """Create Linknet autoencoder model."""
        model = LinknetWithEncoderWrapper(
            encoder_name=self.config.encoder_name,
            encoder_depth=self.config.unet_depth,
            in_channels=self.config.in_channels,
            final_activation_fn=final_activation_fn,
        )
        logger.info(f"Initialized LinkNet with encoder {self.config.encoder_name}")
        return model

    def _create_patch_mlp(self) -> PatchMLP:
        """Create PatchMLP model."""
        model = PatchMLP(
            in_channels=self.config.in_channels,
            patch_size=self.config.patch_mlp_config.get("patch_size", 8),
            stride=self.config.patch_mlp_config.get("stride", None),
            mlp_hidden_dims=self.config.patch_mlp_config.get(
                "mlp_hidden_dims", (1024, 512)
            ),
            mlp_dropout=self.config.patch_mlp_config.get("mlp_dropout", 0.1),
        )
        logger.info("Initialized PatchMLP for patch-wise processing.")
        return model

    def _create_smooth_patch_mlp(self) -> SmoothPatchMLP:
        """Create SmoothPatchMLP model."""
        model = SmoothPatchMLP(
            in_channels=self.config.in_channels,
            patch_size=self.config.patch_mlp_config.get("patch_size", 8),
            stride=self.config.patch_mlp_config.get("stride", None),
            mlp_hidden_dims=self.config.patch_mlp_config.get(
                "mlp_hidden_dims", (1024, 512)
            ),
            mlp_dropout=self.config.patch_mlp_config.get("mlp_dropout", 0.1),
        )
        logger.info("Initialized SmoothPatchMLP for smooth patch-wise processing.")
        return model

    def _create_unet_ensemble(self, activation_fn, final_activation_fn) -> UnetEnsemble:
        """Create UnetEnsemble model."""
        model = UnetEnsemble(
            num_molecules=self.config.in_channels,
            base_channels=self.config.base_channels,
            depth=self.config.unet_depth,
            kernel_size=self.config.kernel_size,
            activation=activation_fn,
            final_activation=final_activation_fn,
            use_batchnorm=self.config.use_batchnorm,
        )
        logger.info(
            f"Initialized UnetEnsemble with {self.config.in_channels} "
            f"molecule-specific networks, base channels {self.config.base_channels}"
        )
        return model

    def _create_diffusion_model(self) -> DiffusionReconstructorWrapper:
        """Create Diffusion model."""
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.diffusion_config.get(
                "num_train_timesteps", 1000
            ),
            beta_start=self.config.diffusion_config.get("beta_start", 0.0001),
            beta_end=self.config.diffusion_config.get("beta_end", 0.02),
        )
        diffusion_model = DiffusionReconstructor(
            in_channels=self.config.in_channels,
            scheduler=self.ddpm_scheduler,
            num_steps=self.config.diffusion_config.get("num_steps", 50),
            unet_config=self.config.diffusion_config.get("unet_config", None),
        )
        model = DiffusionReconstructorWrapper(diffusion_model)
        logger.info("Initialized Diffusion autoencoder")
        return model

    def _create_swin_unetr(self) -> SwinUNETRWrapper:
        """Create SwinUNETR model."""
        if self.config.swin_unetr_config is None:
            self.config.swin_unetr_config = dict()
            logger.warning("No SwinUNETR config provided. Using default parameters.")

        model = SwinUNETRWrapper(
            in_channels=self.config.in_channels,
            out_channels=self.config.in_channels,
            patch_size=self.config.swin_unetr_config.get("patch_size", 2),
            depths=self.config.swin_unetr_config.get("depths", (2, 2, 2, 2)),
            num_heads=self.config.swin_unetr_config.get("num_heads", (3, 6, 12, 24)),
            window_size=self.config.swin_unetr_config.get("window_size", 7),
            qkv_bias=self.config.swin_unetr_config.get("qkv_bias", True),
            mlp_ratio=self.config.swin_unetr_config.get("mlp_ratio", 4.0),
            feature_size=self.config.swin_unetr_config.get("feature_size", 24),
            norm_name=self.config.swin_unetr_config.get("norm_name", "instance"),
            drop_rate=self.config.swin_unetr_config.get("drop_rate", 0.0),
            attn_drop_rate=self.config.swin_unetr_config.get("attn_drop_rate", 0.0),
            dropout_path_rate=self.config.swin_unetr_config.get(
                "dropout_path_rate", 0.0
            ),
            patch_norm=self.config.swin_unetr_config.get("patch_norm", False),
            downsample=self.config.swin_unetr_config.get("downsample", "merging"),
            spatial_dims=self.config.swin_unetr_config.get("spatial_dims", 2),
            use_v2=self.config.swin_unetr_config.get("use_v2", False),
        )
        logger.info("Initialized SwinUNETR for autoencoder.")
        return model

    def _create_dual_encoder(self, activation_fn) -> nn.Module:
        if self.config.dual_encoder_config.get("cross_modal", False):
            model = CrossModalDualEncoderModel(
                molecule_channels=self.config.in_channels,
                hsi_in_channels=self.config.dual_encoder_config.get(
                    "hsi_in_channels", 15
                ),
                hsi_out_channels=self.config.dual_encoder_config.get(
                    "hsi_out_channels", 302
                ),
                base_channels=self.config.base_channels,
                depth=self.config.dual_encoder_config.get("depth", 4),
                activation=activation_fn,
                use_batchnorm=self.config.use_batchnorm,
            )
            logger.info("Initialized CrossModalDualEncoderModel.")
            return model
        else:
            model = DualEncoderModel(
                molecule_channels=self.config.in_channels,
                hsi_in_channels=self.config.dual_encoder_config.get(
                    "hsi_in_channels", 15
                ),
                hsi_out_channels=self.config.dual_encoder_config.get(
                    "hsi_out_channels", 302
                ),
                base_channels=self.config.base_channels,
                depth=self.config.dual_encoder_config.get("depth", 4),
                target_size=self.config.dual_encoder_config.get(
                    "target_size", self.config.center_crop_size
                ),
                activation=activation_fn,
                use_batchnorm=self.config.use_batchnorm,
            )
            logger.info("Initialized DualEncoderModel.")
            return model

    def _apply_residual_wrappers(self, model: nn.Module) -> nn.Module:
        """Apply residual wrappers if configured."""
        if self.config.use_smooth_residual_autoencoder:
            model = SmoothResidualAutoencoder(base_ae=model, C=self.config.in_channels)
            logger.info("Using SmoothResidualAutoencoder with smooth residuals.")
        elif self.config.use_residual_autoencoder:
            model = ResidualAutoencoder(base_ae=model)
            logger.info(
                "Using ResidualAutoencoder to predict delta instead of absolute values."
            )
        return model

    def _log_model_info(self, model: nn.Module):
        """Log model parameter information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f"Model created with {total_params:,} parameters "
            f"({trainable_params:,} trainable)"
        )

    def get_ddpm_scheduler(self):
        """Get the DDPM scheduler for diffusion models."""
        return self.ddpm_scheduler
