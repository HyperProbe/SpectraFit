from typing import Any, Dict, Dict, Optional, Tuple
from diffusers import UNet2DModel, DDPMScheduler
import torch


class DiffusionReconstructor(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        scheduler: DDPMScheduler,
        num_steps: int = 50,
        unet_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # Build the UNet to expect concatenated inputs:
        # The UNet will receive [noisy_full_maps, clean_reduced_maps] during training
        # and [noisy_full_maps, clean_reduced_maps] during inference

        base_config = {
            "sample_size": 224,
            "in_channels": in_channels * 2,  # noisy_full + clean_reduced channels
            "out_channels": in_channels,  # predict noise with same channels as full
        }

        final_cfg = dict(base_config)
        if unet_config:
            final_cfg.update(unet_config)

        self.unet = UNet2DModel(**final_cfg)
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.in_channels = in_channels

    def forward(self, reduced_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
          reduced_maps: (B, H, W, C_red)  – reduced wavelength reconstructions
        Returns:
          denoised full maps: (B, H, W, C_full)
        """
        # For inference: we generate full maps conditioned on reduced maps
        if self.training:
            # During training, this should not be called
            raise RuntimeError(
                "Forward method should not be called during training. Use the training step in model_pipeline."
            )

        B, H, W, C_red = reduced_maps.shape
        device = reduced_maps.device

        # 1) permute to (B, C_red, H, W)
        reduced_maps = reduced_maps.permute(0, 3, 1, 2)

        # 2) configure scheduler for a num_steps run
        self.scheduler.set_timesteps(self.num_steps)
        # extract the descending timesteps as Python ints
        timesteps = self.scheduler.timesteps.tolist()

        # 3) initialize x as full noise at highest noise level
        C_full = self.in_channels 
        x = torch.randn((B, C_full, H, W), device=device)

        # 4) denoise through the scheduled timesteps (descending)
        with torch.no_grad():  # Only disable gradients during inference
            for t in timesteps:
                # concatenate the noisy full map with the reduced map
                x_and_cond = torch.cat(
                    [x, reduced_maps], dim=1
                )  # (B, C_full + C_red, H, W)

                # predict the noise residual (t can be passed as an int)
                model_out = self.unet(sample=x_and_cond, timestep=t, return_dict=True)
                noise_pred = model_out.sample

                # perform one denoising step
                x = self.scheduler.step(noise_pred, t, x).prev_sample

        # return to (B, H, W, C_full)
        return x.permute(0, 2, 3, 1)

    def predict_noise(
        self, x_t: torch.Tensor, reduced_maps: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise for training. This is the method that should be called during training.

        Args:
            x_t: (B, C_full, H, W) - noisy full maps
            reduced_maps: (B, C_red, H, W) - clean reduced maps
            timestep: (B,) - timestep for each sample
        Returns:
            noise_pred: (B, C_full, H, W) - predicted noise
        """
        # Concatenate noisy full maps with clean reduced maps
        x_and_cond = torch.cat([x_t, reduced_maps], dim=1)  # (B, C_full + C_red, H, W)

        # Predict noise using UNet
        model_out = self.unet(sample=x_and_cond, timestep=timestep, return_dict=True)
        return model_out.sample


class DiffusionReconstructorWrapper(torch.nn.Module):
    """
    Wrapper around DiffusionReconstructor that provides a consistent interface
    for both training and inference in the ModelPipeline.
    """

    def __init__(self, diffusion_model: DiffusionReconstructor):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, reduced_maps: torch.Tensor) -> torch.Tensor:
        """
        Interface for ModelPipeline. During training this should not be called.
        During inference, this will perform the full denoising process.
        """
        return self.diffusion_model(reduced_maps)

    @property
    def unet(self):
        """Provide access to the UNet for training"""
        return self.diffusion_model.unet

    def predict_noise(
        self, x_t: torch.Tensor, reduced_maps: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Method for training - predict noise"""
        return self.diffusion_model.predict_noise(x_t, reduced_maps, timestep)
