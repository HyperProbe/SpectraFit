"""
Configuration management for the model pipeline.

This module contains the ModelConfig dataclass with all configuration parameters
for training and validating autoencoders.
"""

import json
import torch
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Literal, Optional, Sequence, Tuple, Union, List
from pathlib import Path

from src.molecules import MoleculeIndex


@dataclass
class ModelConfig:
    """
    Configuration class for autoencoder training.

    For detailed information about loss_weights parameters and their default values,
    see the documentation for the loss_weights field below.
    """

    # Model architecture
    autoencoder_type: Literal[
        "Conv",
        "Unet",
        "Linknet",
        "PatchMLP",
        "SmoothPatchMLP",
        "UnetEnsemble",
        "Diffusion",
        "SwinUNETR",
        "DualEncoder",
        "MonaiAutoencoder",
        "MonaiUnet"
    ] = "Conv"  # Conv, Unet, UnetEnsemble, etc.
    ssr: bool = False  # Use autoencoder for spectral super-resolution
    use_residual_autoencoder: bool = False
    use_smooth_residual_autoencoder: bool = False  # Use SmoothResidualAutoencoder
    encoder_name: str = None
    in_channels: int = 10  # Number of molecule concentration channels
    out_channels: int = 10  # Number of output channels
    chosen_molecules: Optional[list] = (
        None  # List of molecule names from MoleculeIndex, e.g., ['HBO2', 'HB', 'FAT']
    )
    merge_channels_to_signals: Optional[List[str]] = None
    base_channels: int = 64  # Base channels for Unet
    unet_depth: int = 5  # Depth of Unet
    use_batchnorm: bool = True  # Use batch normalization in Unet
    batch_print: int = 10  # Print every n batches
    fused_helicoid_config: Optional[Dict] = None
    channels_first: bool = False

    diffusion_config: Optional[Dict] = None
    swin_unetr_config: Optional[Dict] = None
    dual_encoder_config: Optional[Dict] = None
    encoder_channels: list = None
    decoder_channels: list = None
    kernel_size: int = 3
    stride: int | Sequence[int] = 2
    norm: Literal["batch", "instance", "layer", "group", "none"] = "batch"
    num_res_units: int = 0  # Number of residual units in MONAI Autoencoder
    dropout: float = 0.0  # Dropout rate

    padding: int = 1
    activation: Literal["ReLU", "LeakyReLU", "ELU", "PReLU"] = (
        "ReLU"  # ReLU, LeakyReLU, ELU, etc.
    )
    final_activation: Optional[str] = None  # sigmoid, tanh, etc.

    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 10
    optimizer: Literal["Adam", "SGD", "RMSprop"] = "Adam"  # Adam, SGD, RMSprop
    scheduler: Optional[str] = "StepLR"  # StepLR, ReduceLROnPlateau, None
    scheduler_params: dict = None
    multires_training: bool = False  # Use multi-resolution
    multires_crop_sizes: list = field(default_factory=lambda: [128, 64])
    deterministic: bool = False  # Use deterministic training

    # Regularization
    lambda_tv: Union[float, None] = None  # Total variation regularization
    # Loss function
    loss_function: Literal[
        "MSE",
        "L1",
        "Huber",
        "weighted",
        "weighted_with_penalty",
        "SSIM",
        "weighted_ssim_with_mse",
        "weighted_ssim_with_l1",
        "adversarial",
        "perceptual",
        "perceptualMonai",
    ] = "MSE"  # MSE, L1, Huber
    loss_weights: Optional[dict] = None  # For weighted losses
    patch_mlp_config: Optional[dict] = None
    noise_std: float = 0.0  # Standard deviation of noise for DAE
    """
    Optional dictionary containing loss-specific parameters. Each loss function has different
    configurable parameters with their respective default values:
    
    For 'weighted' loss function:
        - alpha (float): Weight balance parameter, default=0.5
    
    For 'weighted_with_penalty' loss function:
        - alpha (float): Weight balance parameter, default=0.5
        - lambda_weighted_mse (float): Weight for weighted MSE component, default=1.0
        - lambda_hbt (float): Weight for HBT signal penalty, default=1.0
        - lambda_diffCCO (float): Weight for diffCCO signal penalty, default=1.0
    
    For 'SSIM' loss function:
        - gaussian_kernel (bool): Use Gaussian kernel for SSIM, default=True
        - kernel_size (int): Size of the kernel for SSIM computation, default=11
        - reduction (str): Reduction method for SSIM, default="elementwise_mean"
    
    For 'weighted_ssim_with_mse' loss function:
        - alpha (float): Weight balance between SSIM and MSE, default=0.5
        - gaussian_kernel (bool): Use Gaussian kernel for SSIM, default=True
        - kernel_size (int): Size of the kernel for SSIM computation, default=11
        - reduction (str): Reduction method for SSIM, default="elementwise_mean"
    
    For 'perceptual' loss function:
        - layer_weights (list): Weights for different VGG layers, default=None (equal weights)
        - lambda_mse (float): Weight for MSE component, default=1.0
        - lambda_perceptual (float): Weight for perceptual component, default=0.1
    
    For 'adversarial' loss function:
        - alpha (float): Weight balance for base criterion (SSIM+MSE), default=0.5
        - gaussian_kernel (bool): Use Gaussian kernel for SSIM, default=True
        - sigma (float): Sigma parameter for Gaussian kernel, default=1.5
        - kernel_size (int): Size of the kernel for SSIM computation, default=11
        - reduction (str): Reduction method for SSIM, default="elementwise_mean"
        - ndf (int): Number of discriminator filters, default=64
        - n_layers (int): Number of discriminator layers, default=3
        - lr_disc (float): Learning rate for discriminator optimizer, default=1e-4
        - lambda_adv (float): Weight for adversarial loss component, default=0.001
    """

    # Data
    reduced_wl_dir: str = (
        "../results/biopsy2_spectral_unmixing/spectral_unmixing_avgpool_4_coarseness_1_left_500_right_900_reduced_wl_set"
    )
    gt_dir: str = (
        "../results/biopsy2_spectral_unmixing/spectral_unmixing_avgpool_4_coarseness_1_left_500_right_900_t1_all_molecules"
    )
    split_ratios: tuple = (0.8, 0.1, 0.1)
    random_seed: int = 42
    num_workers: int = 8
    pin_memory: bool = True
    sample_type: Literal["helicoid", "biopsy2"] = "biopsy2"  # helicoid, biopsy
    validation_ids: Optional[list] = None  # List of patient IDs for validation
    test_ids: Optional[list] = None  # List of patient IDs for testing
    center_crop_size: Optional[Tuple[int, int]] = None  # Center crop size (H, W)
    normalization_dir: Optional[str] = None  # Directory for normalization statistics
    image_wise_normalization: bool = False  # Apply image-wise normalization
    augmentation_ratio: float = 0.0  # Ratio of augmented samples in training set
    random_crop_size: Optional[Tuple[int, int]] = None  # Random crop size (H, W)
    train_on_original_res: bool = False  # Train on original resolution divisible by 8
    evaluate_full_resolution: bool = False  # Evaluate on full resolution
    inferer_mode: Literal["sliding_window", "whole"] = "whole"
    sliding_window_roi_size: Tuple[int, int] = (64, 64)
    sliding_window_overlap: float = 0.25  # Overlap ratio for sliding window inference

    # Training options
    early_stopping_patience: int = 20
    save_best_model: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    log_coarseness: int = 8  # Coarseness for logging molecule reconstructions

    # Logging and output
    project: str = "biopsy-autoencoder"
    experiment_name: str = "autoencoder_experiment"
    output_dir: str = "../results"
    log_level: str = "INFO"
    with_wandb: bool = True  # Use Weights & Biases for logging
    log_image_epoch: int = 1

    # Device
    device: str = "cuda:0"  # auto, cpu, cuda
    device_ids: Optional[List[int]] = None

    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128, 256]
        if self.scheduler_params is None:
            self.scheduler_params = {"step_size": 30, "gamma": 0.1}

        # Handle chosen molecules: update in_channels if specified
        if self.chosen_molecules is not None:
            # Validate molecule names
            valid_molecules = [name for name, _ in MoleculeIndex.__members__.items()]
            for mol in self.chosen_molecules:
                if mol not in valid_molecules:
                    raise ValueError(
                        f"Invalid molecule name: {mol}. Valid options: {valid_molecules}"
                    )
            self.in_channels = len(self.chosen_molecules)

        if self.merge_channels_to_signals is not None:
            if self.in_channels != len(self.merge_channels_to_signals):
                raise ValueError(
                    f"Length of merge_channels_to_signals ({len(self.merge_channels_to_signals)}) "
                    f"must match in_channels ({self.in_channels})."
                )

        if self.dual_encoder_config is None:
            self.dual_encoder_config = {}

        if self.fused_helicoid_config is None:
            self.fused_helicoid_config = {}
        if self.autoencoder_type == "DualEncoder":
            self.channels_first = True

        if self.center_crop_size or self.random_crop_size:
            self.sliding_window_roi_size = (
                self.center_crop_size or self.random_crop_size or (216, 216)
            )

        # Handle device selection
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_ids is None:
            device_id = (
                int(self.device.split(":")[-1]) if "cuda" in self.device else None
            )
            self.device_ids = [device_id] if device_id is not None else [0]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        config = cls(**config_dict)
        config.__post_init__()  # Ensure post-init processing
        return config

    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        """Load config from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_default_config(
    reduced_wl_dir: str,
    gt_dir: str,
    experiment_name: str = "autoencoder_default",
    output_dir: str = "./results",
) -> ModelConfig:
    """
    Create a default configuration for autoencoder training.

    Parameters
    ----------
    reduced_wl_dir : str
        Path to reduced wavelength concentration maps directory.
    gt_dir : str
        Path to ground truth concentration maps directory.
    experiment_name : str, optional
        Name for the experiment.
    output_dir : str, optional
        Output directory for results.

    Returns
    -------
    ModelConfig
        Default configuration object.
    """
    return ModelConfig(
        # Model architecture - typical for concentration maps
        in_channels=10,  # Adjust based on your molecule count
        encoder_channels=[32, 64, 128],
        kernel_size=3,
        stride=2,
        padding=1,
        activation="ReLU",
        # Training parameters
        batch_size=8,  # Start with smaller batch size for memory
        learning_rate=1e-3,
        num_epochs=100,
        optimizer="Adam",
        scheduler="StepLR",
        scheduler_params={"step_size": 30, "gamma": 0.1},
        # Data paths
        reduced_wl_dir=reduced_wl_dir,
        gt_dir=gt_dir,
        split_ratios=(0.8, 0.1, 0.1),
        # Output
        experiment_name=experiment_name,
        output_dir=output_dir,
        # Early stopping and checkpointing
        early_stopping_patience=15,
        save_best_model=True,
        checkpoint_interval=10,
    )
