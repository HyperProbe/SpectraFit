"""
Evaluation manager for the model pipeline.

This module handles model evaluation, metrics calculation, and visualization.
"""

import io
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from typing import Dict, Literal, Optional

import wandb
from monai.inferers import sliding_window_inference
from src.dataset.concentrations_dataset import Signal
from src.molecules import MoleculeIndex
from .config import ModelConfig


class EvaluationManager:
    """Manages model evaluation and metrics calculation."""

    def __init__(self, config: ModelConfig, model: nn.Module, criterion: nn.Module):
        """
        Initialize the evaluation manager.

        Parameters
        ----------
        config : ModelConfig
            Configuration object.
        model : nn.Module
            The model to evaluate.
        criterion : nn.Module
            Loss criterion for evaluation.
        """
        self.config = config
        self.model = model
        self.criterion = criterion
        self.metrics = None

    def _get_data(self, batch, data_key: Literal["gt", "reduced_wl"]):
        if self.config.autoencoder_type == "DualEncoder":
            return (
                batch["concentration_data"][data_key]["coef_list"]
                .to(self.config.device)
                .float()
            )
        elif self.config.ssr:
            if data_key == "reduced_wl":
                data_key_ssr = "hsi_cube"
            else:
                data_key_ssr = "hsi_cube_original"
            return batch[data_key_ssr].to(self.config.device).float()
        return batch[data_key]["coef_list"].to(self.config.device).float()


    def evaluate(
        self,
        data_loader: DataLoader,
        with_wandb: bool = True,
    ):
        """
        Evaluate model on the provided data loader.
        Parameters
        ----------
        data_loader : DataLoader
            Data loader to evaluate on.
        with_wandb : bool, optional
            Whether to log metrics to wandb. Default is True.

        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics.
        """
        if self.config.ssr:
            return self.evaluate_ssr(data_loader, with_wandb)
        else:
            return self.evaluate_concentration_map_recon(data_loader, with_wandb)

    def evaluate_concentration_map_recon(
        self,
        data_loader: DataLoader,
        with_wandb: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on the provided data loader.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader to evaluate on.
        with_wandb : bool, optional
            Whether to log metrics to wandb. Default is True.

        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        if self.config.evaluate_full_resolution:
            logger.info("Evaluating on full resolution")

        # Initialize metric accumulators (sums for means)
        metrics_accumulator = self._initialize_metrics_accumulator()
        # Lists for per-sample metrics (for std computation)
        mse_values, mae_values, ssim_values, psnr_values = [], [], [], []
        mse_input_values, mae_input_values, ssim_input_values, psnr_input_values = (
            [],
            [],
            [],
            [],
        )

        # Initialize per-channel MSE accumulators
        total_mse_per_channel = torch.zeros(
            self.config.in_channels, device=self.config.device
        )
        total_mse_input_per_channel = torch.zeros(
            self.config.in_channels, device=self.config.device
        )

        # Use reduction='none' to obtain per-image metrics for std calculation
        ssim_criterion = StructuralSimilarityIndexMeasure(reduction="none").to(
            self.config.device
        )
        # For PSNR, use default reduction and compute batch mean, then collect per-batch values
        psnr_criterion = PeakSignalNoiseRatio().to(self.config.device)

        num_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                gt_coef = self._get_data(batch, "gt")

                input_data = self._get_data(batch, "reduced_wl")
                target_data = gt_coef

                output = self.predict(batch)

                output_BCHW = self.permute_to_BCHW(output)
                target_BCHW = self.permute_to_BCHW(target_data)
                input_BCHW = self.permute_to_BCHW(input_data)

                # Per-sample metrics (shape: B) - all using B,C,H,W format
                mse_samples = torch.mean(
                    (output_BCHW - target_BCHW) ** 2, dim=(1, 2, 3)
                )  # (B,)
                mae_samples = torch.mean(
                    (output_BCHW - target_BCHW).abs(), dim=(1, 2, 3)
                )  # (B,)
                ssim_samples = ssim_criterion(output_BCHW, target_BCHW)  # (B,)

                # For PSNR, compute batch average but replicate for each sample
                # This maintains the original calculation method
                psnr_batch = psnr_criterion(output_BCHW, target_BCHW)  # scalar
                psnr_samples = torch.full(
                    (output_BCHW.shape[0],),
                    psnr_batch.item(),
                    device=self.config.device,
                )

                # Baseline per-sample metrics - all using B,C,H,W format
                mse_input_samples = torch.mean(
                    (input_BCHW - target_BCHW) ** 2, dim=(1, 2, 3)
                )
                mae_input_samples = torch.mean(
                    (input_BCHW - target_BCHW).abs(), dim=(1, 2, 3)
                )
                ssim_input_samples = ssim_criterion(input_BCHW, target_BCHW)

                # For baseline PSNR, same approach
                psnr_input_batch = psnr_criterion(input_BCHW, target_BCHW)  # scalar
                psnr_input_samples = torch.full(
                    (input_BCHW.shape[0],),
                    psnr_input_batch.item(),
                    device=self.config.device,
                )  # Accumulate per-sample lists (on CPU for memory efficiency)
                # Flatten and convert to list to handle any tensor shape issues
                mse_values.extend(mse_samples.detach().cpu().numpy().flatten().tolist())
                mae_values.extend(mae_samples.detach().cpu().numpy().flatten().tolist())
                ssim_values.extend(
                    ssim_samples.detach().cpu().numpy().flatten().tolist()
                )
                psnr_values.extend(
                    psnr_samples.detach().cpu().numpy().flatten().tolist()
                )
                mse_input_values.extend(
                    mse_input_samples.detach().cpu().numpy().flatten().tolist()
                )
                mae_input_values.extend(
                    mae_input_samples.detach().cpu().numpy().flatten().tolist()
                )
                ssim_input_values.extend(
                    ssim_input_samples.detach().cpu().numpy().flatten().tolist()
                )
                psnr_input_values.extend(
                    psnr_input_samples.detach().cpu().numpy().flatten().tolist()
                )

                # Batch mean metrics for averaging - all using B,C,H,W format
                # print(output_BCHW.shape, target_BCHW.shape)
                batch_metrics = {
                    "loss": self.criterion(output, target_data).item(),
                    "mse": float(mse_samples.mean().item()),
                    "mae": float(mae_samples.mean().item()),
                    "ssim": float(ssim_samples.mean().item()),
                    "psnr": float(psnr_samples.mean().item()),
                }
                batch_metrics_input = {
                    "loss": self.criterion(input_data, target_data).item(),
                    "mse": float(mse_input_samples.mean().item()),
                    "mae": float(mae_input_samples.mean().item()),
                    "ssim": float(ssim_input_samples.mean().item()),
                    "psnr": float(psnr_input_samples.mean().item()),
                }

                # Calculate per-channel MSE - using B,C,H,W format
                mse_per_channel = torch.mean(
                    (output_BCHW - target_BCHW) ** 2, dim=(0, 2, 3)
                )
                mse_input_per_channel = torch.mean(
                    (input_BCHW - target_BCHW) ** 2, dim=(0, 2, 3)
                )

                # Accumulate metrics
                batch_size = gt_coef.size(0)
                self._accumulate_metrics(
                    metrics_accumulator, batch_metrics, batch_metrics_input, batch_size
                )

                # Accumulate per-channel MSE
                total_mse_per_channel += mse_per_channel * batch_size
                total_mse_input_per_channel += mse_input_per_channel * batch_size
                num_samples += batch_size

        # Calculate final metrics
        final_metrics = self._finalize_metrics(
            metrics_accumulator,
            total_mse_per_channel,
            total_mse_input_per_channel,
            num_samples,
            # numpy arrays for std
            np.array(mse_values),
            np.array(mae_values),
            np.array(ssim_values),
            np.array(psnr_values),
            np.array(mse_input_values),
            np.array(mae_input_values),
            np.array(ssim_input_values),
            np.array(psnr_input_values),
        )

        if with_wandb:
            wandb.log(final_metrics)

        logger.info(f"Evaluation metrics: {final_metrics}")
        self.metrics = final_metrics
        return final_metrics

    def _initialize_metrics_accumulator(self) -> Dict[str, float]:
        """Initialize the metrics accumulator dictionary."""
        return {
            "total_loss": 0.0,
            "total_mse": 0.0,
            "total_mae": 0.0,
            "total_ssim": 0.0,
            "total_psnr": 0.0,
            "total_loss_input_data": 0.0,
            "total_mse_input_data": 0.0,
            "total_mae_input_data": 0.0,
            "total_ssim_input_data": 0.0,
            "total_psnr_input_data": 0.0,
        }

    def _accumulate_metrics(
        self, accumulator, batch_metrics, batch_metrics_input, batch_size
    ):
        """Accumulate batch metrics into the total accumulator."""
        accumulator["total_loss"] += batch_metrics["loss"] * batch_size
        accumulator["total_mse"] += batch_metrics["mse"] * batch_size
        accumulator["total_mae"] += batch_metrics["mae"] * batch_size
        accumulator["total_ssim"] += batch_metrics["ssim"] * batch_size
        accumulator["total_psnr"] += batch_metrics["psnr"] * batch_size

        accumulator["total_loss_input_data"] += batch_metrics_input["loss"] * batch_size
        accumulator["total_mse_input_data"] += batch_metrics_input["mse"] * batch_size
        accumulator["total_mae_input_data"] += batch_metrics_input["mae"] * batch_size
        accumulator["total_ssim_input_data"] += batch_metrics_input["ssim"] * batch_size
        accumulator["total_psnr_input_data"] += batch_metrics_input["psnr"] * batch_size

    def _finalize_metrics(
        self,
        accumulator,
        total_mse_per_channel,
        total_mse_input_per_channel,
        num_samples,
        mse_values: np.ndarray,
        mae_values: np.ndarray,
        ssim_values: np.ndarray,
        psnr_values: np.ndarray,
        mse_input_values: np.ndarray,
        mae_input_values: np.ndarray,
        ssim_input_values: np.ndarray,
        psnr_input_values: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate final averaged metrics."""
        # Calculate average per-channel MSE
        avg_mse_per_channel = total_mse_per_channel / num_samples
        avg_mse_input_per_channel = total_mse_input_per_channel / num_samples
        # Means (from sums)
        mean_mse = accumulator["total_mse"] / num_samples
        mean_mae = accumulator["total_mae"] / num_samples
        mean_ssim = accumulator["total_ssim"] / num_samples
        mean_psnr = accumulator["total_psnr"] / num_samples

        mean_mse_input = accumulator["total_mse_input_data"] / num_samples
        mean_mae_input = accumulator["total_mae_input_data"] / num_samples
        mean_ssim_input = accumulator["total_ssim_input_data"] / num_samples
        mean_psnr_input = accumulator["total_psnr_input_data"] / num_samples

        # Std (population, ddof=0). Use ddof=1 for sample std if preferred.
        std_mse = float(np.std(mse_values, ddof=0)) if mse_values.size else 0.0
        std_mae = float(np.std(mae_values, ddof=0)) if mae_values.size else 0.0
        std_ssim = float(np.std(ssim_values, ddof=0)) if ssim_values.size else 0.0
        std_psnr = float(np.std(psnr_values, ddof=0)) if psnr_values.size else 0.0

        std_mse_input = (
            float(np.std(mse_input_values, ddof=0)) if mse_input_values.size else 0.0
        )
        std_mae_input = (
            float(np.std(mae_input_values, ddof=0)) if mae_input_values.size else 0.0
        )
        std_ssim_input = (
            float(np.std(ssim_input_values, ddof=0)) if ssim_input_values.size else 0.0
        )
        std_psnr_input = (
            float(np.std(psnr_input_values, ddof=0)) if psnr_input_values.size else 0.0
        )

        metrics = {
            "test/loss": accumulator["total_loss"] / num_samples,
            "test/mse": mean_mse,
            "test/mae": mean_mae,
            "test/ssim": mean_ssim,
            "test/psnr": mean_psnr,
            # Standard deviations
            "test/std/mse": std_mse,
            "test/std/mae": std_mae,
            "test/std/ssim": std_ssim,
            "test/std/psnr": std_psnr,
            # Baseline means
            "test/input_baseline/loss_input_data": accumulator["total_loss_input_data"]
            / num_samples,
            "test/input_baseline/mse_input_data": mean_mse_input,
            "test/input_baseline/mae_input_data": mean_mae_input,
            "test/input_baseline/ssim_input_data": mean_ssim_input,
            "test/input_baseline/psnr_input_data": mean_psnr_input,
            # Baseline stds
            "test/input_baseline/std/mse_input_data": std_mse_input,
            "test/input_baseline/std/mae_input_data": std_mae_input,
            "test/input_baseline/std/ssim_input_data": std_ssim_input,
            "test/input_baseline/std/psnr_input_data": std_psnr_input,
        }

        if self.config.merge_channels_to_signals is None:
            # Add per-channel MSE metrics with molecule names
            metrics = self._get_mse_per_channel(
                metrics, avg_mse_per_channel, avg_mse_input_per_channel
            )
        else:
            metrics = self._get_mse_for_merged_signals(
                metrics, avg_mse_per_channel, avg_mse_input_per_channel
            )

        return metrics

    def _get_mse_per_channel(
        self, metrics, avg_mse_per_channel, avg_mse_input_per_channel
    ) -> torch.Tensor:
        molecule_names = self._get_molecule_names()
        for i, molecule_name in enumerate(molecule_names):
            if i < len(avg_mse_per_channel):  # Safety check
                metrics[f"test/mse_per_channel/{molecule_name}"] = avg_mse_per_channel[
                    i
                ].item()
                metrics[f"test/input_baseline/mse_per_channel/{molecule_name}"] = (
                    avg_mse_input_per_channel[i].item()
                )

        return metrics

    def _get_mse_for_merged_signals(
        self, metrics, avg_mse_per_channel, avg_mse_input_per_channel
    ) -> torch.Tensor:
        for i, signal in enumerate(self.config.merge_channels_to_signals):
            metrics[f"test/mse_per_signal/{signal}"] = avg_mse_per_channel[i].item()
            metrics[f"test/input_baseline/mse_per_signal/{signal}"] = (
                avg_mse_input_per_channel[i].item()
            )
        return metrics

    def _get_molecule_names(self) -> list:
        """Get molecule names for per-channel metrics."""
        if self.config.chosen_molecules is not None:
            return self.config.chosen_molecules
        else:
            return [name for name, _ in MoleculeIndex.__members__.items()]

    def validate_epoch(
        self, val_loader: DataLoader, current_epoch: int, best_val_loss: float
    ) -> float:
        """
        Validate for one epoch with optional visualization logging.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.
        current_epoch : int
            Current epoch number for logging.

        Returns
        -------
        float
            Average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # Get ground truth concentration maps
                target_data = self._get_data(batch, "gt")
                input_data = self._get_data(batch, "reduced_wl")


                # Forward pass
                output = self.predict(batch)
                loss = self.criterion(output, target_data)

                # Log first batch reconstruction for visualization
                if (
                    i == 0
                    and self.config.with_wandb
                    and current_epoch % self.config.log_image_epoch == 0
                ):
                    args_molecule_recon = {
                        "step": current_epoch,
                        "original": self.permute_to_BHWC(target_data)[0].cpu().numpy(),
                        "reconstructed": self.permute_to_BHWC(output)[0].cpu().numpy(),
                        "input_data": self.permute_to_BHWC(input_data)[0].cpu().numpy(),
                        "coarseness": self.config.log_coarseness,
                    }

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        is_best = avg_loss < best_val_loss
        if (
            self.config.with_wandb
            and current_epoch % self.config.log_image_epoch == 0
            and is_best
            and not self.config.ssr
        ):
            self.log_molecule_recon(**args_molecule_recon)
        return avg_loss

    def evaluate_ssr(
        self,
        data_loader: DataLoader,
        with_wandb: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate SSR baseline on the provided data loader.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader to evaluate on.

        Returns
        -------
        Dict[str, float]
            Dictionary containing SSR evaluation metrics.
        """
        self.model.eval()
        if self.config.evaluate_full_resolution:
            logger.info("Evaluating on full resolution")

        # Lists for per-sample metrics (for std computation)
        mse_values, mae_values, ssim_values, psnr_values = [], [], [], []
        ssim_criterion = StructuralSimilarityIndexMeasure(reduction="none").to(
            self.config.device
        )
        psnr_criterion = PeakSignalNoiseRatio().to(self.config.device)
        num_samples = 0
        metrics_accumulator = {
            "test/mse": 0.0,
            "test/mae": 0.0,
            "test/ssim": 0.0,
            "test/psnr": 0.0,
        }
        with torch.no_grad():
            for batch in data_loader:
                gt_cube = self._get_data(batch, "gt")
                output = self.predict(batch)

                mse_samples = torch.mean((output - gt_cube) ** 2, dim=(1, 2, 3))
                mae_samples = torch.mean((output - gt_cube).abs(), dim=(1, 2, 3))
                ssim_samples = ssim_criterion(output, gt_cube)
                psnr_batch = psnr_criterion(output, gt_cube)
                psnr_samples = torch.full(
                    (output.shape[0],),
                    psnr_batch.item(),
                    device=self.config.device,
                )

                mse_values.extend(mse_samples.detach().cpu().numpy().flatten().tolist())
                mae_values.extend(mae_samples.detach().cpu().numpy().flatten().tolist())
                ssim_values.extend(
                    ssim_samples.detach().cpu().numpy().flatten().tolist()
                )
                psnr_values.extend(
                    psnr_samples.detach().cpu().numpy().flatten().tolist()
                )
                batch_size = gt_cube.size(0)
                num_samples += batch_size

                self._accumulate_metrics_ssr(
                    metrics_accumulator,
                    {
                        "mse": np.mean(mse_values),
                        "mae": np.mean(mae_values),
                        "ssim": np.mean(ssim_values),
                        "psnr": np.mean(psnr_values),
                    },
                    batch_size,
                )
        # Finalize metrics
        final_metrics = self.finalize_metrics_ssr(
            metrics_accumulator,
            num_samples,
            np.array(mse_values),
            np.array(mae_values),
            np.array(ssim_values),
            np.array(psnr_values),
        )

        if with_wandb:
            wandb.log(final_metrics)
        logger.info(f"SSR Evaluation metrics: {final_metrics}")
        self.metrics = final_metrics
        return final_metrics

    def _accumulate_metrics_ssr(self, metrics_accumulator, batch_metrics, batch_size):
        metrics_accumulator["test/mse"] += batch_metrics["mse"] * batch_size
        metrics_accumulator["test/mae"] += batch_metrics["mae"] * batch_size
        metrics_accumulator["test/ssim"] += batch_metrics["ssim"] * batch_size
        metrics_accumulator["test/psnr"] += batch_metrics["psnr"] * batch_size

    def finalize_metrics_ssr(
        self,
        metrics_accumulator,
        num_samples,
        mse_values,
        mae_values,
        ssim_values,
        psnr_values,
    ) -> Dict[str, float]:
        return {
            "test/mse": metrics_accumulator["test/mse"] / num_samples,
            "test/mae": metrics_accumulator["test/mae"] / num_samples,
            "test/ssim": metrics_accumulator["test/ssim"] / num_samples,
            "test/psnr": metrics_accumulator["test/psnr"] / num_samples,
            "test/std/mse": float(np.std(mse_values, ddof=0)),
            "test/std/mae": float(np.std(mae_values, ddof=0)),
            "test/std/ssim": float(np.std(ssim_values, ddof=0)),
            "test/std/psnr": float(np.std(psnr_values, ddof=0)),
        }

    def evaluate_individual_samples(
        self,
        data_loader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on individual test samples to get per-sample metrics.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader to evaluate on (typically test loader).

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping sample IDs to their individual test metrics.
            Each sample's metrics include: mse, mae, ssim, psnr, and baseline metrics.
        """
        self.model.eval()
        logger.info("Evaluating individual test samples")

        # Dictionary to store per-sample results
        sample_results = {}

        # Initialize metric criteria
        ssim_criterion = StructuralSimilarityIndexMeasure(reduction="none").to(
            self.config.device
        )
        psnr_criterion = PeakSignalNoiseRatio().to(self.config.device)

        with torch.no_grad():
            for batch in data_loader:
                # Get sample information
                sample_ids = batch["id"]
                batch_size = len(sample_ids)

                # Get data
                gt_coef = self._get_data(batch, "gt")
                input_data = self._get_data(batch, "reduced_wl")
                target_data = gt_coef

                # Get model predictions
                output = self.predict(batch)

                # Convert to BCHW format for consistent metric computation
                output_BCHW = self.permute_to_BCHW(output)
                target_BCHW = self.permute_to_BCHW(target_data)
                input_BCHW = self.permute_to_BCHW(input_data)

                # Compute per-sample metrics
                mse_samples = (
                    torch.mean((output_BCHW - target_BCHW) ** 2, dim=(1, 2, 3))
                    .cpu()
                    .numpy()
                )  # (B,)

                mae_samples = (
                    torch.mean((output_BCHW - target_BCHW).abs(), dim=(1, 2, 3))
                    .cpu()
                    .numpy()
                )  # (B,)

                # Compute SSIM individually for each sample to avoid dimension issues
                ssim_samples = []
                for i in range(batch_size):
                    ssim_val = (
                        ssim_criterion(output_BCHW[i : i + 1], target_BCHW[i : i + 1])
                        .cpu()
                        .item()
                    )
                    ssim_samples.append(ssim_val)
                ssim_samples = np.array(ssim_samples)

                # For PSNR, compute individually for each sample
                psnr_samples = []
                for i in range(batch_size):
                    psnr_val = (
                        psnr_criterion(output_BCHW[i : i + 1], target_BCHW[i : i + 1])
                        .cpu()
                        .item()
                    )
                    psnr_samples.append(psnr_val)
                psnr_samples = np.array(psnr_samples)

                # Compute baseline metrics (input vs target)
                mse_baseline_samples = (
                    torch.mean((input_BCHW - target_BCHW) ** 2, dim=(1, 2, 3))
                    .cpu()
                    .numpy()
                )  # (B,)

                mae_baseline_samples = (
                    torch.mean((input_BCHW - target_BCHW).abs(), dim=(1, 2, 3))
                    .cpu()
                    .numpy()
                )  # (B,)

                # Compute baseline SSIM individually for each sample to avoid dimension issues
                ssim_baseline_samples = []
                for i in range(batch_size):
                    ssim_val = (
                        ssim_criterion(input_BCHW[i : i + 1], target_BCHW[i : i + 1])
                        .cpu()
                        .item()
                    )
                    ssim_baseline_samples.append(ssim_val)
                ssim_baseline_samples = np.array(ssim_baseline_samples)

                # For baseline PSNR, compute individually for each sample
                psnr_baseline_samples = []
                for i in range(batch_size):
                    psnr_val = (
                        psnr_criterion(input_BCHW[i : i + 1], target_BCHW[i : i + 1])
                        .cpu()
                        .item()
                    )
                    psnr_baseline_samples.append(psnr_val)
                psnr_baseline_samples = np.array(psnr_baseline_samples)

                # Store results for each sample in the batch
                for i, sample_id in enumerate(sample_ids):
                    # Create sample metrics dictionary
                    sample_metrics = {
                        # Model performance metrics
                        "mse": float(mse_samples[i]),
                        "mae": float(mae_samples[i]),
                        "ssim": float(ssim_samples[i]),
                        "psnr": float(psnr_samples[i]),
                        # Baseline performance metrics
                        "mse_baseline": float(mse_baseline_samples[i]),
                        "mae_baseline": float(mae_baseline_samples[i]),
                        "ssim_baseline": float(ssim_baseline_samples[i]),
                        "psnr_baseline": float(psnr_baseline_samples[i]),
                    }

                    # Store in results dictionary
                    sample_results[sample_id] = sample_metrics

        logger.info(f"Evaluated {len(sample_results)} individual samples")
        return sample_results

    def predict(self, batch):
        input_data_concentration_map = self._get_data(batch, "reduced_wl")

        if self.config.inferer_mode == "sliding_window":
            return self._predict_with_sliding_window(
                batch, input_data_concentration_map
            )
        else:
            return self._predict_whole(batch, input_data_concentration_map)

    def _predict_whole(self, batch, input_data_concentration_map):
        """Direct model prediction without sliding window."""
        if self.config.autoencoder_type == "DualEncoder":
            input_data_hsi = batch["hsi_cube"].to(self.config.device).float()
            output = self.model(input_data_concentration_map, input_data_hsi)
            return output

        return self.model(input_data_concentration_map)

    def _predict_with_sliding_window(self, batch, input_data_concentration_map):
        """Model prediction using sliding window inference."""
        # Store original format for output conversion
        original_tensor = input_data_concentration_map

        # Convert input to NCHW format for MONAI compatibility
        input_nchw = self.permute_to_BCHW(input_data_concentration_map)

        # Create predictor function that handles shape conversions
        if self.config.autoencoder_type == "DualEncoder":
            input_data_hsi = batch["hsi_cube"].to(self.config.device).float()
            input_hsi_nchw = self.permute_to_BCHW(input_data_hsi)

            def predictor(x_nchw):
                if self.config.channels_first:
                    # Model expects NCHW, use directly
                    output_model = self.model(x_nchw, input_hsi_nchw)
                    return output_model  # Already NCHW
                else:
                    # Model expects NHWC, convert back and forth
                    x_model_format = self.permute_to_BHWC(x_nchw)
                    hsi_model_format = self.permute_to_BHWC(input_hsi_nchw)
                    output_model = self.model(x_model_format, hsi_model_format)
                    return self.permute_to_BCHW(output_model)  # Convert back to NCHW

        else:

            def predictor(x_nchw):
                if self.config.channels_first:
                    # Model expects NCHW, use directly
                    output_model = self.model(x_nchw)
                    return output_model  # Already NCHW
                else:
                    # Model expects NHWC, convert back and forth
                    x_model_format = self.permute_to_BHWC(x_nchw)
                    output_model = self.model(x_model_format)
                    return self.permute_to_BCHW(output_model)  # Convert back to NCHW

        # Common sliding window inference arguments
        sw_kwargs = {
            "inputs": input_nchw,
            "roi_size": self.config.sliding_window_roi_size,
            "sw_batch_size": 1,
            "predictor": predictor,
            "overlap": self.config.sliding_window_overlap,
            "mode": "constant",
            "device": self.config.device,
        }

        # Perform sliding window inference in NCHW format
        output_nchw = sliding_window_inference(**sw_kwargs)

        # Convert output back to original format (determine from original tensor shape)
        if self._is_channels_first_format(original_tensor):
            return output_nchw  # Keep NCHW format
        else:
            return self.permute_to_BHWC(output_nchw)  # Convert to NHWC format

    def _is_channels_first_format(self, tensor: torch.Tensor) -> bool:
        """
        Determine if tensor is in channels-first (BCHW) format based on shape.

        Parameters
        ----------
        tensor : torch.Tensor
            4D tensor to check format for.

        Returns
        -------
        bool
            True if tensor is in BCHW format, False if in BHWC format.
        """
        if tensor.ndim != 4:
            raise ValueError("Input tensor must be 4-dimensional (BCHW or BHWC).")

        # Check if dimension 1 matches the expected number of channels (BCHW)
        # or if dimension 3 matches the expected number of channels (BHWC)
        return tensor.shape[1] == self.config.in_channels

    def permute_to_BHWC(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Permute a tensor from BCHW to BHWC format.

        Always converts from BCHW to BHWC regardless of current format.
        Used to convert tensors for models that expect NHWC format.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCHW format.

        Returns
        -------
        torch.Tensor
            Tensor in BHWC format.
        """
        if tensor.ndim != 4:
            raise ValueError("Input tensor must be 4-dimensional.")
        if self._is_channels_first_format(tensor):
            return tensor.permute(0, 2, 3, 1)  # BCHW -> BHWC
        return tensor

    def permute_to_BCHW(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Permute a tensor to BCHW format based on the current data format.

        Uses shape detection to determine if conversion is needed.
        Always ensures output is in BCHW format for MONAI compatibility.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor, either in BCHW or BHWC format.

        Returns
        -------
        torch.Tensor
            Tensor in BCHW format.
        """
        if tensor.ndim != 4:
            raise ValueError("Input tensor must be 4-dimensional.")

        if not self._is_channels_first_format(tensor):
            return tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return tensor  # Already BCHW

    def log_molecule_recon(
        self,
        step: int,
        original: np.ndarray,
        reconstructed: np.ndarray,
        input_data: np.ndarray,
        coarseness: int = 1,
    ):
        """
        Logs concentration maps for visualization.

        When merge_channels_to_signals is None, this method logs the default summaries:
        • HBT      = HbO2 + Hb
        • diffCCO  = COXA – CREDA
        • FAT      = FAT

        When merge_channels_to_signals is specified, this method only logs the signals
        specified in that list.

        When chosen_molecules is specified, this method adapts to show only available molecules
        and skips combinations that aren't possible.

        Columns are:
        1) Ground truth (full wavelengths)
        2) Reconstructed
        3) Input (reduced wavelengths)
        4) Absolute error |recon – truth|

        Args:
            step (int): logging step or epoch
            original (H, W, C): ground-truth map (filtered if chosen_molecules is set)
            reconstructed (H, W, C): model output (filtered if chosen_molecules is set)
            input_data (H, W, C): reduced-wavelength–set map (filtered if chosen_molecules is set)
            coarseness (int): subsampling factor
        """

        # Prepare maps
        def sub(m):
            return m[::coarseness, ::coarseness]

        # Create mapping from molecule names to indices in the filtered data
        if self.config.chosen_molecules is not None:
            molecule_to_filtered_idx = {
                mol_name: i for i, mol_name in enumerate(self.config.chosen_molecules)
            }
        else:
            molecule_to_filtered_idx = {
                name: MoleculeIndex[name].value
                for name, _ in MoleculeIndex.__members__.items()
            }

        summaries = []

        # If merge_channels_to_signals is specified, only log those signals
        if self.config.merge_channels_to_signals is not None:
            summaries = self._get_signal_summaries(
                self.config.merge_channels_to_signals,
                original,
                reconstructed,
                input_data,
                molecule_to_filtered_idx,
                sub,
            )
        else:
            summaries = self._get_default_summaries(
                original, reconstructed, input_data, molecule_to_filtered_idx, sub
            )

        if not summaries:
            logger.warning("No molecules available for logging reconstruction")
            return

        self._create_and_log_figure(summaries, step)

    def _get_signal_summaries(
        self,
        signals_to_log,
        original: np.ndarray,
        reconstructed: np.ndarray,
        input_data: np.ndarray,
        molecule_to_filtered_idx: dict,
        sub,
    ):
        """Get summaries for specific signals defined in merge_channels_to_signals."""
        summaries = []

        for i, signal in enumerate(signals_to_log):
            if signal == Signal.HbT:
                orig_hbt = sub(original[..., i])
                recon_hbt = sub(reconstructed[..., i])
                input_hbt = sub(input_data[..., i])
                summaries.append(("HBT", orig_hbt, recon_hbt, input_hbt, "Reds"))

            elif signal == Signal.diffCCO:
                orig_cco = sub(original[..., i])
                recon_cco = sub(reconstructed[..., i])
                input_cco = sub(input_data[..., i])
                summaries.append(("diffCCO", orig_cco, recon_cco, input_cco, "terrain"))

            elif signal == "b":
                # For signal "b", we might need to add specific logic here
                orig_b = sub(original[..., i])
                recon_b = sub(reconstructed[..., i])
                input_b = sub(input_data[..., i])
                summaries.append(("b", orig_b, recon_b, input_b, "seismic_r"))

        return summaries

    def _get_default_summaries(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        input_data: np.ndarray,
        molecule_to_filtered_idx: dict,
        sub,
    ):
        """Get the default summaries (original behavior)."""
        summaries = []

        # HBT = HBO2 + HB (only if both molecules are available)
        if "HBO2" in molecule_to_filtered_idx and "HB" in molecule_to_filtered_idx:
            hbo2_idx = molecule_to_filtered_idx["HBO2"]
            hb_idx = molecule_to_filtered_idx["HB"]

            orig_hbt = sub(original[..., hbo2_idx] + original[..., hb_idx])
            recon_hbt = sub(reconstructed[..., hbo2_idx] + reconstructed[..., hb_idx])
            input_hbt = sub(input_data[..., hbo2_idx] + input_data[..., hb_idx])
            summaries.append(("HBT", orig_hbt, recon_hbt, input_hbt, "Reds"))

        # diffCCO = COXA - CREDA (only if both molecules are available)
        if "COXA" in molecule_to_filtered_idx and "CREDA" in molecule_to_filtered_idx:
            coxa_idx = molecule_to_filtered_idx["COXA"]
            creda_idx = molecule_to_filtered_idx["CREDA"]

            orig_cco = sub(original[..., coxa_idx] - original[..., creda_idx])
            recon_cco = sub(
                reconstructed[..., coxa_idx] - reconstructed[..., creda_idx]
            )
            input_cco = sub(input_data[..., coxa_idx] - input_data[..., creda_idx])
            summaries.append(("diffCCO", orig_cco, recon_cco, input_cco, "terrain"))

        # FAT (if available)
        if "FAT" in molecule_to_filtered_idx:
            fat_idx = molecule_to_filtered_idx["FAT"]

            orig_fat = sub(original[..., fat_idx])
            recon_fat = sub(reconstructed[..., fat_idx])
            input_fat = sub(input_data[..., fat_idx])
            summaries.append(("FAT", orig_fat, recon_fat, input_fat, "viridis"))

        # If no standard summaries are available, show first 3 individual molecules
        if not summaries and self.config.chosen_molecules is not None:
            for i, mol_name in enumerate(self.config.chosen_molecules[:3]):
                orig_mol = sub(original[..., i])
                recon_mol = sub(reconstructed[..., i])
                input_mol = sub(input_data[..., i])
                summaries.append((mol_name, orig_mol, recon_mol, input_mol, "viridis"))

        return summaries

    def _create_and_log_figure(self, summaries, step: int):
        """Create and log the visualization figure."""
        fig, axes = plt.subplots(len(summaries), 4, figsize=(20, 5 * len(summaries)))
        if len(summaries) == 1:
            axes = axes.reshape(1, -1)

        for row, (name, o, r, inp, cmap) in enumerate(summaries):
            err = np.abs(r - o)
            maps = [o, r, inp, err]
            titles = ["Truth", "Reconstructed", "Input (reduced)", "Absolute Error"]
            cmaps = [cmap, cmap, cmap, "bwr"]

            for col, (m, title, cm) in enumerate(zip(maps, titles, cmaps)):
                ax = axes[row, col]
                im = ax.imshow(m, cmap=cm)
                ax.set_title(f"{name} – {title}")
                plt.colorbar(im, ax=ax, orientation="vertical")

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=72,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        im_q = im.quantize(colors=256, method=Image.MEDIANCUT)
        wandb.log({"molecule_reconstruction": wandb.Image(im_q)}, step=step)
        plt.close(fig)

    def reevaluate_and_patch_run(
        self,
        run_id: str,
        test_loader: DataLoader,
        entity: str = "idp2024",
    ):
        """
        Re-evaluate a single W&B run and add missing metrics.

        - Resumes the original run by ID.
        - Writes only the missing metrics into the run's summary by default.
        - Optionally logs the metrics as a time-series point too (log_history=True).

        Returns the dict of metrics actually written.
        """
        api = wandb.Api()
        project = self.config.project
        run = api.run(f"{entity}/{project}/{run_id}")

        # Compute metrics without logging (we decide what to write)
        computed = self.evaluate(test_loader, with_wandb=False) or {}

        existing = set((run.summary or {}).keys())
        to_write = {k: v for k, v in computed.items() if k not in existing}

        if not to_write:
            print(f"[{run_id}] Nothing to write (no metrics or all already present).")
            return {}

        # Re-open the exact same run and write
        with wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume="allow",
            settings=wandb.Settings(code_dir=".", _disable_stats=True),
        ) as resumed:
            # Always update summary (final/aggregate view)
            resumed.summary.update(to_write)

        print(f"[{run_id}] Wrote: {sorted(to_write.keys())}")
        return to_write
