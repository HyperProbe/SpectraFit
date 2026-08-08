"""
Model pipeline for training and validating Conv autoencoder on concentration maps.

This module provides a complete pipeline for training and validating convolutional
autoencoders using the ConcentrationsDataset. It includes configuration management,
training loops, validation, checkpointing, and logging functionality.
"""

import gc
import json
from random import random, sample
import numpy as np
import shutil
import time
import statistics
from loguru import logger
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict
import traceback
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.data_sample.concentration_sample import ConcentrationSample
from src.dataset.data_sample.fused_helicoid_sample import FusedHelicoidSample
from src.dataset.dataset_utils import collate_dataclass
from src.models.pipeline.checkpoint_manager import CheckpointManager
from src.models.pipeline.evaluation_manager import EvaluationManager
from src.models.pipeline.training_manager import TrainingManager
import wandb
from src.models.pipeline.config import ModelConfig
from src.models.pipeline.model_factory import ModelFactory
from src.models.pipeline.data_manager import DataManager
import os
import random


class ModelPipeline:
    """
    Complete pipeline for training and validating Conv autoencoder.

    This class handles:
    - Model initialization and configuration
    - Dataset loading and splitting
    - Training loop with validation
    - Checkpointing and model saving
    - Logging and monitoring
    - Evaluation metrics
    """

    def __init__(self, config: Union[ModelConfig, Dict[str, Any], str]):
        """
        Initialize the model pipeline.

        Parameters
        ----------
        config : ModelConfig, dict, or str
            Configuration object, dictionary, or path to JSON config file.
        """
        if isinstance(config, str):
            self.config = ModelConfig.from_json(config)
        elif isinstance(config, dict):
            self.config = ModelConfig.from_dict(config)
        elif isinstance(config, ModelConfig):
            self.config = config
        else:
            logger.warning(
                "Config should be a dictionary or path to JSON file. "
                "Using default configuration."
            )
            self.config = ModelConfig.from_dict(config)

        # Setup output directories
        self._setup_directories()

        self.model_factory = ModelFactory(self.config)
        self.data_manager = DataManager(self.config)
        # Initialize components (will be set during setup)
        self.evaluation_manager = None
        self.training_manager = None
        self.checkpoint_manager = None

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.with_wandb = self.config.with_wandb
        self.set_seed(self.config.random_seed, deterministic=self.config.deterministic)

        # Training state
        self.current_epoch = 1
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        logger.info(
            f"Initialized ModelPipeline with config: {self.config.experiment_name}"
        )
        logger.info(f"Using device: {self.config.device}")

    def set_seed(self, seed=42, deterministic=True):
        os.environ["PYTHONHASHSEED"] = str(seed)
        # For cuBLAS determinism on CUDA (Linux):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
        # (Optional but recommended) disable TF32 which can change numerics:
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(
                True
            )  # throws if a non-deterministic op is hit
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def _setup_directories(self):
        """Create necessary output directories."""
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Save config
        self.config.save_json(self.output_dir / "config.json")

    def setup_all(self):
        logger.info("Setting up all components...")

        self.model = self.model_factory.create_model()
        self.checkpoint_manager = CheckpointManager(self.config, self.output_dir)
        self.train_loader, self.val_loader, self.test_loader = (
            self.data_manager.setup_data()
        )
        self.training_manager = TrainingManager(
            self.config, self.model, self.model_factory.get_ddpm_scheduler()
        )
        self.training_manager.setup_training()
        self.evaluation_manager = EvaluationManager(
            self.config, self.model, self.training_manager.criterion
        )

        logger.info("Pipeline setup complete")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns
        -------
        float
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.training_manager.train_step(batch)

            total_loss += loss.item()
            num_batches += 1

            # Log batch progress
            if batch_idx % self.config.batch_print == 0:
                logger.debug(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self, best_val_loss) -> float:
        """
        Validate for one epoch.

        Returns
        -------
        float
            Average validation loss for the epoch.
        """
        return self.evaluation_manager.validate_epoch(
            self.val_loader, self.current_epoch, best_val_loss
        )

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop.

        Parameters
        ----------
        resume_from_checkpoint : str, optional
            Path to checkpoint file to resume training from.
        """
        # Setup all components
        self.setup_all()

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.current_epoch = self.checkpoint_manager.load_checkpoint(
                resume_from_checkpoint,
                self.model,
                self.training_manager.optimizer,
                self.training_manager.scheduler,
            )

        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        # Early stopping counter
        patience_counter = 0

        for epoch in range(self.current_epoch, self.config.num_epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()

            # Train epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate epoch
            val_loss = self.validate_epoch(self.best_val_loss)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - start_time

            # Log epoch results
            logger.info(
                f"Epoch {epoch}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # Tensorboard logging
            if self.with_wandb:
                wandb.log({"epoch:": epoch, "train/loss": train_loss}, step=epoch)
                wandb.log({"epoch:": epoch, "val/loss": val_loss}, step=epoch)

            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"New best validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                epoch,
                self.model,
                self.training_manager.optimizer,
                self.training_manager.scheduler,
                is_best,
            )

            # Update learning rate
            if self.training_manager.scheduler:
                if isinstance(
                    self.training_manager.scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.training_manager.scheduler.step(val_loss)
                else:
                    self.training_manager.scheduler.step()

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement"
                )
                break

        logger.info(
            f"Training completed. Best validation loss: {self.best_val_loss:.6f}"
        )

    def evaluate(
        self,
        data_loader: Optional[DataLoader] = None,
        with_wandb=True,
        evaluate_on_full_res=False,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Parameters
        ----------
        data_loader : DataLoader, optional
            Data loader to evaluate on. If None, uses test_loader.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """
        if data_loader is None:
            data_loader = self.test_loader
        if evaluate_on_full_res:
            logger.info("Evaluating on full resolution data...")
            # replace test dataset with full res version
            data_loader = self.data_manager.get_full_res_test_loader()

        return self.evaluation_manager.evaluate(data_loader, with_wandb)

    def evaluate_individual_samples(
        self,
        data_loader: Optional[DataLoader] = None,
        evaluate_on_full_res: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on individual test samples to get per-sample metrics.

        Parameters
        ----------
        data_loader : DataLoader, optional
            Data loader to evaluate on. If None, uses test_loader.
        evaluate_on_full_res : bool, optional
            Whether to evaluate on full resolution data. Default is False.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping sample IDs to their individual test metrics.
        """
        if data_loader is None:
            data_loader = self.test_loader
        if evaluate_on_full_res:
            logger.info("Evaluating individual samples on full resolution data...")
            data_loader = self.data_manager.get_full_res_test_loader()

        return self.evaluation_manager.evaluate_individual_samples(data_loader)


    def as_batch(self, sample):
        return collate_dataclass([sample])

    def predict_sample(self, sample: ConcentrationSample | FusedHelicoidSample):
        """
        Predict on a single sample.

        Parameters
        ----------
        sample : ConcentrationSample or FusedHelicoidSample
            Single data sample to predict on.

        Returns
        -------
        torch.Tensor
            Predicted output tensor.
        """
        self.model.eval()
        with torch.no_grad():
            batch = self.as_batch(sample)
            output = self.evaluation_manager.predict(batch)
        return output

    def predict(
        self, input_data: torch.Tensor, hsi_data: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Make predictions with the trained model.

        Parameters
        ----------
        input_data : torch.Tensor
            Input tensor of shape (B, H, W, C).
        hsi_data: optional hsi data for DualEncoder

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor of same shape.
        """
        self.model.eval()
        with torch.no_grad():
            input_data = input_data.to(self.config.device)
            if hsi_data is not None:
                hsi_data = hsi_data.to(self.config.device)
                output = self.model(input_data, hsi_data)
            else:
                output = self.model(input_data)
        return output

    def benchmark_inference(
        self, sample_id: str, n_warmup=10, n_runs: int = 10, use_cuda_events=True
    ):
        self.model.eval()
        sample = self.data_manager.dataset.get_sample_by_id(sample_id)
        sample_tensor = self.as_batch(sample)

        times = []
        with torch.inference_mode():
            # Warm-up runs
            for _ in range(n_warmup):
                _ = self.evaluation_manager.predict(sample_tensor)
            torch.cuda.synchronize()

            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                for _ in range(n_runs):
                    start.record()
                    _ = self.evaluation_manager.predict(sample_tensor)
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end))  # ms
            else:
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    _ = self.evaluation_manager.predict(sample_tensor)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000)  # ms
        return {
            "avg_ms": sum(times) / len(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "p50_ms": statistics.median(times),
            "p90_ms": sorted(times)[int(0.9 * len(times)) - 1],
            "runs": len(times),
        }

    def start_pipeline(self):
        """
        Start the complete training and evaluation pipeline.
        This method initializes all components and runs the training and evaluation.
        """
        logger.info("Starting model pipeline...")
        with wandb.init(
            project=self.config.project,
            config=asdict(self.config),
            name=self.config.experiment_name,
        ):
            self.train_and_evaluate()
        logger.info("Model pipeline completed.")

    def train_and_evaluate(self):
        self.train()
        self.load_best_model()
        self.evaluate(self.test_loader)

    def load_best_model(self):
        """Load the best saved model."""
        best_model_path = self.checkpoint_manager.get_best_model_path()
        if best_model_path.exists():
            self.checkpoint_manager.load_checkpoint(
                str(best_model_path),
                self.model,
                self.training_manager.optimizer,
                self.training_manager.scheduler,
            )
        else:
            logger.warning(
                "Best model checkpoint not found. Using current model state."
            )


def train_sweep(config=None):
    """
    Train a model using a sweep configuration.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary for the sweep.
    """
    with wandb.init(config=config) as run:
        config = wandb.config
        api = wandb.Api()
        sweep = api.sweep(run.sweep_id)
        sweep_name = sweep.name
        project = sweep.project
        sweep_dir = Path("../results/models/sweep") / project / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)

        config.output_dir = sweep_dir
        config.experiment_name = run.name

        metrics_file = sweep_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        try:
            pipeline = ModelPipeline(config)
            pipeline.train_and_evaluate()

            current_mse = pipeline.evaluation_manager.metrics["test/mse"]
            logger.info(f"Current MSE: {current_mse:.6f}")

            run_dir = sweep_dir / run.name
            # compare to best so far
            if all_metrics:
                # find best run (smallest MSE)
                best_run, best_mse = min(all_metrics.items(), key=lambda kv: kv[1])
                if current_mse > best_mse:
                    # worse than best: drop this run's folder
                    if run_dir.exists():
                        logger.info(
                            f"Run {run.name} has worse MSE ({current_mse:.6f}) than best run {best_run} ({best_mse:.6f}). Removing run directory."
                        )
                        shutil.rmtree(run_dir)
                else:
                    # better: drop the old best's folder
                    old_dir = sweep_dir / best_run
                    if old_dir.exists():
                        logger.info(
                            f"Run {run.name} has better MSE ({current_mse:.6f}) than best run {best_run} ({best_mse:.6f}). Removing old best run directory."
                        )
                        shutil.rmtree(old_dir)
            # else: first run, nothing to compare

            all_metrics[run.name] = current_mse
            with open(metrics_file, "w") as f:
                json.dump(all_metrics, f, indent=2)
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Error during sweep run {run.name}: {e}")
            # If there was an error, remove the run directory if it was created
            stack = traceback.format_exc()
            logger.error(stack)

            run_dir = sweep_dir / run.name
            if run_dir.exists():
                shutil.rmtree(run_dir)

            torch.cuda.empty_cache()
            gc.collect()


def load_model_pipeline(path: str, device: str) -> ModelPipeline:
    """
    Load a ModelPipeline from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.

    Returns
    -------
    ModelPipeline
        Loaded ModelPipeline instance.
    """
    base_path = Path(path)
    config_path = base_path / "config.json"
    checkpoint_path = base_path / "checkpoints" / "best_model.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config = ModelConfig.from_json(config_path)

    config.device = device
    pipeline = ModelPipeline(config)
    config.device_ids = [torch.device(device)]
    pipeline.setup_all()
    pipeline.checkpoint_manager.load_checkpoint(
        str(checkpoint_path),
        pipeline.model,
        pipeline.training_manager.optimizer,
        pipeline.training_manager.scheduler,
    )
    return pipeline
