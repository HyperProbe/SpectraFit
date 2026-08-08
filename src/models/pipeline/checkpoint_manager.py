"""
Checkpoint manager for the model pipeline.

This module handles model saving, loading, and checkpoint management.
"""

import torch
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any

from .config import ModelConfig


class CheckpointManager:
    """Manages model checkpointing and loading."""
    
    def __init__(self, config: ModelConfig, output_dir: Path):
        """
        Initialize the checkpoint manager.
        
        Parameters
        ----------
        config : ModelConfig
            Configuration object.
        output_dir : Path
            Output directory for saving checkpoints.
        """
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training state
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
    
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_best: bool = False,
    ):
        """
        Save model checkpoint.
        
        Parameters
        ----------
        epoch : int
            Current epoch number.
        model : torch.nn.Module
            Model to save.
        optimizer : torch.optim.Optimizer
            Optimizer to save.
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler to save.
        is_best : bool, optional
            Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler else None
            ),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }
        
        # Save regular checkpoint
        if (
            self.config.save_checkpoints
            and epoch % self.config.checkpoint_interval == 0
        ):
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best and self.config.save_best_model:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> int:
        """
        Load model from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file.
        model : torch.nn.Module
            Model to load state into.
        optimizer : torch.optim.Optimizer
            Optimizer to load state into.
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Scheduler to load state into.
        
        Returns
        -------
        int
            The epoch number from the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        current_epoch = checkpoint["epoch"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        logger.info(f"Loaded checkpoint from epoch {current_epoch}")
        return current_epoch
    
    def get_best_model_path(self) -> Path:
        """Get the path to the best saved model."""
        return self.checkpoint_dir / "best_model.pth"
    
    def update_training_history(self, train_loss: float, val_loss: float, is_best: bool):
        """
        Update training history and best validation loss.
        
        Parameters
        ----------
        train_loss : float
            Training loss for current epoch.
        val_loss : float
            Validation loss for current epoch.
        is_best : bool
            Whether current validation loss is the best so far.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if is_best:
            self.best_val_loss = val_loss
