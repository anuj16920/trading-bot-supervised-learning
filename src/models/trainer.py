"""Unified deep learning trainer for AQRF.

Handles TCN, Transformer, and Regime training with AMP, 8-bit Adam,
and MLflow logging. Optimized for RTX 3050 4GB VRAM.
"""
import gc
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import mlflow
import structlog

from src.utils.config import ModelConfig, TCNConfig, TransformerConfig, RegimeConfig
from src.utils.gpu import setup_cuda, log_vram, release_vram

logger = structlog.get_logger(__name__)


def get_optimizer(model: nn.Module, lr: float, weight_decay: float, use_8bit: bool = True):
    """Get optimizer with optional 8-bit quantization."""
    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        except ImportError:
            logger.warning("bitsandbytes not available, using standard AdamW")
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


class ForexDataset(Dataset):
    """Lazy-loading dataset from memory-mapped features."""

    def __init__(
        self,
        mmap_path: str,
        labels_path: Optional[str] = None,
        seq_len: int = 60,
    ):
        self.mmap = np.load(mmap_path, mmap_mode='r')
        self.seq_len = seq_len

        if labels_path:
            self.labels = np.load(labels_path, mmap_mode='r')
        else:
            # Generate dummy labels for inference
            self.labels = np.zeros((len(self.mmap), 2))

    def __len__(self) -> int:
        return len(self.mmap)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.mmap[idx].copy()).float()
        y = torch.tensor(self.labels[idx]).float()
        return x, y


class DLTrainer:
    """Unified trainer for all DL models."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        config: ModelConfig,
        output_dir: Path = Path("./checkpoints"),
    ):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = setup_cuda()
        self.model.to(self.device)

        self.scaler = GradScaler()
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    @staticmethod
    def make_loader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 4,
    ) -> DataLoader:
        """Build a DataLoader with GPU-optimal settings per PRD."""
        if num_workers > 0:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )

    def setup_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        """Setup optimizer and scheduler."""
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = get_optimizer(
            self.model,
            lr=lr,
            weight_decay=weight_decay,
            use_8bit=self.config.use_8bit_adam,
        )

        # Pick per-model sub-config for label_smoothing and warmup_epochs
        sub_cfg = getattr(self.config, self.model_name, None)
        warmup_epochs = getattr(sub_cfg, 'warmup_epochs', 2)
        label_smoothing = getattr(sub_cfg, 'label_smoothing', 0.1)

        # Linear warm-up for warmup_epochs, then ReduceLROnPlateau
        self._warmup_epochs = warmup_epochs
        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        plateau = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        self._warmup_scheduler = warmup
        self._plateau_scheduler = plateau

        self.dir_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # HuberLoss is more robust than MSE for magnitude — outlier pip spikes
        # won't dominate and collapse the loss to zero
        self.mag_criterion = nn.HuberLoss(delta=0.01)

        logger.info(
            "training_setup",
            model=self.model_name,
            lr=lr,
            weight_decay=weight_decay,
            use_8bit=self.config.use_8bit_adam,
        )

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_dir_loss = 0.0
        total_mag_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            # Split labels: [direction, magnitude]
            batch_y_dir = batch_y[:, 0].long()
            # log1p-scale magnitude so tiny pip values aren't drowned to zero
            batch_y_mag = torch.log1p(batch_y[:, 1] * 1e4)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                dir_logits, mag_pred = self.model(batch_x)

                dir_loss = self.dir_criterion(dir_logits, batch_y_dir)
                mag_loss = self.mag_criterion(mag_pred.squeeze(), batch_y_mag)

                loss = dir_loss + 0.3 * mag_loss

            # Backward with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_dir_loss += dir_loss.item()
            total_mag_loss += mag_loss.item()
            n_batches += 1

            del batch_x, batch_y

        # Log VRAM
        log_vram()

        return {
            "train_loss": total_loss / n_batches,
            "train_dir_loss": total_dir_loss / n_batches,
            "train_mag_loss": total_mag_loss / n_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            batch_y_dir = batch_y[:, 0].long()
            batch_y_mag = torch.log1p(batch_y[:, 1] * 1e4)

            with autocast("cuda"):
                dir_logits, mag_pred = self.model(batch_x)

                dir_loss = self.dir_criterion(dir_logits, batch_y_dir)
                mag_loss = self.mag_criterion(mag_pred.squeeze(), batch_y_mag)
                loss = dir_loss + 0.3 * mag_loss

            # Accuracy
            pred = dir_logits.argmax(dim=-1)
            correct += (pred == batch_y_dir).sum().item()
            total += batch_y_dir.size(0)

            total_loss += loss.item()
            n_batches += 1

            del batch_x, batch_y

        return {
            "val_loss": total_loss / n_batches,
            "val_accuracy": correct / total if total > 0 else 0,
        }

    def train(
        self,
        epochs: int = 100,
        patience: int = 15,
    ) -> None:
        """Full training loop with early stopping."""
        mlflow.set_experiment("aqrf_dl_models")

        with mlflow.start_run(run_name=self.model_name):
            mlflow.log_params(self.config.model_dump())

            for epoch in range(epochs):
                self.epoch = epoch
                epoch_start = time.time()

                # Train
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                epoch_time = time.time() - epoch_start

                # Combine metrics
                metrics = {
                    **train_metrics,
                    **val_metrics,
                    "epoch": epoch,
                    "epoch_time": epoch_time,
                }

                # Log to MLflow
                mlflow.log_metrics(metrics, step=epoch)

                logger.info(
                    "epoch_complete",
                    model=self.model_name,
                    **metrics,
                )

                # Warm-up for first N epochs, then ReduceLROnPlateau
                if epoch < self._warmup_epochs:
                    self._warmup_scheduler.step()
                else:
                    self._plateau_scheduler.step(val_metrics["val_loss"])

                # Early stopping
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.patience_counter = 0
                    self.save_checkpoint("best")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        logger.info("early_stopping_triggered", epoch=epoch)
                        break

            # Save final
            self.save_checkpoint("final")

    def save_checkpoint(self, suffix: str = "best") -> None:
        """Save model checkpoint."""
        path = self.output_dir / f"{self.model_name}_{suffix}.pth"

        # Save FP32
        torch.save(self.model.state_dict(), path)

        # Save FP16 for inference
        fp16_path = self.output_dir / f"{self.model_name}_{suffix}_fp16.pth"
        torch.save(self.model.half().state_dict(), fp16_path)
        self.model.float()  # Restore FP32

        logger.info("checkpoint_saved", path=str(path), suffix=suffix)

    def cleanup(self) -> None:
        """Release all resources."""
        del self.optimizer, self.scaler, self._warmup_scheduler, self._plateau_scheduler
        release_vram()
        logger.info("trainer_cleanup_complete")
