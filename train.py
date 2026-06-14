"""AQRF Training Entry Point

Quick start training script for TCN, Transformer, and Regime models.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import structlog

from src.models.tcn import TCN
from src.models.transformer import ForexTransformer
from src.models.regime import RegimeDetector
from src.models.trainer import DLTrainer, ForexDataset
from src.utils.config import TCNConfig, TransformerConfig, RegimeConfig, ModelConfig
from src.utils.gpu import setup_cuda, log_vram
from src.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


def create_dummy_data(output_dir: Path, n_samples: int = 10000, seq_len: int = 60, n_features: int = 12):
    """Create dummy data for testing the training pipeline."""
    logger.info("creating_dummy_data", n_samples=n_samples, seq_len=seq_len, n_features=n_features)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy features
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    
    # Create dummy labels: [direction (0 or 1), magnitude]
    y_dir = np.random.randint(0, 2, size=n_samples)
    y_mag = np.abs(np.random.randn(n_samples)) * 0.01
    y = np.stack([y_dir, y_mag], axis=1).astype(np.float32)
    
    # Save as memory-mapped files
    train_size = int(n_samples * 0.8)
    
    np.save(output_dir / "train_features.npy", X[:train_size])
    np.save(output_dir / "train_features_labels.npy", y[:train_size])
    np.save(output_dir / "val_features.npy", X[train_size:])
    np.save(output_dir / "val_features_labels.npy", y[train_size:])
    
    logger.info("dummy_data_created", train_samples=train_size, val_samples=n_samples - train_size)


def train_model(model_name: str, use_dummy: bool = False):
    """Train a specific model."""
    logger.info("starting_training", model=model_name)
    
    # Setup logging
    setup_logging(log_level="INFO", pretty=False)
    
    # Setup CUDA
    device = setup_cuda()
    log_vram()
    
    # Create or check for data
    data_dir = Path("./data/processed")
    labels_exist = (data_dir / "train_features_labels.npy").exists()
    if use_dummy or not (data_dir / "train_features.npy").exists() or not labels_exist:
        logger.warning("no_processed_data_found", creating_dummy=True)
        create_dummy_data(data_dir)
    
    # Use reprocessed data if available (5-bar labels with pip threshold)
    train_feat = (data_dir / "train_features_v2.npy") if (data_dir / "train_features_v2.npy").exists() else (data_dir / "train_features.npy")
    train_lab  = (data_dir / "train_features_v2_labels.npy") if (data_dir / "train_features_v2_labels.npy").exists() else (data_dir / "train_features_labels.npy")
    val_feat   = (data_dir / "val_features_v2.npy") if (data_dir / "val_features_v2.npy").exists() else (data_dir / "val_features.npy")
    val_lab    = (data_dir / "val_features_v2_labels.npy") if (data_dir / "val_features_v2_labels.npy").exists() else (data_dir / "val_features_labels.npy")

    # Load datasets
    train_dataset = ForexDataset(
        mmap_path=str(train_feat),
        labels_path=str(train_lab),
        seq_len=60,
    )

    val_dataset = ForexDataset(
        mmap_path=str(val_feat),
        labels_path=str(val_lab),
        seq_len=60,
    )
    
    # Create model and config — model_config must share the same sub-config instance
    # so trainer uses the same lr/wd/batch_size as the model was built with
    model_config = ModelConfig()
    if model_name == "tcn":
        config = model_config.tcn
        model = TCN(config)
        batch_size = config.batch_size
        lr = config.learning_rate
        wd = config.weight_decay
    elif model_name == "transformer":
        config = model_config.transformer
        model = ForexTransformer(config)
        batch_size = config.batch_size
        lr = config.learning_rate
        wd = config.weight_decay
    elif model_name == "regime":
        config = model_config.regime
        model = RegimeDetector(config)
        batch_size = config.batch_size
        lr = config.learning_rate
        wd = config.weight_decay
    else:
        logger.error("invalid_model", model=model_name)
        sys.exit(1)
    
    logger.info("model_created", model=model_name, params=sum(p.numel() for p in model.parameters()))
    
    # Create data loaders — use 0 workers on Windows (no fork), 4 on Linux/WSL2
    import platform
    n_workers = 0 if platform.system() == "Windows" else 4

    train_loader = DLTrainer.make_loader(train_dataset, batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DLTrainer.make_loader(val_dataset, batch_size, shuffle=False, num_workers=n_workers)
    
    # Create trainer
    trainer = DLTrainer(
        model=model,
        model_name=model_name,
        config=model_config,
        output_dir=Path("./checkpoints"),
    )
    
    # Setup training
    trainer.setup_training(
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=wd,
    )
    
    # Train
    try:
        trainer.train(epochs=model_config.epochs, patience=model_config.early_stopping_patience)
        logger.info("training_complete", model=model_name)
    except KeyboardInterrupt:
        logger.warning("training_interrupted")
        trainer.save_checkpoint("interrupted")
    finally:
        trainer.cleanup()


def main():
    parser = argparse.ArgumentParser(description="AQRF Model Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tcn", "transformer", "regime"],
        help="Model to train",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy data for testing",
    )
    
    args = parser.parse_args()
    
    train_model(args.model, use_dummy=args.dummy)


if __name__ == "__main__":
    main()
