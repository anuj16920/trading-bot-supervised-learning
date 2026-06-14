"""Prepare features from OHLCV data in chunks.

Processes data in batches to avoid memory issues.
Creates memory-mapped numpy arrays for efficient training.
"""
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import structlog
from tqdm import tqdm

logger = structlog.get_logger(__name__)


def compute_simple_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute basic features from OHLCV data."""
    
    # Returns
    df = df.with_columns([
        ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("return_1"),
        ((pl.col("close") / pl.col("close").shift(5)) - 1).alias("return_5"),
        ((pl.col("close") / pl.col("close").shift(20)) - 1).alias("return_20"),
    ])
    
    # Price features
    df = df.with_columns([
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct"),
        ((pl.col("close") - pl.col("open")) / pl.col("close")).alias("body_pct"),
        (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio"),
    ])
    
    # Moving averages
    for window in [5, 20, 60]:
        df = df.with_columns([
            (pl.col("close") / pl.col("close").rolling_mean(window) - 1).alias(f"ma_ratio_{window}"),
        ])
    
    # Volatility
    for window in [20, 60]:
        df = df.with_columns([
            pl.col("return_1").rolling_std(window).alias(f"vol_{window}"),
        ])
    
    # Momentum
    df = df.with_columns([
        (pl.col("close") - pl.col("close").shift(20)).alias("momentum_20"),
        (pl.col("close") - pl.col("close").shift(60)).alias("momentum_60"),
    ])
    
    # RSI-like
    df = df.with_columns([
        pl.when(pl.col("return_1") > 0)
        .then(pl.col("return_1"))
        .otherwise(0)
        .rolling_mean(14)
        .alias("gain_14"),
        
        pl.when(pl.col("return_1") < 0)
        .then(-pl.col("return_1"))
        .otherwise(0)
        .rolling_mean(14)
        .alias("loss_14"),
    ])
    
    df = df.with_columns([
        (pl.col("gain_14") / (pl.col("gain_14") + pl.col("loss_14") + 1e-10)).alias("rsi_14"),
    ])
    
    # Spread features
    df = df.with_columns([
        (pl.col("spread_avg") / pl.col("close")).alias("spread_pct"),
        pl.col("spread_avg").rolling_mean(20).alias("spread_ma20"),
    ])
    
    return df


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int = 60,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences from features and labels.
    
    Args:
        features: (n_samples, n_features)
        labels: (n_samples, 2) - [direction, magnitude]
        seq_len: Sequence length
        stride: Step size between sequences
        
    Returns:
        X: (n_sequences, seq_len, n_features)
        y: (n_sequences, 2)
    """
    n_samples = len(features)
    n_sequences = (n_samples - seq_len) // stride + 1
    
    X = np.zeros((n_sequences, seq_len, features.shape[1]), dtype=np.float32)
    y = np.zeros((n_sequences, 2), dtype=np.float32)
    
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        
        X[i] = features[start_idx:end_idx]
        y[i] = labels[end_idx - 1]  # Label is at the end of sequence
    
    return X, y


def process_year_chunk(
    csv_path: Path,
    seq_len: int = 60,
    stride: int = 1,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a single year CSV file."""
    
    logger.info("processing_file", path=str(csv_path))
    
    try:
        # Read CSV with Polars (fast!)
        df = pl.read_csv(
            csv_path,
            dtypes={
                "datetime": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "spread_avg": pl.Float64,
                "tick_count": pl.Int64,
            }
        )
        
        logger.info("csv_loaded", rows=len(df))
        
        # Compute features
        df = compute_simple_features(df)
        
        # Create labels: direction (0=down, 1=up) and magnitude
        df = df.with_columns([
            pl.when(pl.col("return_1").shift(-1) > 0)
            .then(1)
            .otherwise(0)
            .alias("direction"),
            
            pl.col("return_1").shift(-1).abs().alias("magnitude"),
        ])
        
        # Select feature columns (exclude datetime, OHLCV, and intermediate columns)
        feature_cols = [
            "return_1", "return_5", "return_20",
            "range_pct", "body_pct", "volume_ratio",
            "ma_ratio_5", "ma_ratio_20", "ma_ratio_60",
            "vol_20", "vol_60",
            "momentum_20", "momentum_60",
            "rsi_14",
            "spread_pct", "spread_ma20",
        ]
        
        # Drop NaN rows
        df = df.drop_nulls()
        
        if len(df) < seq_len + 10:
            logger.warning("insufficient_data", rows=len(df))
            return None, None
        
        # Convert to numpy
        features = df.select(feature_cols).to_numpy().astype(np.float32)
        labels = df.select(["direction", "magnitude"]).to_numpy().astype(np.float32)
        
        # Normalize features (simple z-score)
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        features = np.clip(features, -5, 5)  # Clip outliers
        
        # Create sequences
        X, y = create_sequences(features, labels, seq_len=seq_len, stride=stride)
        
        logger.info("sequences_created", n_sequences=len(X), n_features=features.shape[1])
        
        # Cleanup
        del df, features, labels
        gc.collect()
        
        return X, y
        
    except Exception as e:
        logger.error("processing_failed", path=str(csv_path), error=str(e))
        return None, None


def prepare_dataset(
    data_dir: Path = Path("data/EURUSD/M1"),
    output_dir: Path = Path("data/processed"),
    seq_len: int = 60,
    stride: int = 5,  # Skip every 5 bars to reduce data size
    train_years: list = [2016, 2017, 2018, 2019, 2020, 2021],
    val_years: list = [2022],
    test_years: list = [2023, 2024],
):
    """Prepare training, validation, and test datasets."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, years in [("train", train_years), ("val", val_years), ("test", test_years)]:
        logger.info("processing_split", split=split_name, years=years)
        
        all_X = []
        all_y = []
        
        for year in tqdm(years, desc=f"Processing {split_name}"):
            csv_path = data_dir / f"EURUSD_1min_{year}.csv"
            
            if not csv_path.exists():
                logger.warning("file_not_found", path=str(csv_path))
                continue
            
            X, y = process_year_chunk(csv_path, seq_len=seq_len, stride=stride)
            
            if X is not None:
                all_X.append(X)
                all_y.append(y)
                
                # Free memory
                del X, y
                gc.collect()
        
        if not all_X:
            logger.error("no_data_processed", split=split_name)
            continue
        
        # Concatenate all years
        logger.info("concatenating_data", split=split_name, n_chunks=len(all_X))
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        
        # Free memory
        del all_X, all_y
        gc.collect()
        
        # Save as memory-mapped files
        X_path = output_dir / f"{split_name}_features.npy"
        y_path = output_dir / f"{split_name}_labels.npy"
        
        logger.info("saving_data", split=split_name, shape=X_combined.shape)
        np.save(X_path, X_combined)
        np.save(y_path, y_combined)
        
        logger.info(
            "split_complete",
            split=split_name,
            samples=len(X_combined),
            features=X_combined.shape[2],
            size_mb=X_combined.nbytes / 1024 / 1024,
        )
        
        # Free memory
        del X_combined, y_combined
        gc.collect()
    
    logger.info("dataset_preparation_complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare features for training")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/EURUSD/M1"),
        help="Directory containing OHLCV CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride between sequences (higher = less data, faster)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )
    
    prepare_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
