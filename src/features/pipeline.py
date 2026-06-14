"""Feature pipeline for AQRF.

Rolling z-score normalization (no lookahead bias).
Clips outliers. Outputs memory-mapped numpy arrays.
"""
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import structlog
from tqdm import tqdm

from src.utils.config import DataConfig, FeatureConfig

logger = structlog.get_logger(__name__)


def rolling_zscore(
    series: pl.Series,
    window: int = 500,
) -> pl.Series:
    """Compute rolling z-score using only past data.

    Args:
        series: Input series
        window: Rolling window size

    Returns:
        Z-scored series
    """
    rolling_mean = series.rolling_mean(window_size=window, min_periods=window)
    rolling_std = series.rolling_std(window_size=window, min_periods=window)

    # Avoid division by zero
    zscore = (series - rolling_mean) / rolling_std.clip_min(1e-8)
    return zscore


def normalize_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    config: Optional[FeatureConfig] = None,
) -> pl.DataFrame:
    """Normalize features with rolling z-score.

    Args:
        df: DataFrame with raw features
        feature_cols: Columns to normalize
        config: Feature configuration

    Returns:
        Normalized DataFrame
    """
    config = config or FeatureConfig()

    for col in feature_cols:
        if col not in df.columns:
            continue

        z = rolling_zscore(df[col], config.zscore_window)

        # Clip outliers
        z = z.clip(
            lower_bound=-config.zscore_clip,
            upper_bound=config.zscore_clip
        )

        df = df.with_columns(z.alias(f"{col}_norm"))

    logger.info("features_normalized", columns_normalized=len(feature_cols))
    return df


def extract_feature_windows(
    df: pl.DataFrame,
    feature_cols: list[str],
    seq_len: int = 60,
) -> np.ndarray:
    """Extract sliding windows for training.

    Args:
        df: DataFrame with normalized features
        feature_cols: Feature column names
        seq_len: Window length

    Returns:
        Array of shape (n_samples, seq_len, n_features)
    """
    data = df.select(feature_cols).to_numpy().astype(np.float32)
    n_samples = len(data) - seq_len + 1

    if n_samples <= 0:
        return np.array([])

    # Memory-efficient sliding window via stride tricks
    shape = (n_samples, seq_len, len(feature_cols))
    strides = (data.strides[0], data.strides[0], data.strides[1])

    windows = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides, writeable=False
    )

    # Return a copy to ensure contiguous memory
    return np.ascontiguousarray(windows)


def save_memory_mapped(
    windows: np.ndarray,
    path: Path,
) -> Path:
    """Save windows to memory-mapped file.

    Args:
        windows: Array of shape (n_samples, seq_len, n_features)
        path: Output file path

    Returns:
        Path to saved file
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save as .npy first, then memmap
    np.save(path, windows)

    # Create memmap version
    mmap_path = path.with_suffix('.mmap')
    fp = np.memmap(
        mmap_path,
        dtype='float32',
        mode='w+',
        shape=windows.shape
    )
    fp[:] = windows[:]
    fp.flush()

    logger.info(
        "memory_mapped_saved",
        path=str(mmap_path),
        shape=windows.shape,
        size_mb=windows.nbytes / 1e6,
    )
    return mmap_path


def build_feature_pipeline(
    m1_df: pl.DataFrame,
    output_dir: Path,
    config: Optional[FeatureConfig] = None,
    data_config: Optional[DataConfig] = None,
) -> dict:
    """Full feature pipeline: compute -> normalize -> window -> memmap.

    Args:
        m1_df: M1 OHLCV DataFrame
        output_dir: Directory for output files
        config: Feature configuration
        data_config: Data configuration

    Returns:
        Dict with paths and statistics
    """
    from src.features.ohlcv_features import compute_ohlcv_features
    from src.features.mtf_features import compute_mtf_features

    config = config or FeatureConfig()
    data_config = data_config or DataConfig()

    logger.info("pipeline_start", input_rows=m1_df.height)

    # 1. Compute OHLCV features
    df = compute_ohlcv_features(m1_df, config, data_config)

    # 2. Compute multi-timeframe features
    df = compute_mtf_features(df, config)

    # 3. Identify feature columns (exclude raw price/volume/timestamp)
    exclude = {"timestamp", "open", "high", "low", "close", "volume", "vwap"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # 4. Normalize
    df = normalize_features(df, feature_cols, config)

    # Use normalized columns
    norm_cols = [c for c in df.columns if c.endswith("_norm")]

    # 5. Extract windows
    windows = extract_feature_windows(df, norm_cols, config.seq_len)

    # 6. Save memory-mapped
    mmap_path = save_memory_mapped(windows, output_dir / "features.npy")

    # Cleanup
    del windows
    gc.collect()

    stats = {
        "input_rows": m1_df.height,
        "output_windows": windows.shape[0] if windows.size > 0 else 0,
        "seq_len": config.seq_len,
        "n_features": len(norm_cols),
        "mmap_path": str(mmap_path),
        "feature_columns": norm_cols,
    }
    logger.info("pipeline_complete", **stats)
    return stats
