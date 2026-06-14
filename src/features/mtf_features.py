"""Multi-timeframe feature engineering for AQRF.

Resamples M1 data to M5, M15, H1, H4.
Aligns all timeframes to M1 index via forward fill.
"""
from typing import Optional

import polars as pl
import structlog

from src.utils.config import FeatureConfig
from src.features.ohlcv_features import compute_ohlcv_features

logger = structlog.get_logger(__name__)


def resample_ohlcv(
    df: pl.DataFrame,
    timeframe: str,
) -> pl.DataFrame:
    """Resample M1 OHLCV to higher timeframe.

    Args:
        df: M1 OHLCV DataFrame
        timeframe: Target timeframe (M5, M15, H1, H4)

    Returns:
        Resampled DataFrame
    """
    # Parse timeframe to interval
    if timeframe == "M5":
        interval = "5m"
    elif timeframe == "M15":
        interval = "15m"
    elif timeframe == "H1":
        interval = "1h"
    elif timeframe == "H4":
        interval = "4h"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Group by time bucket
    df = df.with_columns(
        pl.col("timestamp").dt.truncate(interval).alias("bucket")
    )

    resampled = df.group_by("bucket").agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]).sort("bucket").rename({"bucket": "timestamp"})

    logger.info("resampled", from_tf="M1", to_tf=timeframe, rows=resampled.height)
    return resampled


def compute_mtf_features(
    m1_df: pl.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pl.DataFrame:
    """Compute multi-timeframe features aligned to M1.

    Args:
        m1_df: M1 OHLCV DataFrame
        config: Feature configuration

    Returns:
        M1 DataFrame with higher-TF features appended
    """
    config = config or FeatureConfig()
    base_df = m1_df.clone()

    for tf in config.timeframes:
        if tf == "M1":
            continue

        # Resample
        tf_df = resample_ohlcv(m1_df, tf)

        # Compute features for this TF
        tf_features = compute_ohlcv_features(tf_df, config)

        # Select key features to merge (avoid column name collisions)
        cols_to_merge = ["timestamp"] + [
            c for c in tf_features.columns 
            if c not in ["open", "high", "low", "close", "volume", "timestamp"]
        ]

        # Rename with TF suffix
        tf_features = tf_features.select(cols_to_merge)
        rename_map = {c: f"{c}_{tf}" for c in cols_to_merge if c != "timestamp"}
        tf_features = tf_features.rename(rename_map)

        # Forward fill align to M1 (asof join)
        base_df = base_df.join_asof(
            tf_features,
            on="timestamp",
            strategy="forward",
        )

    logger.info("mtf_features_computed", columns=base_df.columns, rows=base_df.height)
    return base_df
