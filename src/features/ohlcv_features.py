"""OHLCV feature engineering for AQRF.

Pure math features from open/high/low/close/volume.
Zero traditional indicators. Zero lookahead bias.
"""
from typing import Optional

import numpy as np
import polars as pl
import structlog

from src.utils.config import DataConfig, FeatureConfig

logger = structlog.get_logger(__name__)


def compute_ohlcv_features(
    df: pl.DataFrame,
    config: Optional[FeatureConfig] = None,
    data_config: Optional[DataConfig] = None,
) -> pl.DataFrame:
    """Compute OHLCV-based features.

    Args:
        df: DataFrame with open, high, low, close, volume
        config: Feature configuration
        data_config: Data configuration

    Returns:
        DataFrame with engineered features
    """
    config = config or FeatureConfig()
    data_config = data_config or DataConfig()
    pip = data_config.pip_size

    # Basic returns
    df = df.with_columns([
        (pl.col("close") / pl.col("open")).log().alias("log_return_open_close"),
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return_close_close"),
        ((pl.col("high") - pl.col("low")) / pip).alias("high_low_range_pips"),
    ])

    # Realized volatility (rolling std of log returns)
    for window in config.vol_windows:
        df = df.with_columns(
            pl.col("log_return_close_close")
            .rolling_std(window_size=window, min_periods=window)
            .alias(f"realized_vol_{window}")
        )

    # Price velocity (rate of change)
    for window in config.velocity_windows:
        df = df.with_columns(
            ((pl.col("close") - pl.col("close").shift(window)) / window)
            .alias(f"price_velocity_{window}")
        )

    # Price acceleration (change in velocity)
    df = df.with_columns(
        (pl.col("price_velocity_5") - pl.col("price_velocity_5").shift(1))
        .alias("price_acceleration")
    )

    # Order flow imbalance (requires tick data aggregation, placeholder for OHLCV)
    # Approximate using close position in range
    df = df.with_columns(
        ((pl.col("close") - pl.col("low")) / 
         (pl.col("high") - pl.col("low")).clip_min(1e-8) * 2 - 1)
        .alias("close_position")
    )

    # Volume features
    df = df.with_columns([
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20, min_periods=20))
        .alias("volume_ratio_20"),
        pl.col("volume").log().alias("log_volume"),
    ])

    # Rolling mean spread proxy (using high-low as proxy)
    df = df.with_columns(
        pl.col("high_low_range_pips")
        .rolling_mean(window_size=20, min_periods=20)
        .alias("rolling_mean_range_20")
    )

    # VWAP deviation
    typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    vwap = (typical_price * pl.col("volume")).cum_sum() / pl.col("volume").cum_sum()

    df = df.with_columns([
        vwap.alias("vwap"),
        ((pl.col("close") - vwap) / 
         pl.col("realized_vol_20").clip_min(1e-8))
        .alias("vwap_deviation")
    ])

    # Autocorrelation features - skip for now, not available in this polars version
    # returns = pl.col("log_return_close_close")
    # for lag in config.autocorr_windows:
    #     df = df.with_columns(
    #         returns.rolling_corr(
    #             returns.shift(lag),
    #             window_size=config.autocorr_rolling,
    #             min_periods=config.autocorr_rolling
    #         ).alias(f"autocorr_lag{lag}")
    #     )

    logger.info("ohlcv_features_computed", columns=df.columns, rows=df.height)
    return df
