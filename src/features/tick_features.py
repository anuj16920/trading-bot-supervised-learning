"""Tick-level feature engineering for AQRF.

Pure math from bid/ask/volume — zero traditional indicators.
All features use only past data (no lookahead bias).
"""
import numpy as np
import polars as pl
import structlog

from src.utils.config import DataConfig, FeatureConfig

logger = structlog.get_logger(__name__)


def compute_tick_features(
    df: pl.DataFrame,
    config: Optional[DataConfig] = None,
) -> pl.DataFrame:
    """Compute tick-level features.

    Args:
        df: DataFrame with bid, ask, volume columns
        config: Data configuration

    Returns:
        DataFrame with added feature columns
    """
    config = config or DataConfig()
    pip = config.pip_size

    df = df.with_columns([
        # Mid price
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_price"),

        # Spread in pips
        ((pl.col("ask") - pl.col("bid")) / pip).alias("spread_pips"),
    ])

    df = df.with_columns([
        # Log return (uses only past data via shift)
        (pl.col("mid_price") / pl.col("mid_price").shift(1)).log().alias("log_return"),

        # Tick direction
        pl.col("log_return").sign().alias("tick_direction"),

        # Volume (raw)
        pl.col("volume").alias("volume_raw"),
    ])

    # Fill first row nulls
    df = df.with_columns([
        pl.col("log_return").fill_null(0),
        pl.col("tick_direction").fill_null(0),
    ])

    logger.info("tick_features_computed", columns=df.columns, rows=df.height)
    return df
