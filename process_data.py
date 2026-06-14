"""Process EURUSD multi-timeframe OHLCV data into RL-ready feature arrays.

Reads from EURUSD/ohlcv/{1min,5min,15min,1hour,4hour}/ directories.
Outputs: data/processed/{train,val,test}_features.npy  shape (N, seq_len, n_features)
         data/processed/{train,val,test}_prices.npy    shape (N, 2)  [bid, ask mid]
         data/processed/norm_stats.npz                 mean/std from train split

Features per M1 bar (32 total):
  M1  (9): return_1, return_5, return_20, range_pct, body_pct, volume_ratio,
           ma_ratio_20, vol_20, spread_pct
  M5  (5): m5_return, m5_range, m5_vol, m5_ma20_ratio, m5_body
  M15 (5): m15_return, m15_range, m15_vol, m15_ma20_ratio, m15_body
  H1  (5): h1_return, h1_range, h1_vol, h1_ma20_ratio, h1_body
  H4  (4): h4_return, h4_range, h4_vol, h4_ma20_ratio
  Session (4): hour_sin, hour_cos, dow_sin, dow_cos
"""
import gc
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from tqdm import tqdm

from src.utils.logging import setup_logging

logger = structlog.get_logger(__name__)

N_FEATURES = 32
SEQ_LEN = 60
STRIDE = 3

DATA_BASE  = Path("data/EURUSD")
OUTPUT_DIR = Path("data/processed")

SPLITS = {
    "train": list(range(2016, 2022)),   # 2016-2021
    "val":   [2022],
    "test":  [2023, 2024],
}


def _read_ohlcv(tf_dir: Path, years: list[int]) -> pl.DataFrame:
    """Read and concatenate yearly CSV files for a timeframe."""
    frames = []
    for year in years:
        matches = list(tf_dir.glob(f"*{year}*.csv"))
        if not matches:
            continue
        df = pl.read_csv(matches[0], try_parse_dates=False)
        # Normalise column names to lowercase
        df = df.rename({c: c.lower() for c in df.columns})
        # Ensure datetime column exists
        dt_col = next((c for c in df.columns if "date" in c or "time" in c), None)
        if dt_col and dt_col != "datetime":
            df = df.rename({dt_col: "datetime"})
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).sort("datetime")


def _ohlcv_features(df: pl.DataFrame, prefix: str, vol_window: int = 20) -> pl.DataFrame:
    """Compute return, range, body, vol, ma-ratio features for a timeframe."""
    close = pl.col("close")
    # First pass: columns that don't depend on each other
    out = df.with_columns([
        ((close / close.shift(1)) - 1).alias(f"{prefix}_return"),
        ((pl.col("high") - pl.col("low")) / close).alias(f"{prefix}_range"),
        ((close - pl.col("open")) / close).alias(f"{prefix}_body"),
        (close / close.rolling_mean(20) - 1).alias(f"{prefix}_ma20"),
    ])
    # Second pass: vol depends on {prefix}_return existing
    out = out.with_columns([
        pl.col(f"{prefix}_return").rolling_std(vol_window).alias(f"{prefix}_vol"),
    ])
    return out


def _session_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cyclic hour-of-day and day-of-week features."""
    # Parse datetime — handle timezone suffix
    dt = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
    return df.with_columns([
        (2 * np.pi * dt.dt.hour() / 24).sin().cast(pl.Float32).alias("hour_sin"),
        (2 * np.pi * dt.dt.hour() / 24).cos().cast(pl.Float32).alias("hour_cos"),
        (2 * np.pi * dt.dt.weekday() / 5).sin().cast(pl.Float32).alias("dow_sin"),
        (2 * np.pi * dt.dt.weekday() / 5).cos().cast(pl.Float32).alias("dow_cos"),
    ])


def _resample_features(m1_df: pl.DataFrame, tf_df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Left-join higher-timeframe features onto M1 bars by forward-fill.

    The TF key is shifted +1 period so that a TF candle starting at 09:00
    gets key 10:00. An M1 bar at 10:01 (bucket 10:00) therefore joins the
    09:00–10:00 TF candle, which fully closed at 10:00 — no future leakage.
    """
    if tf_df.is_empty():
        # Fill with zeros if timeframe data missing
        for col in [f"{prefix}_return", f"{prefix}_range", f"{prefix}_body",
                    f"{prefix}_vol", f"{prefix}_ma20"]:
            m1_df = m1_df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))
        return m1_df

    # Truncate datetime to timeframe resolution for join key
    tf_minutes = {"m5": 5, "m15": 15, "h1": 60, "h4": 240}
    mins = tf_minutes.get(prefix, 1)
    period_secs = mins * 60

    m1_dt = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
    tf_dt = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)

    # M1 key: floor M1 bar to its TF bucket (e.g. 10:01 -> bucket 10:00 for H1)
    m1_key = (m1_dt.dt.epoch(time_unit="s") // period_secs * period_secs).alias("_jk")
    # TF key shifted +1 period: a TF candle starting at 09:00 gets key 10:00.
    # M1 bar at 10:01 (bucket 10:00) therefore joins the 09:00 TF candle,
    # which closed at 10:00 — fully in the past, no future leakage.
    tf_key = (tf_dt.dt.epoch(time_unit="s") // period_secs * period_secs + period_secs).alias("_jk")

    feat_cols = [f"{prefix}_return", f"{prefix}_range", f"{prefix}_body",
                 f"{prefix}_vol", f"{prefix}_ma20"]

    tf_select = tf_df.select(["datetime"] + [c for c in tf_df.columns if c in feat_cols])
    tf_select = tf_select.with_columns(tf_key).drop("datetime")

    m1_joined = m1_df.with_columns(m1_key).join(tf_select, on="_jk", how="left").drop("_jk")

    # Forward-fill any missing higher-tf values
    for col in feat_cols:
        if col in m1_joined.columns:
            m1_joined = m1_joined.with_columns(pl.col(col).forward_fill())
        else:
            m1_joined = m1_joined.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))

    return m1_joined


def _resample_ohlcv(m1_df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """Resample M1 dataframe to a higher timeframe in-memory."""
    period_secs = minutes * 60
    dt = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
    epoch = dt.dt.epoch(time_unit="s")
    bucket = (epoch // period_secs * period_secs).alias("_bucket")

    # Build synthetic datetime string for the bucket open time
    resampled = (
        m1_df.with_columns(bucket)
        .group_by("_bucket")
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ])
        .sort("_bucket")
        .with_columns(
            # Convert epoch bucket back to datetime string for _resample_features compatibility
            pl.from_epoch(pl.col("_bucket"), time_unit="s")
              .cast(pl.Datetime)
              .dt.strftime("%Y-%m-%d %H:%M:%S")
              .alias("datetime")
        )
        .drop("_bucket")
    )
    return resampled


def process_split(years: list[int], split_name: str, norm_stats: dict | None = None):
    """Process one data split, return (X, prices, timestamps, norm_stats)."""
    logger.info("processing_split", split=split_name, years=years)

    m1_dir = DATA_BASE / "M1"
    h1_dir = DATA_BASE / "H1"

    all_features, all_prices, all_timestamps = [], [], []

    for year in tqdm(years, desc=f"{split_name}"):
        # ── Load M1 ──────────────────────────────────────────────────
        m1_matches = list(m1_dir.glob(f"*{year}*.csv"))
        if not m1_matches:
            logger.warning("m1_file_not_found", year=year)
            continue

        m1 = pl.read_csv(m1_matches[0], try_parse_dates=False)
        m1 = m1.rename({c: c.lower() for c in m1.columns})
        dt_col = next((c for c in m1.columns if "date" in c or "time" in c), None)
        if dt_col and dt_col != "datetime":
            m1 = m1.rename({dt_col: "datetime"})

        if m1.height < 1000:
            continue

        # ── M1 features ──────────────────────────────────────────────
        m1 = m1.with_columns([
            ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("return_1"),
        ])
        m1 = m1.with_columns([
            ((pl.col("close") / pl.col("close").shift(5)) - 1).alias("return_5"),
            ((pl.col("close") / pl.col("close").shift(20)) - 1).alias("return_20"),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct"),
            ((pl.col("close") - pl.col("open")) / pl.col("close")).alias("body_pct"),
        ])
        m1 = m1.with_columns([
            (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio"),
            (pl.col("close") / pl.col("close").rolling_mean(20) - 1).alias("ma_ratio_20"),
            pl.col("return_1").rolling_std(20).alias("vol_20"),
        ])
        # Spread feature (use spread_avg if present, else 0)
        if "spread_avg" in m1.columns:
            m1 = m1.with_columns(
                (pl.col("spread_avg") / pl.col("close")).alias("spread_pct")
            )
        else:
            m1 = m1.with_columns(pl.lit(0.0).cast(pl.Float32).alias("spread_pct"))

        # ── Session features ─────────────────────────────────────────
        m1 = _session_features(m1)

        # ── Higher timeframe features ─────────────────────────────────
        # M5/M15/H4 are resampled from M1 in-memory; H1 loaded from disk if present
        m1_raw_for_resample = m1.select(["datetime", "open", "high", "low", "close", "volume"])
        for prefix, minutes in [("m5", 5), ("m15", 15), ("h1", 60), ("h4", 240)]:
            if prefix == "h1":
                tf_raw = _read_ohlcv(h1_dir, [year])
                if tf_raw.is_empty():
                    tf_raw = _resample_ohlcv(m1_raw_for_resample, minutes)
            else:
                tf_raw = _resample_ohlcv(m1_raw_for_resample, minutes)
            if not tf_raw.is_empty():
                tf_raw = _ohlcv_features(tf_raw, prefix)
            m1 = _resample_features(m1, tf_raw, prefix)

        # Drop H4 body (we only take 4 h4 features)
        if "h4_body" in m1.columns:
            m1 = m1.drop("h4_body")

        # ── Build feature matrix ──────────────────────────────────────
        feature_cols = [
            # M1 (9)
            "return_1", "return_5", "return_20",
            "range_pct", "body_pct", "volume_ratio",
            "ma_ratio_20", "vol_20", "spread_pct",
            # M5 (5)
            "m5_return", "m5_range", "m5_body", "m5_vol", "m5_ma20",
            # M15 (5)
            "m15_return", "m15_range", "m15_body", "m15_vol", "m15_ma20",
            # H1 (5)
            "h1_return", "h1_range", "h1_body", "h1_vol", "h1_ma20",
            # H4 (4)
            "h4_return", "h4_range", "h4_vol", "h4_ma20",
            # Session (4)
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ]

        # Keep only columns that exist
        feature_cols = [c for c in feature_cols if c in m1.columns]

        m1 = m1.drop_nulls(subset=feature_cols)

        if m1.height < SEQ_LEN + 10:
            continue

        features = m1.select(feature_cols).to_numpy().astype(np.float32)

        # Timestamps: parse datetime column to Unix seconds (int64)
        dt_parsed = (
            pl.col("datetime")
            .str.replace(r"\+.*$", "")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            .dt.epoch(time_unit="s")
        )
        timestamps = m1.select(dt_parsed.alias("ts"))["ts"].to_numpy().astype(np.int64)

        # Price array: use mid = (bid+ask)/2 estimate from close + half spread
        if "spread_avg" in m1.columns:
            spread = m1["spread_avg"].to_numpy()
        else:
            spread = np.zeros(len(features), dtype=np.float32)
        close_arr = m1["close"].to_numpy()
        bid = (close_arr - spread / 2).astype(np.float32)
        ask = (close_arr + spread / 2).astype(np.float32)
        prices = np.stack([bid, ask], axis=1)  # (N, 2)

        # ── Create sequences ──────────────────────────────────────────
        n_rows = len(features)
        for i in range(0, n_rows - SEQ_LEN, STRIDE):
            all_features.append(features[i:i + SEQ_LEN])
            all_prices.append(prices[i + SEQ_LEN - 1])
            all_timestamps.append(timestamps[i + SEQ_LEN - 1])  # timestamp of last bar in seq

        logger.info("year_done", year=year, sequences=len(all_features))
        del m1, features, prices, bid, ask, timestamps
        gc.collect()

    if not all_features:
        logger.error("no_data_for_split", split=split_name)
        return None, None, None, norm_stats

    X = np.stack(all_features,   axis=0)   # (N, seq_len, n_features)
    P = np.stack(all_prices,     axis=0)   # (N, 2)
    T = np.array(all_timestamps, dtype=np.int64)  # (N,) Unix seconds

    actual_n = X.shape[2]
    logger.info("split_shape", split=split_name, X=X.shape, P=P.shape, n_features=actual_n)

    # ── Normalize (compute from train, apply to all) ──────────────────
    if norm_stats is None:
        # Train split: compute mean/std over (N, seq_len) axis
        mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, F)
        std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        norm_stats = {"mean": mean, "std": std}
        logger.info("norm_stats_computed")

    X = (X - norm_stats["mean"]) / norm_stats["std"]
    X = np.clip(X, -5.0, 5.0)

    del all_features, all_prices, all_timestamps
    gc.collect()
    return X, P, T, norm_stats


def main():
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    norm_stats = None

    for split_name, years in SPLITS.items():
        X, P, T, norm_stats = process_split(years, split_name, norm_stats)
        if X is None:
            logger.error("split_failed", split=split_name)
            continue

        x_path = OUTPUT_DIR / f"{split_name}_features.npy"
        p_path = OUTPUT_DIR / f"{split_name}_prices.npy"
        t_path = OUTPUT_DIR / f"{split_name}_timestamps.npy"

        # Temp-rename to avoid Windows mmap lock conflicts
        x_tmp = OUTPUT_DIR / f"{split_name}_features_tmp.npy"
        p_tmp = OUTPUT_DIR / f"{split_name}_prices_tmp.npy"
        t_tmp = OUTPUT_DIR / f"{split_name}_timestamps_tmp.npy"
        np.save(str(x_tmp), X)
        np.save(str(p_tmp), P)
        np.save(str(t_tmp), T)
        x_tmp.replace(x_path)
        p_tmp.replace(p_path)
        t_tmp.replace(t_path)

        logger.info("saved", split=split_name, X=X.shape, mb=X.nbytes // (1024 * 1024))
        del X, P, T
        gc.collect()

    # Save normalisation stats
    np.savez(str(OUTPUT_DIR / "norm_stats.npz"), **norm_stats)
    logger.info("norm_stats_saved")
    logger.info("processing_complete")


if __name__ == "__main__":
    main()
