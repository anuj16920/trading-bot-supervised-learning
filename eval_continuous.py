"""True calendar monthly evaluator for AQRF.

Unlike eval_monthly.py (which averages random episode windows), this script
runs a single continuous simulation from Jan 1 to Dec 31 with capital
carrying forward across months.  Each month's return is:

    (equity_at_month_end - equity_at_month_start) / equity_at_month_start

Open positions are NOT force-closed at month boundaries — they carry forward.
They ARE force-closed at year end to realize final P&L.

This is the only evaluator whose monthly numbers can be trusted for real
trading performance assessment.

Usage:
    python eval_continuous.py
    python eval_continuous.py --model checkpoints/rl/phase3/best/best_model.zip
    python eval_continuous.py --years 2022,2023,2024,2025,2026
    python eval_continuous.py --years 2022,2023,2024,2025,2026 --audit
        (--audit: raises slippage to 1.0 pip, spread to 1.2 pip for harsh test)
"""
import argparse
import gc
import sys
from calendar import month_abbr
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from stable_baselines3 import PPO

from src.rl.environment import ForexTradingEnv
from src.utils.config import load_config, FrictionConfig
from src.utils.logging import setup_logging
import structlog

logger = structlog.get_logger(__name__)

DATA_BASE  = Path("data/EURUSD")
SEQ_LEN    = 60
STRIDE     = 3
ALL_YEARS  = list(range(2016, 2027))
MONTHS     = list(range(1, 13))


# ── Feature engineering (identical to process_data.py, with leakage fix) ─

def _ohlcv_features(df, prefix, vol_window=20):
    close = pl.col("close")
    out = df.with_columns([
        ((close / close.shift(1)) - 1).alias(f"{prefix}_return"),
        ((pl.col("high") - pl.col("low")) / close).alias(f"{prefix}_range"),
        ((close - pl.col("open")) / close).alias(f"{prefix}_body"),
        (close / close.rolling_mean(20) - 1).alias(f"{prefix}_ma20"),
    ])
    return out.with_columns([
        pl.col(f"{prefix}_return").rolling_std(vol_window).alias(f"{prefix}_vol"),
    ])


def _session_features(df):
    dt = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(
        pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False
    )
    return df.with_columns([
        (2 * np.pi * dt.dt.hour() / 24).sin().cast(pl.Float32).alias("hour_sin"),
        (2 * np.pi * dt.dt.hour() / 24).cos().cast(pl.Float32).alias("hour_cos"),
        (2 * np.pi * dt.dt.weekday() / 5).sin().cast(pl.Float32).alias("dow_sin"),
        (2 * np.pi * dt.dt.weekday() / 5).cos().cast(pl.Float32).alias("dow_cos"),
    ])


def _read_ohlcv(tf_dir, year):
    matches = list(tf_dir.glob(f"*{year}*.csv"))
    if not matches:
        return pl.DataFrame()
    df = pl.read_csv(matches[0], try_parse_dates=False)
    df = df.rename({c: c.lower() for c in df.columns})
    dt_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if dt_col and dt_col != "datetime":
        df = df.rename({dt_col: "datetime"})
    return df.sort("datetime")


def _resample_features(m1_df, tf_df, prefix):
    """Join higher-TF features with 1-candle lag to prevent future leakage."""
    feat_cols = [f"{prefix}_return", f"{prefix}_range", f"{prefix}_body",
                 f"{prefix}_vol", f"{prefix}_ma20"]
    if tf_df.is_empty():
        for col in feat_cols:
            m1_df = m1_df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))
        return m1_df

    tf_minutes = {"m5": 5, "m15": 15, "h1": 60, "h4": 240}
    mins = tf_minutes.get(prefix, 1)
    period_secs = mins * 60

    m1_dt = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
    tf_dt  = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)

    m1_key = (m1_dt.dt.epoch(time_unit="s") // period_secs * period_secs).alias("_jk")
    # TF key shifted +1 period: no future leakage (matches process_data.py fix).
    tf_key  = (tf_dt.dt.epoch(time_unit="s") // period_secs * period_secs + period_secs).alias("_jk")

    tf_select = tf_df.select(["datetime"] + [c for c in tf_df.columns if c in feat_cols])
    tf_select = tf_select.with_columns(tf_key).drop("datetime")
    m1_joined = m1_df.with_columns(m1_key).join(tf_select, on="_jk", how="left").drop("_jk")
    for col in feat_cols:
        if col in m1_joined.columns:
            m1_joined = m1_joined.with_columns(pl.col(col).forward_fill())
        else:
            m1_joined = m1_joined.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))
    return m1_joined


def _resample_ohlcv(m1_df, minutes):
    """Resample M1 dataframe to a higher timeframe in-memory."""
    period_secs = minutes * 60
    dt = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
    bucket = (dt.dt.epoch(time_unit="s") // period_secs * period_secs).alias("_bucket")
    return (
        m1_df.select(["datetime", "open", "high", "low", "close", "volume"])
        .with_columns(bucket)
        .group_by("_bucket")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
        .sort("_bucket")
        .with_columns(
            pl.from_epoch(pl.col("_bucket"), time_unit="s")
              .cast(pl.Datetime).dt.strftime("%Y-%m-%d %H:%M:%S").alias("datetime")
        )
        .drop("_bucket")
    )


def build_year_data(year, norm_stats):
    """Build (features, prices, month_labels) for a full year."""
    m1_dir = DATA_BASE / "M1"
    h1_dir = DATA_BASE / "H1"

    m1_matches = list(m1_dir.glob(f"*{year}*.csv"))
    if not m1_matches:
        return None, None, None

    m1 = pl.read_csv(m1_matches[0], try_parse_dates=False)
    m1 = m1.rename({c: c.lower() for c in m1.columns})
    dt_col = next((c for c in m1.columns if "date" in c or "time" in c), None)
    if dt_col and dt_col != "datetime":
        m1 = m1.rename({dt_col: "datetime"})
    if m1.height < 1000:
        return None, None, None

    m1 = m1.with_columns(
        pl.col("datetime").str.slice(5, 2).cast(pl.Int32).alias("_month")
    )

    m1 = m1.with_columns([((pl.col("close") / pl.col("close").shift(1)) - 1).alias("return_1")])
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
    if "spread_avg" in m1.columns:
        m1 = m1.with_columns((pl.col("spread_avg") / pl.col("close")).alias("spread_pct"))
    else:
        m1 = m1.with_columns(pl.lit(0.0).cast(pl.Float32).alias("spread_pct"))

    m1 = _session_features(m1)

    m1_raw = m1.select(["datetime", "open", "high", "low", "close", "volume"])
    for prefix, minutes in [("m5", 5), ("m15", 15), ("h1", 60), ("h4", 240)]:
        if prefix == "h1":
            tf_raw = _read_ohlcv(h1_dir, year)
            if tf_raw.is_empty():
                tf_raw = _resample_ohlcv(m1_raw, minutes)
        else:
            tf_raw = _resample_ohlcv(m1_raw, minutes)
        if not tf_raw.is_empty():
            tf_raw = _ohlcv_features(tf_raw, prefix)
        m1 = _resample_features(m1, tf_raw, prefix)

    if "h4_body" in m1.columns:
        m1 = m1.drop("h4_body")

    feature_cols = [
        "return_1","return_5","return_20","range_pct","body_pct",
        "volume_ratio","ma_ratio_20","vol_20","spread_pct",
        "m5_return","m5_range","m5_body","m5_vol","m5_ma20",
        "m15_return","m15_range","m15_body","m15_vol","m15_ma20",
        "h1_return","h1_range","h1_body","h1_vol","h1_ma20",
        "h4_return","h4_range","h4_vol","h4_ma20",
        "hour_sin","hour_cos","dow_sin","dow_cos",
    ]
    feature_cols = [c for c in feature_cols if c in m1.columns]
    m1 = m1.drop_nulls(subset=feature_cols)

    features   = m1.select(feature_cols).to_numpy().astype(np.float32)
    months_arr = m1["_month"].to_numpy()

    spread    = m1["spread_avg"].to_numpy() if "spread_avg" in m1.columns else np.zeros(len(features), np.float32)
    close_arr = m1["close"].to_numpy()
    bid = (close_arr - spread / 2).astype(np.float32)
    ask = (close_arr + spread / 2).astype(np.float32)
    prices = np.stack([bid, ask], axis=1)

    mean = norm_stats["mean"].reshape(-1)
    std  = norm_stats["std"].reshape(-1)
    features = (features - mean) / std
    features = np.clip(features, -5.0, 5.0)

    return features, prices, months_arr


# ── Continuous runner ─────────────────────────────────────────────────

class ContinuousRunner:
    """Runs the agent step-by-step over a full sequence without episode resets.

    Capital and position carry forward across the entire run.
    Month boundaries are recorded so per-month equity snapshots can be taken.
    """

    def __init__(self, features, prices, months_arr, cfg_rl, model):
        self.features   = features
        self.prices     = prices
        self.months_arr = months_arr
        self.cfg        = cfg_rl
        self.model      = model

        # Build the FULL sequence (no episode slicing — stride still applies)
        n_rows = len(features)
        self.seq_indices = list(range(0, n_rows - SEQ_LEN, STRIDE))
        self.month_at    = [months_arr[i] for i in self.seq_indices]

    def run(self):
        """Simulate the whole year.  Returns per-month metrics dict."""
        capital     = self.cfg.initial_capital
        peak_cap    = capital
        position    = 0.0
        entry_price = 0.0
        bars_held   = 0
        total_trades = 0
        winning      = 0

        friction = self.cfg.friction
        spread_half = (friction.eval_spread_pips * 0.0001) / 2.0
        slip        = friction.eval_slippage_pips * 0.0001
        delay_bars  = friction.eval_delay_bars

        # Pending action queue for delay simulation
        pending_action         = None
        pending_bars_remaining = 0

        # Per-month tracking
        month_start_equity = {m: None for m in MONTHS}
        month_end_equity   = {m: None for m in MONTHS}
        month_trades       = {m: 0    for m in MONTHS}
        month_winners      = {m: 0    for m in MONTHS}
        month_equity_curves = {m: []  for m in MONTHS}

        prev_month = None

        def effective_prices(idx):
            bid_raw, ask_raw = self.prices[self.seq_indices[idx] + SEQ_LEN - 1]
            mid = (bid_raw + ask_raw) / 2.0
            return mid - spread_half, mid + spread_half

        def apply_slip(price, direction):
            return price + direction * slip

        def unrealised(idx):
            if position == 0.0:
                return 0.0
            bid, ask = effective_prices(idx)
            mid = (bid + ask) * 0.5
            return (mid - entry_price) * position * self.cfg.lot_size

        def equity(idx):
            return capital + unrealised(idx)

        # Observation builder — matches ForexTradingEnv._obs() exactly.
        # self.features is (N, n_feat); we build the (seq_len, n_feat) window
        # by slicing [start:start+SEQ_LEN] from the raw feature array.
        def make_obs(step_i):
            start = self.seq_indices[step_i]
            mkt   = self.features[start : start + SEQ_LEN]  # (seq_len, n_feat)
            upnl_pct     = unrealised(step_i) / self.cfg.initial_capital
            dd           = max(0.0, (peak_cap - equity(step_i)) / peak_cap) if peak_cap > 0 else 0.0
            held_norm    = min(bars_held / 100.0, 1.0)
            cooldown_norm = 0.0  # no cooldown tracking in continuous runner
            seq_len = mkt.shape[0]
            port = np.full((seq_len, 5), [position, upnl_pct, dd, held_norm, cooldown_norm], dtype=np.float32)
            return np.concatenate([mkt, port], axis=-1).astype(np.float32)

        n_steps = len(self.seq_indices)

        for step_i in range(n_steps):
            cur_month = self.month_at[step_i]

            # Record month start equity on first bar of each month
            if cur_month != prev_month:
                eq = equity(step_i)
                month_start_equity[cur_month] = eq
                if prev_month is not None:
                    month_end_equity[prev_month] = eq  # previous month ended here
                prev_month = cur_month

            month_equity_curves[cur_month].append(equity(step_i))

            obs = make_obs(step_i)
            raw_action, _ = self.model.predict(obs[np.newaxis], deterministic=True)
            raw_action = int(raw_action[0]) if hasattr(raw_action, '__len__') else int(raw_action)

            # Delay queue
            if pending_action is not None:
                pending_bars_remaining -= 1
                if pending_bars_remaining <= 0:
                    action = pending_action
                    pending_action = None
                else:
                    action = 0
            elif delay_bars > 0 and raw_action in (1, 2):
                pending_action = raw_action
                pending_bars_remaining = delay_bars
                action = 0
            else:
                action = raw_action

            # Check SL/TP before executing agent action
            pnl = 0.0
            trade_closed = False
            bid, ask = effective_prices(step_i)

            if position != 0.0:
                mid = (bid + ask) * 0.5
                if position > 0:
                    upnl_pips = (mid - entry_price) / 0.0001
                    if upnl_pips <= -self.cfg.stop_loss_pips:
                        ep = apply_slip(bid, -1)
                        pnl = (ep - entry_price) * position * self.cfg.lot_size
                        if pnl > 0: winning += 1
                        month_trades[cur_month] += 1
                        if pnl > 0: month_winners[cur_month] += 1
                        total_trades += 1; trade_closed = True
                        capital += pnl; position = 0.0; bars_held = 0
                    elif upnl_pips >= self.cfg.take_profit_pips:
                        ep = apply_slip(bid, -1)
                        pnl = (ep - entry_price) * position * self.cfg.lot_size
                        if pnl > 0: winning += 1
                        month_trades[cur_month] += 1
                        if pnl > 0: month_winners[cur_month] += 1
                        total_trades += 1; trade_closed = True
                        capital += pnl; position = 0.0; bars_held = 0
                elif position < 0:
                    upnl_pips = (entry_price - mid) / 0.0001
                    if upnl_pips <= -self.cfg.stop_loss_pips:
                        ep = apply_slip(ask, +1)
                        pnl = (entry_price - ep) * abs(position) * self.cfg.lot_size
                        if pnl > 0: winning += 1
                        month_trades[cur_month] += 1
                        if pnl > 0: month_winners[cur_month] += 1
                        total_trades += 1; trade_closed = True
                        capital += pnl; position = 0.0; bars_held = 0
                    elif upnl_pips >= self.cfg.take_profit_pips:
                        ep = apply_slip(ask, +1)
                        pnl = (entry_price - ep) * abs(position) * self.cfg.lot_size
                        if pnl > 0: winning += 1
                        month_trades[cur_month] += 1
                        if pnl > 0: month_winners[cur_month] += 1
                        total_trades += 1; trade_closed = True
                        capital += pnl; position = 0.0; bars_held = 0

            # Execute agent action (if SL/TP didn't already close)
            if not trade_closed:
                if action == 1:  # buy
                    if position < 0:
                        ep = apply_slip(ask, +1)
                        pnl = (entry_price - ep) * abs(position) * self.cfg.lot_size
                        capital += pnl
                        if pnl > 0: winning += 1
                        month_trades[cur_month] += 1
                        if pnl > 0: month_winners[cur_month] += 1
                        total_trades += 1
                        position = 0.0
                    if position == 0.0:
                        entry_price = apply_slip(ask, +1)
                        position = 1.0; bars_held = 0
                        total_trades += 1; month_trades[cur_month] += 1

                elif action == 2:  # sell
                    if position > 0:
                        ep = apply_slip(bid, -1)
                        pnl = (ep - entry_price) * position * self.cfg.lot_size
                        capital += pnl
                        if pnl > 0: winning += 1
                        month_trades[cur_month] += 1
                        if pnl > 0: month_winners[cur_month] += 1
                        total_trades += 1
                        position = 0.0
                    if position == 0.0:
                        entry_price = apply_slip(bid, -1)
                        position = -1.0; bars_held = 0
                        total_trades += 1; month_trades[cur_month] += 1

                elif action == 3:  # close
                    if position > 0:
                        ep = apply_slip(bid, -1)
                        pnl = (ep - entry_price) * position * self.cfg.lot_size
                        capital += pnl
                        if pnl > 0: winning += 1
                        month_trades[cur_month] += 1
                        if pnl > 0: month_winners[cur_month] += 1
                        total_trades += 1
                        position = 0.0; bars_held = 0
                    elif position < 0:
                        ep = apply_slip(ask, +1)
                        pnl = (entry_price - ep) * abs(position) * self.cfg.lot_size
                        capital += pnl
                        if pnl > 0: winning += 1
                        month_trades[cur_month] += 1
                        if pnl > 0: month_winners[cur_month] += 1
                        total_trades += 1
                        position = 0.0; bars_held = 0

            if position != 0.0:
                bars_held += 1

            peak_cap = max(peak_cap, equity(step_i))

        # Force-close at year end
        if position != 0.0:
            last_i = n_steps - 1
            bid, ask = effective_prices(last_i)
            if position > 0:
                ep = apply_slip(bid, -1)
                pnl = (ep - entry_price) * position * self.cfg.lot_size
            else:
                ep = apply_slip(ask, +1)
                pnl = (entry_price - ep) * abs(position) * self.cfg.lot_size
            capital += pnl
            last_month = self.month_at[last_i]
            month_trades[last_month] += 1
            if pnl > 0:
                winning += 1
                month_winners[last_month] += 1
            total_trades += 1
            position = 0.0

        # Close out last month
        if prev_month is not None and month_end_equity[prev_month] is None:
            month_end_equity[prev_month] = capital

        # Build per-month result rows
        results = []
        for m in MONTHS:
            start_eq = month_start_equity[m]
            end_eq   = month_end_equity[m]
            if start_eq is None or end_eq is None or start_eq <= 0:
                results.append({"month": m, "ret_pct": None, "win_rate": None,
                                 "trades": None, "trades_per_day": None,
                                 "max_dd_pct": None})
                continue
            ret_pct = (end_eq - start_eq) / start_eq * 100.0
            n_t  = month_trades[m]
            n_w  = month_winners[m]
            wr   = n_w / n_t if n_t > 0 else 0.0
            # Approximate trading days in month (~21)
            tpd  = n_t / 21.0

            curve = np.array(month_equity_curves[m])
            if len(curve) > 1:
                peak  = np.maximum.accumulate(curve)
                dd    = ((peak - curve) / peak).max() * 100.0
            else:
                dd = 0.0

            results.append({
                "month":          m,
                "ret_pct":        ret_pct,
                "win_rate":       wr,
                "trades":         n_t,
                "trades_per_day": tpd,
                "max_dd_pct":     dd,
                "start_equity":   start_eq,
                "end_equity":     end_eq,
            })

        return results, capital, total_trades, winning


# ── Main ──────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="checkpoints/rl/phase3/best/best_model.zip")
    parser.add_argument("--years",   type=str, default=None,
                        help="Comma-separated years, e.g. 2022,2023,2024,2025,2026")
    parser.add_argument("--audit",   action="store_true",
                        help="Harsh audit mode: spread=1.2pip, slip=1.0pip")
    args = parser.parse_args()

    cfg = load_config()

    # Audit mode: override friction to harsh values
    if args.audit:
        cfg.rl.friction.eval_spread_pips   = 1.2
        cfg.rl.friction.eval_slippage_pips = 1.0
        cfg.rl.friction.eval_delay_bars    = 1
        cfg.rl.friction.randomize          = False
        logger.info("audit_mode", spread=1.2, slip=1.0, delay=1)

    eval_years = ALL_YEARS if not args.years else [int(y.strip()) for y in args.years.split(",")]

    norm_path = Path("data/processed/norm_stats.npz")
    if not norm_path.exists():
        raise FileNotFoundError("norm_stats.npz not found — run process_data.py first")
    ns = np.load(str(norm_path))
    norm_stats = {"mean": ns["mean"], "std": ns["std"]}

    model_path = Path(args.model)
    if not model_path.exists():
        for c in ["checkpoints/rl/phase3/best/best_model.zip",
                  "checkpoints/rl/best/best_model.zip",
                  "checkpoints/rl/ppo_forex_final.zip"]:
            if Path(c).exists():
                model_path = Path(c); break
        else:
            raise FileNotFoundError("No model checkpoint found")

    logger.info("loading_model", path=str(model_path))
    model = PPO.load(str(model_path))

    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "AUDIT" if args.audit else "STANDARD"
    mn = [month_abbr[m] for m in MONTHS]

    header = "  Year  |  " + " |  ".join(f"{m:>4}" for m in mn) + "  | Annual"
    print(f"\n{'='*len(header)}")
    print(f"  AQRF Agent -- Continuous Calendar Eval  [{mode_tag}]")
    print(f"  Model: {model_path}")
    if args.audit:
        print(f"  Friction: spread=1.2pip  slip=1.0pip  delay=1bar")
    print(f"{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    # year -> list of 12 month result dicts
    all_year_results = {}
    # For OOS summary
    oos_years = set(range(2022, 2027))

    for year in eval_years:
        logger.info("processing_year", year=year)
        features, prices, months_arr = build_year_data(year, norm_stats)
        if features is None:
            print(f"  {year}  | {'no data':^{len(header)-12}}")
            continue

        runner  = ContinuousRunner(features, prices, months_arr, cfg.rl, model)
        results, final_cap, total_trades, winning = runner.run()

        all_year_results[year] = results

        # Annual return from start of year capital to end
        valid = [r for r in results if r["ret_pct"] is not None]
        if valid:
            start_eq = valid[0]["start_equity"]
            end_eq   = valid[-1]["end_equity"]
            annual_ret = (end_eq - start_eq) / start_eq * 100.0 if start_eq > 0 else 0.0
        else:
            annual_ret = 0.0

        def fmt(v):
            if v is None: return "  N/A"
            return f"{v:+.1f}%"

        row = [fmt(r["ret_pct"]) for r in results]
        oos_tag = " [OOS]" if year in oos_years else " [IN] "
        print(f"  {year}{oos_tag}| " + " | ".join(f"{v:>6}" for v in row) + f"  | {annual_ret:+.2f}%")

        del features, prices, months_arr
        gc.collect()

    print(f"{'='*len(header)}\n")

    # ── OOS Summary ───────────────────────────────────────────────────
    oos_res = {y: v for y, v in all_year_results.items() if y in oos_years}
    if oos_res:
        oos_monthly_r  = []
        oos_monthly_wr = []
        oos_monthly_td = []
        oos_monthly_dd = []
        for m_idx, m in enumerate(MONTHS):
            r_vals  = [all_year_results[y][m_idx]["ret_pct"]        for y in oos_res if all_year_results[y][m_idx]["ret_pct"]  is not None]
            wr_vals = [all_year_results[y][m_idx]["win_rate"]        for y in oos_res if all_year_results[y][m_idx]["win_rate"] is not None]
            td_vals = [all_year_results[y][m_idx]["trades_per_day"]  for y in oos_res if all_year_results[y][m_idx]["trades_per_day"] is not None]
            dd_vals = [all_year_results[y][m_idx]["max_dd_pct"]      for y in oos_res if all_year_results[y][m_idx]["max_dd_pct"] is not None]
            oos_monthly_r.append(np.mean(r_vals)   if r_vals  else None)
            oos_monthly_wr.append(np.mean(wr_vals)  if wr_vals else None)
            oos_monthly_td.append(np.mean(td_vals)  if td_vals else None)
            oos_monthly_dd.append(np.mean(dd_vals)  if dd_vals else None)

        print("  OOS Monthly Averages (2022-2026)")
        print("  Return :  " + "  ".join(f"{v:+.1f}%" if v is not None else "  N/A " for v in oos_monthly_r))
        print("  WinRate:  " + "  ".join(f"{v*100:.0f}%" if v is not None else "  N/A " for v in oos_monthly_wr))
        print("  Trd/Day:  " + "  ".join(f"{v:.1f} " if v is not None else " N/A  " for v in oos_monthly_td))
        print()

    # ── Save Markdown ─────────────────────────────────────────────────
    md_lines = [f"# AQRF Agent — Continuous Calendar Evaluation\n"]
    md_lines.append(f"**Model:** `{model_path}`  ")
    md_lines.append(f"**Mode:** {mode_tag}  ")
    if args.audit:
        md_lines.append(f"**Friction:** spread=1.2pip  slip=1.0pip  delay=1bar  ")
    md_lines.append(f"\n> Each row = true Jan-to-Dec continuous simulation. Capital carries forward across months.\n")

    def fmd(v):
        return "N/A" if v is None else f"{v:+.1f}%"

    # Return table
    md_lines.append("## Monthly Return %\n")
    md_lines.append("| Year | IS/OOS | " + " | ".join(mn) + " | Annual |")
    md_lines.append("|------|--------|" + "|".join(["------"]*12) + "|--------|")
    trained = set(range(2016, 2022))
    for year, results in sorted(all_year_results.items()):
        tag = "IN" if year in trained else "**OOS**"
        valid = [r for r in results if r["ret_pct"] is not None]
        annual = (valid[-1]["end_equity"] - valid[0]["start_equity"]) / valid[0]["start_equity"] * 100.0 if valid else None
        md_lines.append(f"| {year} | {tag} | " + " | ".join(fmd(r["ret_pct"]) for r in results) + f" | **{fmd(annual)}** |")

    # Trades/day table
    md_lines.append("\n## Monthly Trades/Day\n")
    md_lines.append("| Year | IS/OOS | " + " | ".join(mn) + " | Avg |")
    md_lines.append("|------|--------|" + "|".join(["------"]*12) + "|-----|")
    for year, results in sorted(all_year_results.items()):
        tag = "IN" if year in trained else "**OOS**"
        def ftd(v): return "N/A" if v is None else f"{v:.1f}"
        vals = [r["trades_per_day"] for r in results]
        avg  = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
        md_lines.append(f"| {year} | {tag} | " + " | ".join(ftd(r["trades_per_day"]) for r in results) + f" | **{ftd(avg)}** |")

    # Win rate table
    md_lines.append("\n## Monthly Win Rate\n")
    md_lines.append("| Year | IS/OOS | " + " | ".join(mn) + " | Avg |")
    md_lines.append("|------|--------|" + "|".join(["------"]*12) + "|-----|")
    for year, results in sorted(all_year_results.items()):
        tag = "IN" if year in trained else "**OOS**"
        def fwr(v): return "N/A" if v is None else f"{v*100:.1f}%"
        vals = [r["win_rate"] for r in results if r["win_rate"] is not None]
        avg  = np.mean(vals) if vals else None
        md_lines.append(f"| {year} | {tag} | " + " | ".join(fwr(r["win_rate"]) for r in results) + f" | **{fwr(avg)}** |")

    suffix = "_audit" if args.audit else ""
    md_path = out_dir / f"continuous_monthly{suffix}.md"
    md_path.write_text("\n".join(md_lines))
    logger.info("markdown_saved", path=str(md_path))

    # ── Equity curve chart ────────────────────────────────────────────
    if all_year_results:
        fig, ax = plt.subplots(figsize=(16, 6))
        cmap = plt.cm.tab20
        for i, (year, results) in enumerate(sorted(all_year_results.items())):
            valid = [r for r in results if r["ret_pct"] is not None]
            if not valid:
                continue
            xs = [r["month"] for r in valid]
            ys = []
            running = 0.0
            start_eq = valid[0]["start_equity"]
            for r in valid:
                running = (r["end_equity"] - start_eq) / start_eq * 100.0
                ys.append(running)
            is_oos = year in oos_years
            ax.plot(xs, ys, label=f"{year}{'*' if is_oos else ''}",
                    color=cmap(i / max(len(all_year_results), 1)),
                    linewidth=2.0 if is_oos else 1.0,
                    linestyle="-" if is_oos else "--")

        ax.axhline(0, color="red", linewidth=1.0, linestyle="--")
        ax.set_xticks(MONTHS)
        ax.set_xticklabels(mn)
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_title(f"AQRF — Continuous Monthly Equity [{mode_tag}]\n(* = OOS years, dashed = in-sample)")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        chart_path = out_dir / f"continuous_equity_curve{suffix}.png"
        plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("chart_saved", path=str(chart_path))

    print(f"Saved:")
    print(f"  {md_path}")
    print(f"  {out_dir}/continuous_equity_curve{suffix}.png")


if __name__ == "__main__":
    main()
