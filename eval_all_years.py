"""Evaluate the trained PPO agent on every available year (2016-2026).

Processes each year on-the-fly (no pre-saved .npy needed), runs 30 episodes,
saves per-year metrics and a combined summary to eval_results/yearly_results.md
and eval_results/yearly_results.csv.

Usage:
    python eval_all_years.py
    python eval_all_years.py --model checkpoints/rl/best/best_model.zip
    python eval_all_years.py --episodes 30
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl
from stable_baselines3 import PPO

from src.rl.environment import ForexTradingEnv
from src.utils.config import load_config
from src.utils.logging import setup_logging
import structlog

logger = structlog.get_logger(__name__)

OHLCV_BASE = Path("EURUSD/ohlcv")
N_FEATURES = 32
SEQ_LEN = 60
STRIDE = 3

ALL_YEARS = list(range(2016, 2027))   # 2016 → 2026 inclusive


# ── Feature engineering (same logic as process_data.py) ──────────────

def _ohlcv_features(df: pl.DataFrame, prefix: str, vol_window: int = 20) -> pl.DataFrame:
    close = pl.col("close")
    out = df.with_columns([
        ((close / close.shift(1)) - 1).alias(f"{prefix}_return"),
        ((pl.col("high") - pl.col("low")) / close).alias(f"{prefix}_range"),
        ((close - pl.col("open")) / close).alias(f"{prefix}_body"),
        (close / close.rolling_mean(20) - 1).alias(f"{prefix}_ma20"),
    ])
    out = out.with_columns([
        pl.col(f"{prefix}_return").rolling_std(vol_window).alias(f"{prefix}_vol"),
    ])
    return out


def _session_features(df: pl.DataFrame) -> pl.DataFrame:
    dt = pl.col("datetime").str.replace(r"\+.*$", "").str.strptime(
        pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False
    )
    return df.with_columns([
        (2 * np.pi * dt.dt.hour() / 24).sin().cast(pl.Float32).alias("hour_sin"),
        (2 * np.pi * dt.dt.hour() / 24).cos().cast(pl.Float32).alias("hour_cos"),
        (2 * np.pi * dt.dt.weekday() / 5).sin().cast(pl.Float32).alias("dow_sin"),
        (2 * np.pi * dt.dt.weekday() / 5).cos().cast(pl.Float32).alias("dow_cos"),
    ])


def _read_ohlcv(tf_dir: Path, year: int) -> pl.DataFrame:
    matches = list(tf_dir.glob(f"*{year}*.csv"))
    if not matches:
        return pl.DataFrame()
    df = pl.read_csv(matches[0], try_parse_dates=False)
    df = df.rename({c: c.lower() for c in df.columns})
    dt_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if dt_col and dt_col != "datetime":
        df = df.rename({dt_col: "datetime"})
    return df.sort("datetime")


def _resample_features(m1_df: pl.DataFrame, tf_df: pl.DataFrame, prefix: str) -> pl.DataFrame:
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

    # M1 bar floors to its own TF bucket; TF key shifted back 1 period so only
    # fully-closed TF candles are visible (no in-progress candle leakage).
    m1_key = (m1_dt.dt.epoch(time_unit="s") // period_secs * period_secs).alias("_jk")
    tf_key  = (tf_dt.dt.epoch(time_unit="s") // period_secs * period_secs - period_secs).alias("_jk")

    tf_select = tf_df.select(["datetime"] + [c for c in tf_df.columns if c in feat_cols])
    tf_select = tf_select.with_columns(tf_key).drop("datetime")

    m1_joined = m1_df.with_columns(m1_key).join(tf_select, on="_jk", how="left").drop("_jk")
    for col in feat_cols:
        if col in m1_joined.columns:
            m1_joined = m1_joined.with_columns(pl.col(col).forward_fill())
        else:
            m1_joined = m1_joined.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))
    return m1_joined


def build_year_data(year: int, norm_stats: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Build (features, prices) arrays for a single year using saved norm stats."""
    m1_dir  = OHLCV_BASE / "1min"
    m5_dir  = OHLCV_BASE / "5min"
    m15_dir = OHLCV_BASE / "15min"
    h1_dir  = OHLCV_BASE / "1hour"
    h4_dir  = OHLCV_BASE / "4hour"

    m1_matches = list(m1_dir.glob(f"*{year}*.csv"))
    if not m1_matches:
        return None, None

    m1 = pl.read_csv(m1_matches[0], try_parse_dates=False)
    m1 = m1.rename({c: c.lower() for c in m1.columns})
    dt_col = next((c for c in m1.columns if "date" in c or "time" in c), None)
    if dt_col and dt_col != "datetime":
        m1 = m1.rename({dt_col: "datetime"})

    if m1.height < 1000:
        return None, None

    # M1 features
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

    for prefix, tf_dir in [("m5", m5_dir), ("m15", m15_dir), ("h1", h1_dir), ("h4", h4_dir)]:
        tf_raw = _read_ohlcv(tf_dir, year)
        if not tf_raw.is_empty():
            tf_raw = _ohlcv_features(tf_raw, prefix)
        m1 = _resample_features(m1, tf_raw, prefix)

    if "h4_body" in m1.columns:
        m1 = m1.drop("h4_body")

    feature_cols = [
        "return_1", "return_5", "return_20", "range_pct", "body_pct",
        "volume_ratio", "ma_ratio_20", "vol_20", "spread_pct",
        "m5_return", "m5_range", "m5_body", "m5_vol", "m5_ma20",
        "m15_return", "m15_range", "m15_body", "m15_vol", "m15_ma20",
        "h1_return", "h1_range", "h1_body", "h1_vol", "h1_ma20",
        "h4_return", "h4_range", "h4_vol", "h4_ma20",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    feature_cols = [c for c in feature_cols if c in m1.columns]
    m1 = m1.drop_nulls(subset=feature_cols)

    if m1.height < SEQ_LEN + 10:
        return None, None

    features = m1.select(feature_cols).to_numpy().astype(np.float32)

    spread = m1["spread_avg"].to_numpy() if "spread_avg" in m1.columns else np.zeros(len(features), np.float32)
    close_arr = m1["close"].to_numpy()
    bid = (close_arr - spread / 2).astype(np.float32)
    ask = (close_arr + spread / 2).astype(np.float32)
    prices = np.stack([bid, ask], axis=1)

    # Build sequences
    all_feat, all_price = [], []
    n_rows = len(features)
    for i in range(0, n_rows - SEQ_LEN, STRIDE):
        all_feat.append(features[i:i + SEQ_LEN])
        all_price.append(prices[i + SEQ_LEN - 1])

    if not all_feat:
        return None, None

    X = np.stack(all_feat).astype(np.float32)
    P = np.stack(all_price).astype(np.float32)

    # Apply train norm stats
    X = (X - norm_stats["mean"]) / norm_stats["std"]
    X = np.clip(X, -5.0, 5.0)

    return X, P


# ── Episode runner ────────────────────────────────────────────────────

def run_episode(env: ForexTradingEnv, model: PPO):
    obs, _ = env.reset()
    done = False
    step_capitals = [env.capital]
    actions_taken = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        step_capitals.append(env.capital)
        actions_taken.append(int(action))

    return {
        "capital_curve": np.array(step_capitals),
        "final_capital": env.capital,
        "total_trades":  env.total_trades,
        "winning_trades": env.winning,
        "actions": actions_taken,
    }


def compute_metrics(results: list, initial_capital: float, bars_per_day: int = 1440) -> dict:
    final_capitals = [r["final_capital"] for r in results]
    total_trades   = [r["total_trades"]  for r in results]
    winning_trades = [r["winning_trades"] for r in results]

    pnls = [fc - initial_capital for fc in final_capitals]
    total_t = sum(total_trades)
    total_w = sum(winning_trades)
    win_rate = total_w / total_t if total_t > 0 else 0.0

    # ep_len is in environment steps; each step covers STRIDE M1 bars.
    # Convert to actual M1 bars then to trading days.
    ep_len = len(results[0]["capital_curve"]) - 1
    actual_bars = ep_len * STRIDE          # true M1 bars in episode
    trading_days = actual_bars / bars_per_day
    trades_per_day = np.mean(total_trades) / trading_days if trading_days > 0 else 0.0

    max_dds = []
    for r in results:
        curve = r["capital_curve"]
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / peak
        max_dds.append(dd.max())

    ep_returns = np.array(pnls) / initial_capital
    sharpe = 0.0
    if ep_returns.std() > 0:
        episodes_per_year = 252 / trading_days
        sharpe = (ep_returns.mean() / ep_returns.std()) * np.sqrt(episodes_per_year)

    all_actions = np.array([a for r in results for a in r["actions"]])
    total_steps = len(all_actions)

    return {
        "mean_pnl_usd":   float(np.mean(pnls)),
        "std_pnl_usd":    float(np.std(pnls)),
        "mean_pnl_pct":   float(np.mean(ep_returns) * 100),
        "win_rate":       float(win_rate),
        "trades_per_day": float(trades_per_day),
        "max_drawdown":   float(np.mean(max_dds)),
        "sharpe":         float(sharpe),
        "profitable_eps": int(sum(p > 0 for p in pnls)),
        "n_episodes":     len(results),
        "hold_pct":       float((all_actions == 0).sum() / total_steps * 100),
        "buy_pct":        float((all_actions == 1).sum() / total_steps * 100),
        "sell_pct":       float((all_actions == 2).sum() / total_steps * 100),
        "close_pct":      float((all_actions == 3).sum() / total_steps * 100),
    }


# ── Multi-year PnL chart ──────────────────────────────────────────────

def plot_yearly_pnl(year_curves: dict, initial_capital: float, out_path: Path):
    """Plot mean PnL curve per year, all on one chart."""
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle("AQRF Agent — Per-Year Mean PnL Curves (2016–2026)", fontsize=14, fontweight="bold")

    cmap = plt.cm.tab20
    years = sorted(year_curves.keys())
    for i, year in enumerate(years):
        curves = year_curves[year]
        mean_curve = np.mean([(c / initial_capital - 1) * 100 for c in curves], axis=0)
        ax.plot(mean_curve, label=str(year), color=cmap(i / len(years)), linewidth=1.5)

    ax.axhline(0, color="red", linewidth=1.0, linestyle="--", label="Break-even")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Bar (M1 minutes)")
    ax.set_title("Mean Episode Return — Each Year")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("yearly_plot_saved", path=str(out_path))


def plot_bar_summary(rows: list[dict], out_path: Path):
    """Bar chart of mean return % per year."""
    years  = [r["year"] for r in rows]
    returns = [r["mean_pnl_pct"] for r in rows]
    colors  = ["steelblue" if v >= 0 else "tomato" for v in returns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("AQRF Agent — Yearly Performance Summary", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.bar(years, returns, color=colors, edgecolor="white")
    ax1.axhline(0, color="red", linewidth=1.0, linestyle="--")
    ax1.set_title("Mean Episode Return (%) per Year")
    ax1.set_ylabel("Return (%)")
    ax1.set_xlabel("Year")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.grid(axis="y", alpha=0.3)
    for x, v in zip(years, returns):
        ax1.text(x, v + 0.3 * np.sign(v), f"{v:.1f}%", ha="center", fontsize=7)

    ax2 = axes[1]
    win_rates = [r["win_rate"] * 100 for r in rows]
    ax2.bar(years, win_rates, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axhline(50, color="red", linewidth=1.0, linestyle="--", label="50% (random)")
    ax2.set_title("Win Rate (%) per Year")
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_xlabel("Year")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for x, v in zip(years, win_rates):
        ax2.text(x, v + 1, f"{v:.1f}%", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("summary_plot_saved", path=str(out_path))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="checkpoints/rl/best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config()
    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load norm stats from train split
    norm_path = Path("data/processed/norm_stats.npz")
    if not norm_path.exists():
        raise FileNotFoundError("norm_stats.npz not found — run process_data.py first")
    ns = np.load(str(norm_path))
    norm_stats = {"mean": ns["mean"], "std": ns["std"]}

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        for c in ["checkpoints/rl/best/best_model.zip", "checkpoints/rl/ppo_forex_final.zip"]:
            if Path(c).exists():
                model_path = Path(c)
                break
        else:
            raise FileNotFoundError("No model found")
    logger.info("loading_model", path=str(model_path))
    model = PPO.load(str(model_path))

    rows = []
    year_curves = {}   # year → list of capital_curve arrays

    print("\n" + "=" * 80)
    print(f"  AQRF Agent -- Full Historical Evaluation  ({args.episodes} episodes/year)")
    print("=" * 80)
    print(f"  {'Year':<6} {'Return%':>8} {'P&L $':>10} {'WinRate':>8} {'Trd/Day':>8} {'MaxDD%':>7} {'Sharpe':>7} {'Prof/N':>7}")
    print("-" * 80)

    for year in ALL_YEARS:
        logger.info("processing_year", year=year)
        X, P = build_year_data(year, norm_stats)

        if X is None or len(X) < 100:
            logger.warning("insufficient_data", year=year)
            print(f"  {year:<6} {'N/A — insufficient data':>60}")
            continue

        env = ForexTradingEnv(X, P, cfg.rl)
        results = []
        for ep in range(args.episodes):
            r = run_episode(env, model)
            results.append(r)

        m = compute_metrics(results, cfg.rl.initial_capital)
        m["year"] = year
        rows.append(m)

        year_curves[year] = [r["capital_curve"] for r in results]

        print(
            f"  {year:<6} "
            f"{m['mean_pnl_pct']:>7.2f}% "
            f"${m['mean_pnl_usd']:>9.2f} "
            f"{m['win_rate']*100:>7.1f}% "
            f"{m['trades_per_day']:>8.1f} "
            f"{m['max_drawdown']*100:>6.2f}% "
            f"{m['sharpe']:>7.2f} "
            f"{m['profitable_eps']:>3d}/{m['n_episodes']:<3d}"
        )

        del X, P, env, results
        gc.collect()

    print("=" * 80 + "\n")

    if not rows:
        print("No data processed.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = out_dir / "yearly_results.csv"
    header = "year,mean_pnl_usd,std_pnl_usd,mean_pnl_pct,win_rate,trades_per_day,max_drawdown,sharpe,profitable_eps,n_episodes,hold_pct,buy_pct,sell_pct,close_pct"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['year']},{r['mean_pnl_usd']:.2f},{r['std_pnl_usd']:.2f},"
            f"{r['mean_pnl_pct']:.4f},{r['win_rate']:.4f},{r['trades_per_day']:.2f},"
            f"{r['max_drawdown']:.4f},{r['sharpe']:.4f},{r['profitable_eps']},{r['n_episodes']},"
            f"{r['hold_pct']:.2f},{r['buy_pct']:.2f},{r['sell_pct']:.2f},{r['close_pct']:.2f}"
        )
    csv_path.write_text("\n".join(lines))
    logger.info("csv_saved", path=str(csv_path))

    # ── Save Markdown ─────────────────────────────────────────────────
    md_path = out_dir / "yearly_results.md"
    trained_years = set(range(2016, 2022))

    md = ["# AQRF Agent — Full Historical Year-by-Year Evaluation\n"]
    md.append(f"**Model:** `{model_path}`  ")
    md.append(f"**Episodes per year:** {args.episodes}  ")
    md.append(f"**Evaluated:** 2026-05-22  \n")
    md.append("> Years 2016–2021 = TRAINING data (in-sample). Years 2022+ = unseen (out-of-sample).\n")
    md.append("| Year | In/Out | Return% | Mean P&L | Win Rate | Trd/Day | Max DD% | Sharpe | Prof/N |")
    md.append("|------|--------|---------|----------|----------|---------|---------|--------|--------|")
    for r in rows:
        tag = "IN-SAMPLE" if r["year"] in trained_years else "**OUT**"
        md.append(
            f"| {r['year']} | {tag} "
            f"| {r['mean_pnl_pct']:+.2f}% "
            f"| ${r['mean_pnl_usd']:,.2f} ±${r['std_pnl_usd']:,.0f} "
            f"| {r['win_rate']*100:.1f}% "
            f"| {r['trades_per_day']:.1f} "
            f"| {r['max_drawdown']*100:.2f}% "
            f"| {r['sharpe']:.2f} "
            f"| {r['profitable_eps']}/{r['n_episodes']} |"
        )

    # Aggregate out-of-sample
    oos = [r for r in rows if r["year"] not in trained_years]
    if oos:
        avg_ret    = np.mean([r["mean_pnl_pct"] for r in oos])
        avg_wr     = np.mean([r["win_rate"] for r in oos]) * 100
        avg_dd     = np.mean([r["max_drawdown"] for r in oos]) * 100
        avg_sharpe = np.mean([r["sharpe"] for r in oos])
        avg_pnl    = np.mean([r["mean_pnl_usd"] for r in oos])
        total_prof = sum(r["profitable_eps"] for r in oos)
        total_eps  = sum(r["n_episodes"] for r in oos)
        md.append(f"| **OOS Avg** | — | **{avg_ret:+.2f}%** | **${avg_pnl:,.2f}** | **{avg_wr:.1f}%** | — | **{avg_dd:.2f}%** | **{avg_sharpe:.2f}** | **{total_prof}/{total_eps}** |")

    md.append("\n## Action Distribution per Year\n")
    md.append("| Year | Hold% | Buy% | Sell% | Close% |")
    md.append("|------|-------|------|-------|--------|")
    for r in rows:
        md.append(f"| {r['year']} | {r['hold_pct']:.1f}% | {r['buy_pct']:.1f}% | {r['sell_pct']:.1f}% | {r['close_pct']:.1f}% |")

    md.append("\n## Charts\n")
    md.append("- `eval_results/yearly_pnl_curves.png` — mean PnL curve per year overlay")
    md.append("- `eval_results/yearly_summary_bars.png` — return% and win rate bar charts")

    md_path.write_text("\n".join(md))
    logger.info("markdown_saved", path=str(md_path))

    # ── Charts ────────────────────────────────────────────────────────
    plot_yearly_pnl(year_curves, cfg.rl.initial_capital, out_dir / "yearly_pnl_curves.png")
    plot_bar_summary(rows, out_dir / "yearly_summary_bars.png")

    print(f"  Results saved to:")
    print(f"    {csv_path}")
    print(f"    {md_path}")
    print(f"    eval_results/yearly_pnl_curves.png")
    print(f"    eval_results/yearly_summary_bars.png\n")


if __name__ == "__main__":
    main()
