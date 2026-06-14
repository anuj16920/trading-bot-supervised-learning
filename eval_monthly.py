"""Monthly returns for the RL model on test data (2023-2024).

Uses data/processed/test_features.npy + test_prices.npy + test_timestamps.npy.
Splits by real calendar month using timestamps (not equal-size slices).
Runs one episode per month-slice, reports per-trade P&L, win rate, SL/TP counts.

Usage:
    python eval_monthly.py
    python eval_monthly.py --model PATH/model.zip

Saves:
    eval_results/monthly_returns_{tag}.csv
"""
import argparse
import sys
from pathlib import Path
from calendar import month_abbr

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.rl.environment import ForexTradingEnv
from src.utils.config import load_config, RLConfig
from src.utils.logging import setup_logging
import structlog

logger = structlog.get_logger(__name__)

# Test data covers 2023-01-01 to 2024-12-31 = 24 months
TEST_MONTHS = [
    "2023-Jan", "2023-Feb", "2023-Mar", "2023-Apr", "2023-May", "2023-Jun",
    "2023-Jul", "2023-Aug", "2023-Sep", "2023-Oct", "2023-Nov", "2023-Dec",
    "2024-Jan", "2024-Feb", "2024-Mar", "2024-Apr", "2024-May", "2024-Jun",
    "2024-Jul", "2024-Aug", "2024-Sep", "2024-Oct", "2024-Nov", "2024-Dec",
]


def run_month_episode(features_slice: np.ndarray, prices_slice: np.ndarray,
                      cfg: RLConfig, model: PPO):
    """Run one episode on a month slice, return per-trade records."""
    env = ForexTradingEnv(features_slice, prices_slice, cfg)

    trades = []
    original_record = env._record_trade

    def patched_record(pnl, sl_hit=False, tp_hit=False):
        trades.append({"pnl": pnl, "sl": sl_hit, "tp": tp_hit, "win": pnl > 0})
        original_record(pnl, sl_hit=sl_hit, tp_hit=tp_hit)

    env._record_trade = patched_record

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    return {
        "final_capital": env.capital,
        "trades":        trades,
        "total_trades":  env.total_trades,
        "sl_hits":       env.sl_hits,
        "tp_hits":       env.tp_hits,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="checkpoints/rl/phase3/20260529_145135/best/best_model.zip")
    parser.add_argument("--tag", type=str, default="v10",
                        help="Version tag used in output filenames (e.g. v9)")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = Path("data/processed")
    out_dir  = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(str(data_dir / "test_features.npy"), mmap_mode="r")
    prices   = np.load(str(data_dir / "test_prices.npy"),   mmap_mode="r")
    n_bars   = len(features)
    logger.info("data_loaded", shape=features.shape)

    # Load timestamps for true calendar-month splitting
    ts_path = data_dir / "test_timestamps.npy"
    if ts_path.exists():
        timestamps = np.load(str(ts_path))
        use_timestamps = True
        logger.info("timestamps_loaded", n=len(timestamps))
    else:
        timestamps = None
        use_timestamps = False
        logger.warning("timestamps_not_found_falling_back_to_equal_split")

    model_path = Path(args.model)
    logger.info("loading_model", path=str(model_path))
    model = PPO.load(str(model_path))

    # Build month slices — true calendar months if timestamps available
    month_slices = []  # list of (label, start_idx, end_idx)
    if use_timestamps:
        import datetime
        for month_label in TEST_MONTHS:
            year  = int(month_label[:4])
            month = list(["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]).index(month_label[5:]) + 1
            # Unix timestamp for first second of this month and next month
            t_start = int(datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc).timestamp())
            if month == 12:
                t_end = int(datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
            else:
                t_end = int(datetime.datetime(year, month + 1, 1, tzinfo=datetime.timezone.utc).timestamp())
            mask = (timestamps >= t_start) & (timestamps < t_end)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            month_slices.append((month_label, int(idxs[0]), int(idxs[-1]) + 1))
        logger.info("calendar_month_slices", n=len(month_slices))
    else:
        # Fallback: equal-size slices
        bars_per_month = n_bars // len(TEST_MONTHS)
        for i, month_label in enumerate(TEST_MONTHS):
            start = i * bars_per_month
            end   = start + bars_per_month if i < len(TEST_MONTHS) - 1 else n_bars
            month_slices.append((month_label, start, end))

    # Results storage
    results = {}

    print()
    print("=" * 72)
    print(f"  {'Month':<12}  {'P&L ($)':>9}  {'Ret%':>6}  {'Trades':>7}  {'Win%':>6}  {'SL':>4}  {'TP':>4}  {'Cumul$':>9}")
    print("-" * 72)

    running_capital = cfg.rl.initial_capital
    total_pnl = 0.0
    total_trades = total_wins = total_sl = total_tp = 0

    for month_label, start, end in month_slices:
        feat_slice  = np.array(features[start:end])
        price_slice = np.array(prices[start:end])

        if len(feat_slice) < 200:
            logger.warning("slice_too_small", month=month_label, bars=len(feat_slice))
            continue

        # Override episode_bars so the agent runs the full month slice
        month_cfg = cfg.rl.model_copy(deep=True)
        month_cfg.episode_bars = len(feat_slice) - 1

        r = run_month_episode(feat_slice, price_slice, month_cfg, model)

        pnl    = r["final_capital"] - cfg.rl.initial_capital
        trades = r["total_trades"]
        sl     = r["sl_hits"]
        tp     = r["tp_hits"]
        wins   = sum(1 for t in r["trades"] if t["win"])
        win_pct = (wins / trades * 100) if trades > 0 else 0.0
        ret_pct = (pnl / running_capital) * 100
        running_capital += pnl

        total_pnl    += pnl
        total_trades += trades
        total_wins   += wins
        total_sl     += sl
        total_tp     += tp

        marker = " +" if pnl > 0 else (" -" if pnl < 0 else "  ")
        print(f"  {month_label:<12}  {pnl:>+9.2f}  {ret_pct:>5.2f}%  {trades:>7}  {win_pct:>5.1f}%  {sl:>4}  {tp:>4}  {running_capital:>9.2f}{marker}")

        results[month_label] = {
            "pnl": pnl, "ret_pct": ret_pct, "trades": trades,
            "win_pct": win_pct, "sl": sl, "tp": tp, "capital": running_capital,
        }

        logger.info("month_done", month=month_label, pnl=round(pnl, 2),
                    trades=trades, win_pct=round(win_pct, 1))

    print("-" * 72)
    total_ret = (running_capital - cfg.rl.initial_capital) / cfg.rl.initial_capital * 100
    total_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    print(f"  {'TOTAL':<12}  {total_pnl:>+9.2f}  {total_ret:>5.2f}%  {total_trades:>7}  {total_win_pct:>5.1f}%  {total_sl:>4}  {total_tp:>4}  {running_capital:>9.2f}")
    print("=" * 72)

    # Summary
    profitable = sum(1 for v in results.values() if v["pnl"] > 0)
    print(f"\n  Profitable months : {profitable} / {len(results)}")
    print(f"  Avg trades/month  : {total_trades / max(len(results), 1):.1f}")
    print(f"  Overall win rate  : {total_win_pct:.1f}%")
    print(f"  SL hits           : {total_sl}  ({total_sl/max(total_trades,1)*100:.1f}%)")
    print(f"  TP hits           : {total_tp}  ({total_tp/max(total_trades,1)*100:.1f}%)")
    print(f"  Net P&L (2y)      : ${total_pnl:+.2f}  ({total_ret:+.2f}% on $10,000)")
    print()

    # Save CSV
    csv_path = out_dir / f"monthly_returns_{args.tag}.csv"
    lines = ["month,pnl,ret_pct,trades,win_pct,sl,tp,capital"]
    for m, v in results.items():
        lines.append(f"{m},{v['pnl']:.2f},{v['ret_pct']:.4f},{v['trades']},{v['win_pct']:.2f},{v['sl']},{v['tp']},{v['capital']:.2f}")
    csv_path.write_text("\n".join(lines))

    # Bar chart
    months_list = list(results.keys())
    pnls_list   = [results[m]["pnl"] for m in months_list]
    colors = ["steelblue" if p >= 0 else "tomato" for p in pnls_list]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"AQRF {args.tag} — Monthly P&L on Test Data (2023-2024)", fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(range(len(months_list)), pnls_list, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(months_list)))
    ax.set_xticklabels(months_list, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("P&L ($)")
    ax.set_title("Monthly P&L")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    cum = np.cumsum([0] + pnls_list) + cfg.rl.initial_capital
    ax2.plot(range(len(months_list) + 1), cum, color="navy", linewidth=2)
    ax2.fill_between(range(len(months_list) + 1), cfg.rl.initial_capital, cum,
                     where=cum >= cfg.rl.initial_capital, alpha=0.2, color="steelblue")
    ax2.fill_between(range(len(months_list) + 1), cfg.rl.initial_capital, cum,
                     where=cum < cfg.rl.initial_capital, alpha=0.2, color="tomato")
    ax2.axhline(cfg.rl.initial_capital, color="red", linewidth=1.0, linestyle="--", label="Start $10,000")
    ax2.set_xticks(range(len(months_list) + 1))
    ax2.set_xticklabels([""] + months_list, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Account Value ($)")
    ax2.set_title("Cumulative Equity Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = out_dir / f"monthly_pnl_{args.tag}.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close()

    # Save TXT results
    txt_path = out_dir / f"{args.tag}_monthly_results.txt"
    txt_lines = [
        f"AQRF {args.tag} - Monthly Returns on Test Data (2023-2024)",
        f"Model: {args.model}",
        f"Evaluated: 2026-06-01",
        f"Data: data/processed/test_features.npy ({n_bars:,} bars, Jan 2023 - Dec 2024)",
        f"Episode config: True calendar months (timestamp-split), deterministic policy",
        "Friction: spread=0.5 pips, slippage=0.3 pips, delay=1 bar (eval mode)",
        "R:R ratio: SL=10 pips / TP=20 pips (1:2)",
        "=" * 72,
        f"  {'Month':<12}  {'P&L ($)':>9}  {'Ret%':>6}  {'Trades':>7}  {'Win%':>6}  {'SL':>4}  {'TP':>4}  {'Cumul$':>9}",
        "-" * 72,
    ]
    rc = cfg.rl.initial_capital
    for m, v in results.items():
        rc += v["pnl"] if m != list(results.keys())[0] else 0
        txt_lines.append(
            f"  {m:<12}  {v['pnl']:>+9.2f}  {v['ret_pct']:>+6.2f}%  {v['trades']:>6}  {v['win_pct']:>5.1f}%  {v['sl']:>4}  {v['tp']:>4}  {v['capital']:>9.2f}"
        )
    txt_lines += [
        "-" * 72,
        f"  {'TOTAL':<12}  {total_pnl:>+9.2f}  {total_ret:>+6.2f}%  {total_trades:>6}  {total_win_pct:>5.1f}%  {total_sl:>4}  {total_tp:>4}  {running_capital:>9.2f}",
        "=" * 72,
        "",
        "SUMMARY",
        "-------",
        f"  Profitable months : {profitable} / {len(results)}  ({profitable/len(results)*100:.1f}%)",
        f"  Best month        : {max(results, key=lambda m: results[m]['pnl'])}  ${max(v['pnl'] for v in results.values()):+.2f}",
        f"  Worst month       : {min(results, key=lambda m: results[m]['pnl'])}  ${min(v['pnl'] for v in results.values()):+.2f}",
        f"  Avg trades/month  : {total_trades / max(len(results), 1):.1f}",
        f"  Overall win rate  : {total_win_pct:.1f}%  (break-even at 1:2 R:R = 33.3%)",
        f"  SL hits           : {total_sl}  ({total_sl/max(total_trades,1)*100:.1f}% of trades)",
        f"  TP hits           : {total_tp}  ({total_tp/max(total_trades,1)*100:.1f}% of trades)",
        f"  Net P&L (2 years) : ${total_pnl:+.2f}  ({total_ret:+.2f}% on $10,000)",
    ]
    txt_path.write_text("\n".join(txt_lines))

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {chart_path}")
    print(f"  Saved: {txt_path}")


if __name__ == "__main__":
    main()
