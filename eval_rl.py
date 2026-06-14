"""Evaluate the trained PPO agent on unseen test data (2023-2024).

Usage:
    python eval_rl.py                          # uses best checkpoint
    python eval_rl.py --model PATH/model.zip   # specific checkpoint
    python eval_rl.py --split val              # run on val instead of test

Reports:
    - Total pips gained/lost
    - Win rate
    - Max drawdown
    - Trades per day
    - Sharpe ratio (annualised)
    - PnL curve saved to eval_results/pnl_curve.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO

from src.rl.environment import ForexTradingEnv
from src.utils.config import load_config
from src.utils.logging import setup_logging
import structlog

logger = structlog.get_logger(__name__)


def run_episode(env: ForexTradingEnv, model: PPO, deterministic: bool = True):
    """Run one full episode and collect step-level data."""
    obs, _ = env.reset()
    done = False
    step_capitals = [env.capital]
    trade_pnls = []
    actions_taken = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        step_capitals.append(env.capital)
        actions_taken.append(int(action))

    # Collect closed trade P&Ls from env state
    return {
        "capital_curve": np.array(step_capitals),
        "final_capital": env.capital,
        "total_trades": env.total_trades,
        "winning_trades": env.winning,
        "actions": actions_taken,
    }


def evaluate(model: PPO, features: np.ndarray, prices: np.ndarray, cfg, n_episodes: int = 50):
    """Run n_episodes random episodes across the dataset."""
    results = []
    env = ForexTradingEnv(features, prices, cfg.rl)

    for ep in range(n_episodes):
        r = run_episode(env, model)
        results.append(r)
        pnl = round(r["final_capital"] - 10000.0, 2)
        logger.info("eval_episode", episode=ep + 1, n=n_episodes,
                    pnl=pnl, trades=r["total_trades"], wins=r["winning_trades"])

    return results


def compute_metrics(results: list, initial_capital: float, bars_per_day: int = 1440):
    """Compute aggregate trading metrics across all episodes."""
    final_capitals = [r["final_capital"] for r in results]
    total_trades   = [r["total_trades"]  for r in results]
    winning_trades = [r["winning_trades"] for r in results]

    # Per-episode pnl in dollar terms
    pnls = [fc - initial_capital for fc in final_capitals]

    # Win rate
    total_t = sum(total_trades)
    total_w = sum(winning_trades)
    win_rate = total_w / total_t if total_t > 0 else 0.0

    # Episode length in bars
    ep_len = len(results[0]["capital_curve"]) - 1

    # Trades per day
    trades_per_day = np.mean(total_trades) / (ep_len / bars_per_day)

    # Max drawdown across all episodes
    max_dds = []
    for r in results:
        curve = r["capital_curve"]
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / peak
        max_dds.append(dd.max())
    max_drawdown = np.mean(max_dds)

    # Sharpe (annualised, from per-episode returns)
    ep_returns = np.array(pnls) / initial_capital
    sharpe = 0.0
    if ep_returns.std() > 0:
        # episodes are ~2 days, 252 trading days/year → scale factor
        episodes_per_year = 252 / (ep_len / bars_per_day)
        sharpe = (ep_returns.mean() / ep_returns.std()) * np.sqrt(episodes_per_year)

    # Action distribution
    all_actions = []
    for r in results:
        all_actions.extend(r["actions"])
    all_actions = np.array(all_actions)
    action_counts = {
        "hold": int((all_actions == 0).sum()),
        "buy":  int((all_actions == 1).sum()),
        "sell": int((all_actions == 2).sum()),
        "close": int((all_actions == 3).sum()),
    }

    return {
        "mean_pnl_usd":    float(np.mean(pnls)),
        "std_pnl_usd":     float(np.std(pnls)),
        "mean_pnl_pct":    float(np.mean(ep_returns) * 100),
        "win_rate":        float(win_rate),
        "trades_per_day":  float(trades_per_day),
        "max_drawdown":    float(max_drawdown),
        "sharpe":          float(sharpe),
        "action_dist":     action_counts,
        "n_episodes":      len(results),
        "profitable_eps":  int(sum(p > 0 for p in pnls)),
    }


def plot_pnl(results: list, initial_capital: float, out_path: Path):
    """Plot PnL curves for all episodes + mean curve."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("AQRF Agent — Test Set Evaluation (2023-2024)", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    all_curves = []
    for r in results:
        curve = (r["capital_curve"] / initial_capital - 1) * 100  # % return
        all_curves.append(curve)
        ax1.plot(curve, alpha=0.25, linewidth=0.8, color="steelblue")

    mean_curve = np.mean(all_curves, axis=0)
    ax1.plot(mean_curve, color="navy", linewidth=2.0, label="Mean episode")
    ax1.axhline(0, color="red", linewidth=1.0, linestyle="--", label="Break-even")
    ax1.set_ylabel("Return (%)")
    ax1.set_xlabel("Bar (M1 minutes)")
    ax1.set_title("PnL Curves — All Episodes")
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.grid(alpha=0.3)

    # Distribution of final returns
    ax2 = axes[1]
    final_rets = [(r["final_capital"] / initial_capital - 1) * 100 for r in results]
    ax2.hist(final_rets, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Break-even")
    ax2.axvline(np.mean(final_rets), color="navy", linewidth=1.5, linestyle="-",
                label=f"Mean = {np.mean(final_rets):.2f}%")
    ax2.set_xlabel("Episode Return (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Episode Returns")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("plot_saved", path=str(out_path))


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="checkpoints/rl/best/best_model.zip",
                        help="Path to trained model ZIP")
    parser.add_argument("--split",  type=str, default="test", choices=["val", "test"],
                        help="Data split to evaluate on")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of random episodes to run")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy (no sampling)")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = Path("data/processed")

    # ── Load data ─────────────────────────────────────────────────────
    feat_path  = data_dir / f"{args.split}_features.npy"
    price_path = data_dir / f"{args.split}_prices.npy"

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Run process_data.py first.")

    features = np.load(str(feat_path), mmap_mode="r")
    prices   = np.load(str(price_path), mmap_mode="r")
    logger.info("data_loaded", split=args.split, shape=features.shape)

    # ── Load model ────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        # Try alternative locations
        for candidate in [
            "checkpoints/rl/best/best_model.zip",
            "checkpoints/rl/ppo_forex_final.zip",
        ]:
            if Path(candidate).exists():
                model_path = Path(candidate)
                break
        else:
            raise FileNotFoundError(f"No model found at {args.model}")

    logger.info("loading_model", path=str(model_path))
    model = PPO.load(str(model_path))

    # ── Evaluate ──────────────────────────────────────────────────────
    logger.info("evaluation_start", episodes=args.episodes, split=args.split)
    results = evaluate(model, features, prices, cfg, n_episodes=args.episodes)

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(results, cfg.rl.initial_capital)

    print("\n" + "=" * 52)
    print(f"  AQRF Agent - {args.split.upper()} SET RESULTS ({args.episodes} episodes)")
    print("=" * 52)
    print(f"  Mean P&L per episode : ${metrics['mean_pnl_usd']:>10.2f}  ±${metrics['std_pnl_usd']:.2f}")
    print(f"  Mean return          : {metrics['mean_pnl_pct']:>10.2f}%")
    print(f"  Win rate             : {metrics['win_rate']*100:>10.1f}%")
    print(f"  Profitable episodes  : {metrics['profitable_eps']:>10d} / {metrics['n_episodes']}")
    print(f"  Trades per day       : {metrics['trades_per_day']:>10.1f}")
    print(f"  Max drawdown (mean)  : {metrics['max_drawdown']*100:>10.2f}%")
    print(f"  Sharpe ratio (ann.)  : {metrics['sharpe']:>10.2f}")
    print("-" * 52)
    ad = metrics["action_dist"]
    total_steps = sum(ad.values())
    print(f"  Action distribution  :")
    print(f"    Hold  : {ad['hold']:>7d}  ({ad['hold']/total_steps*100:.1f}%)")
    print(f"    Buy   : {ad['buy']:>7d}  ({ad['buy']/total_steps*100:.1f}%)")
    print(f"    Sell  : {ad['sell']:>7d}  ({ad['sell']/total_steps*100:.1f}%)")
    print(f"    Close : {ad['close']:>7d}  ({ad['close']/total_steps*100:.1f}%)")
    print("=" * 52 + "\n")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_path = Path("eval_results") / f"pnl_curve_{args.split}.png"
    plot_pnl(results, cfg.rl.initial_capital, plot_path)
    print(f"  PnL chart saved -> {plot_path}\n")


if __name__ == "__main__":
    main()
