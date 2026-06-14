"""RL Paper Trading entry point (Phase 3 — Module 4).

Runs the PPO agent in paper trading mode on held-out test data,
measuring expected vs actual fills to quantify the simulation-to-reality gap.

Usage:
    python paper_trade_rl.py
    python paper_trade_rl.py --model checkpoints/rl/phase3/best/best_model.zip
    python paper_trade_rl.py --split val
    python paper_trade_rl.py --start-bar 0 --end-bar 50000
    python paper_trade_rl.py --capital 10000 --log-dir paper_trades_rl
"""
import argparse
from pathlib import Path

import numpy as np

from src.execution.rl_paper_trader import RLPaperTrader
from src.utils.config import load_config
from src.utils.logging import setup_logging

import structlog
logger = structlog.get_logger(__name__)

DEFAULT_MODEL  = "checkpoints/rl/phase3/best/best_model.zip"
DEFAULT_DATA   = "data/processed"
DEFAULT_SPLIT  = "test"
DEFAULT_LOG    = "paper_trades_rl"


def load_split(data_dir: Path, split: str):
    fx = data_dir / f"{split}_features.npy"
    fp = data_dir / f"{split}_prices.npy"
    if not fx.exists() or not fp.exists():
        raise FileNotFoundError(
            f"Missing {split} data in {data_dir}. Run: python process_data.py"
        )
    X = np.load(str(fx), mmap_mode="r")
    P = np.load(str(fp), mmap_mode="r")
    logger.info("data_loaded", split=split, bars=len(X))
    return X, P


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="RL paper trading with fill quality analysis")
    parser.add_argument("--model",     type=str, default=DEFAULT_MODEL,
                        help="Path to PPO model ZIP")
    parser.add_argument("--data-dir",  type=str, default=DEFAULT_DATA,
                        help="Directory with feature/price .npy files")
    parser.add_argument("--split",     type=str, default=DEFAULT_SPLIT,
                        choices=["train", "val", "test"],
                        help="Data split to trade on")
    parser.add_argument("--start-bar", type=int, default=0,
                        help="First bar index (default: 0)")
    parser.add_argument("--end-bar",   type=int, default=None,
                        help="Last bar index exclusive (default: all)")
    parser.add_argument("--capital",   type=float, default=10_000.0,
                        help="Initial capital (default: 10000)")
    parser.add_argument("--log-dir",   type=str, default=DEFAULT_LOG,
                        help="Directory for trade/fill CSV logs")
    parser.add_argument("--prefix",    type=str, default="rl_paper",
                        help="Filename prefix for saved CSVs")
    parser.add_argument("--config",    type=str, default=None,
                        help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            f"Run 'python train_rl.py --phase3' first, or pass --model <path>"
        )

    X, P = load_split(Path(args.data_dir), args.split)

    trader = RLPaperTrader(
        model_path=model_path,
        config=cfg.rl,
        risk_config=cfg.risk,
        initial_capital=args.capital,
        log_dir=Path(args.log_dir),
    )

    logger.info(
        "paper_trade_start",
        split=args.split,
        start_bar=args.start_bar,
        end_bar=args.end_bar or len(X),
        capital=args.capital,
    )

    stats = trader.run_simulation(
        X, P,
        start_bar=args.start_bar,
        end_bar=args.end_bar,
    )

    fill_analysis = trader.get_fill_analysis()
    trader.save_logs(prefix=args.prefix)

    # Print portfolio summary
    print("\n" + "=" * 55)
    print("RL Paper Trading Results")
    print("=" * 55)
    print(f"  Split:            {args.split}")
    print(f"  Bars traded:      {(args.end_bar or len(X)) - args.start_bar:,}")
    print(f"  Capital:          ${stats['capital']:>10,.2f}")
    print(f"  Total return:     {stats['total_return_pct']:>+8.2f}%")
    print(f"  Peak capital:     ${stats['peak_capital']:>10,.2f}")
    print(f"  Max drawdown:     {stats['drawdown_pct']:>8.2f}%")
    print(f"  Total trades:     {stats['total_trades']}")
    print(f"  Win rate:         {stats['win_rate_pct']:>8.1f}%")
    print(f"  Avg trade PnL:    ${stats['avg_trade_pnl']:>+9.4f}")
    print(f"  SL hits:          {stats['sl_hits']}")
    print(f"  TP hits:          {stats['tp_hits']}")
    print()
    print("Fill Quality Analysis")
    print("-" * 55)
    if "error" not in fill_analysis:
        print(f"  Total fills:      {fill_analysis['n_fills']}")
        print(f"  Mean slippage:    {fill_analysis['mean_slip_pips']:.3f} pips")
        print(f"  Max slippage:     {fill_analysis['max_slip_pips']:.3f} pips")
        print(f"  Total cost:       {fill_analysis['total_cost_pips']:.2f} pips")
        print(f"  Mean confidence:  {fill_analysis['mean_confidence']:.3f}")
        print(f"  Min confidence:   {fill_analysis['min_confidence']:.3f}")
        print(f"  Mean entropy:     {fill_analysis['mean_entropy']:.4f}")
    else:
        print(f"  {fill_analysis['error']}")
    print("=" * 55)
    print(f"  Logs saved to:  {args.log_dir}/{args.prefix}_*.csv")
    print()


if __name__ == "__main__":
    main()
