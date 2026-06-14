"""Stress Testing & Chaos Simulation entry point (Phase 3 — Module 5).

Runs Monte Carlo hostile-market scenarios against the PPO agent and
reports ruin probability, CVaR, Sharpe degradation, and worst drawdown.

Scenarios:
  baseline            Clean market (reference)
  execution_chaos     Extreme spread/slippage/delay randomization
  flash_crash         Synthetic 50-200 pip crashes with partial recovery
  spread_spike        Sudden 5-30 pip spread spikes
  volatility          2x-8x volatility explosions (sustained bursts)
  combined            All hostile effects simultaneously
  all                 Run all scenarios (default)

Usage:
    python stress_test_rl.py
    python stress_test_rl.py --model checkpoints/rl/phase3/best/best_model.zip
    python stress_test_rl.py --scenario flash_crash
    python stress_test_rl.py --scenario all --trials 100 --output stress_reports
"""
import argparse
from pathlib import Path

import numpy as np

from src.backtest.stress_tester import RLStressTester
from src.utils.config import load_config
from src.utils.logging import setup_logging

import structlog
logger = structlog.get_logger(__name__)

DEFAULT_MODEL  = "checkpoints/rl/phase3/best/best_model.zip"
DEFAULT_DATA   = "data/processed"
DEFAULT_OUTPUT = "stress_reports"

SCENARIO_NAMES = ["baseline", "execution_chaos", "flash_crash",
                  "spread_spike", "volatility", "combined", "all"]


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="RL agent stress testing and chaos simulation")
    parser.add_argument("--model",     type=str, default=DEFAULT_MODEL,
                        help="Path to PPO model ZIP")
    parser.add_argument("--data-dir",  type=str, default=DEFAULT_DATA,
                        help="Directory with test_features.npy and test_prices.npy")
    parser.add_argument("--scenario",  type=str, default="all",
                        choices=SCENARIO_NAMES,
                        help="Which scenario(s) to run (default: all)")
    parser.add_argument("--trials",    type=int, default=None,
                        help="Monte Carlo trials per scenario (default: from config, 200)")
    parser.add_argument("--episodes",  type=int, default=None,
                        help="Episodes per trial (default: from config, 20)")
    parser.add_argument("--output",    type=str, default=DEFAULT_OUTPUT,
                        help="Output directory for reports")
    parser.add_argument("--prefix",    type=str, default="stress",
                        help="Filename prefix for saved report")
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

    # Load test data
    data_dir = Path(args.data_dir)
    fx = data_dir / "test_features.npy"
    fp = data_dir / "test_prices.npy"
    if not fx.exists() or not fp.exists():
        raise SystemExit(f"Missing test data in {data_dir}. Run: python process_data.py")

    X = np.load(str(fx), mmap_mode="r")
    P = np.load(str(fp), mmap_mode="r")
    logger.info("test_data_loaded", bars=len(X))

    # Override trial counts if provided
    stress_cfg = cfg.stress.model_copy(deep=True)
    if args.trials is not None:
        stress_cfg.n_trials = args.trials
    if args.episodes is not None:
        stress_cfg.n_episodes_per_trial = args.episodes

    logger.info(
        "stress_test_config",
        scenario=args.scenario,
        n_trials=stress_cfg.n_trials,
        n_episodes=stress_cfg.n_episodes_per_trial,
        model=str(model_path),
    )

    tester = RLStressTester(
        model_path=model_path,
        features=X,
        prices=P,
        rl_config=cfg.rl,
        stress_config=stress_cfg,
        output_dir=Path(args.output),
    )

    # Run selected scenarios
    if args.scenario == "all":
        tester.run_baseline()
        tester.run_execution_chaos()
        tester.run_flash_crash()
        tester.run_spread_spike()
        tester.run_volatility_explosion()
        tester.run_combined_chaos()
    elif args.scenario == "baseline":
        tester.run_baseline()
    elif args.scenario == "execution_chaos":
        tester.run_execution_chaos()
    elif args.scenario == "flash_crash":
        tester.run_flash_crash()
    elif args.scenario == "spread_spike":
        tester.run_spread_spike()
    elif args.scenario == "volatility":
        tester.run_volatility_explosion()
    elif args.scenario == "combined":
        tester.run_combined_chaos()

    # Generate and save report
    report = tester.generate_report()
    save_path = tester.save_report(prefix=args.prefix)

    print()
    print(report)
    print(f"Report saved to: {save_path}")


if __name__ == "__main__":
    main()
