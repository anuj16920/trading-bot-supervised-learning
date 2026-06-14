"""Walk-Forward Adaptive Retraining entry point (Phase 3 — Module 3).

Fine-tunes the Phase 3 PPO model on rolling 6-month windows, chaining
each window's checkpoint into the next so the policy stays adapted to
evolving market conditions.

Usage:
    python train_rl_walkforward.py
    python train_rl_walkforward.py --model checkpoints/rl/phase3/best/best_model.zip
    python train_rl_walkforward.py --vecnorm checkpoints/rl/phase3/vecnormalize_final.pkl
    python train_rl_walkforward.py --window-months 3 --finetune-steps 250000
    python train_rl_walkforward.py --output-dir checkpoints/rl/walkforward_custom
"""
import argparse
import json
from pathlib import Path

from src.rl.walk_forward import RLWalkForwardTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logging

import structlog
logger = structlog.get_logger(__name__)

DEFAULT_MODEL   = "checkpoints/rl/phase3/best/best_model.zip"
DEFAULT_VECNORM = "checkpoints/rl/phase3/vecnormalize_final.pkl"
DEFAULT_OUTPUT  = "checkpoints/rl/walkforward"
DEFAULT_DATA    = "data/processed"


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Walk-forward fine-tuning for PPO agent")
    parser.add_argument("--model",          type=str, default=DEFAULT_MODEL,
                        help="Path to base PPO model ZIP")
    parser.add_argument("--vecnorm",        type=str, default=DEFAULT_VECNORM,
                        help="Path to VecNormalize PKL from base training")
    parser.add_argument("--data-dir",       type=str, default=DEFAULT_DATA,
                        help="Directory with train_features.npy and train_prices.npy")
    parser.add_argument("--output-dir",     type=str, default=DEFAULT_OUTPUT,
                        help="Where per-window checkpoints are saved")
    parser.add_argument("--window-months",  type=int, default=None,
                        help="Size of each fine-tune window (default: from config, 6)")
    parser.add_argument("--finetune-steps", type=int, default=None,
                        help="Training steps per window (default: from config, 500000)")
    parser.add_argument("--config",         type=str, default=None,
                        help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    model_path   = Path(args.model)
    vecnorm_path = Path(args.vecnorm)

    if not model_path.exists():
        logger.error("model_not_found", path=str(model_path))
        raise SystemExit(
            f"Model not found: {model_path}\n"
            f"Run 'python train_rl.py --phase3' first, or pass --model <path>"
        )

    logger.info(
        "walkforward_config",
        model=str(model_path),
        vecnorm_exists=vecnorm_path.exists(),
        window_months=args.window_months or cfg.rl.wf_window_months,
        finetune_steps=args.finetune_steps or cfg.rl.wf_finetune_timesteps,
    )

    trainer = RLWalkForwardTrainer(
        base_model_path=model_path,
        base_vecnorm_path=vecnorm_path,
        data_dir=Path(args.data_dir),
        config=cfg.rl,
        output_dir=Path(args.output_dir),
        window_months=args.window_months,
        finetune_timesteps=args.finetune_steps,
    )

    results = trainer.run()

    # Save results summary
    out_dir = Path(args.output_dir)
    summary_path = out_dir / "walkforward_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("results_saved", path=str(summary_path))

    # Print summary table
    print("\n" + "=" * 60)
    print("Walk-Forward Results Summary")
    print("=" * 60)
    print(f"{'Window':<8} {'Return%':<12} {'Win Rate':<12} {'Train Bars':<12} {'Test Bars'}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['window_idx']:<8} "
            f"{r['mean_return_pct']:>+8.2f}%   "
            f"{r['win_rate']*100:>6.1f}%     "
            f"{r['train_bars']:<12,} "
            f"{r['test_bars']:,}"
        )
    print("=" * 60)

    if results:
        avg_ret = sum(r["mean_return_pct"] for r in results) / len(results)
        avg_wr  = sum(r["win_rate"] for r in results) / len(results)
        print(f"  Average return: {avg_ret:+.2f}%  |  Average win rate: {avg_wr*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
