"""Train the AQRF RL trading agent.

Single entry point. No supervised pre-training step.
The PPO agent with a TCN feature extractor learns directly from market data.

Usage:
    python train_rl.py                   # full training (Phase 2)
    python train_rl.py --phase3          # Phase 3 with domain randomization (8M steps)
    python train_rl.py --dummy           # quick smoke-test with random data
    python train_rl.py --resume PATH     # resume from checkpoint
    python train_rl.py --phase3 --resume PATH  # resume Phase 3 training
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from src.rl.environment import ForexTradingEnv
from src.rl.agent import make_env, create_ppo_agent
from src.rl.callbacks import LoggingCallback, ConfidenceLoggingCallback, StopTrainingOnNoModelImprovement
from src.utils.config import AQRFConfig, load_config
from src.utils.logging import setup_logging

import structlog
logger = structlog.get_logger(__name__)


def load_data(data_dir: Path):
    """Load processed feature/price arrays. Returns (train_X, train_P, val_X, val_P)."""
    def _load(split: str):
        fx = data_dir / f"{split}_features.npy"
        fp = data_dir / f"{split}_prices.npy"
        if not fx.exists() or not fp.exists():
            raise FileNotFoundError(
                f"Missing {split} data. Run: python process_data.py"
            )
        X = np.load(str(fx), mmap_mode="r")
        P = np.load(str(fp), mmap_mode="r")
        logger.info("loaded", split=split, shape=X.shape)
        return X, P

    return _load("train") + _load("val")


def make_dummy_data(seq_len: int = 60, n_feat: int = 32, n_train: int = 5000, n_val: int = 1000):
    """Random data for smoke-testing."""
    rng = np.random.default_rng(42)
    X_tr = rng.standard_normal((n_train, seq_len, n_feat)).astype(np.float32)
    P_tr = np.ones((n_train, 2), dtype=np.float32) * 1.1
    X_vl = rng.standard_normal((n_val,  seq_len, n_feat)).astype(np.float32)
    P_vl = np.ones((n_val,  2), dtype=np.float32) * 1.1
    return X_tr, P_tr, X_vl, P_vl


def build_envs(X: np.ndarray, P: np.ndarray, cfg: AQRFConfig, n_envs: int, is_eval: bool = False):
    """Build vectorised + normalised environments."""
    rl_cfg = cfg.rl

    # Force deterministic friction for eval environments
    if is_eval:
        rl_cfg = rl_cfg.model_copy(deep=True)
        rl_cfg.friction.randomize = False

    # SubprocVecEnv hangs on Windows with large numpy arrays (pickle overhead)
    use_subproc = sys.platform != "win32" and n_envs > 1 and not is_eval

    fns = [make_env(X, P, rl_cfg, rank=i, monitor=True) for i in range(n_envs)]

    if use_subproc:
        vec_env = SubprocVecEnv(fns)
    else:
        vec_env = DummyVecEnv(fns)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,    # observations already z-scored in process_data
        norm_reward=True,
        clip_reward=10.0,
        gamma=rl_cfg.gamma,
    )
    return vec_env


def apply_phase3_config(cfg: AQRFConfig) -> AQRFConfig:
    """Override RLConfig fields for Phase 3 domain-randomization training."""
    cfg = cfg.model_copy(deep=True)
    rl  = cfg.rl
    rl.total_timesteps   = rl.p3_total_timesteps
    rl.ent_coef          = rl.p3_ent_coef
    rl.n_steps           = rl.p3_n_steps
    rl.learning_rate     = rl.p3_learning_rate
    rl.clip_range        = rl.p3_clip_range
    rl.friction.randomize = True   # domain randomization ON
    return cfg


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    # TF32: faster matmul on RTX 30xx with no accuracy loss for RL
    # cudnn.benchmark intentionally OFF — causes 3-5 min hang on first run while
    # it benchmarks Conv1d kernels; not worth it for the TCN's small input size.
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy",   action="store_true", help="Use random data for smoke-test")
    parser.add_argument("--phase3",  action="store_true", help="Phase 3: domain randomization (8M steps)")
    parser.add_argument("--resume",  type=str, default=None, help="Path to checkpoint ZIP")
    parser.add_argument("--config",  type=str, default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    if args.phase3:
        cfg = apply_phase3_config(cfg)
        logger.info("phase3_mode", total_timesteps=cfg.rl.total_timesteps,
                    ent_coef=cfg.rl.ent_coef, lr=cfg.rl.learning_rate,
                    friction_randomize=cfg.rl.friction.randomize)

    rl  = cfg.rl

    # ── Data ──────────────────────────────────────────────────────────
    if args.dummy:
        logger.info("using_dummy_data")
        X_tr, P_tr, X_vl, P_vl = make_dummy_data(
            seq_len=60, n_feat=32,
            n_train=5_000, n_val=1_000,
        )
    else:
        data_dir = Path("data/processed")
        X_tr, P_tr, X_vl, P_vl = load_data(data_dir)

    # ── Environments ──────────────────────────────────────────────────
    n_train_envs = rl.n_envs
    train_env = build_envs(X_tr, P_tr, cfg, n_train_envs)
    eval_env  = build_envs(X_vl, P_vl, cfg, 1, is_eval=True)

    # ── Checkpoints / logs ───────────────────────────────────────────
    from datetime import datetime
    phase_tag  = "phase3" if args.phase3 else "phase2"
    run_stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Each run gets its own timestamped folder so reruns never overwrite previous models.
    # Structure: checkpoints/rl/phase2/20260523_172000/  (best/, eval_logs/, checkpoints)
    #            checkpoints/rl/phase3/20260523_180000/  (best/, eval_logs/, checkpoints)
    ckpt_dir   = Path(f"checkpoints/rl/{phase_tag}/{run_stamp}")
    log_dir    = Path(f"tensorboard_logs/{phase_tag}/{run_stamp}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("checkpoint_dir", path=str(ckpt_dir))

    # ── Agent ─────────────────────────────────────────────────────────
    if args.resume:
        logger.info("resuming_from_checkpoint", path=args.resume)
        from stable_baselines3 import PPO
        model = PPO.load(args.resume, env=train_env,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = create_ppo_agent(train_env, config=rl, tensorboard_dir=str(log_dir))

    # ── Callbacks ─────────────────────────────────────────────────────
    # Build early-stopping callback first so it can be wired into EvalCallback
    # via callback_after_eval= (the only way SB3 actually invokes it).
    early_stop_cb = None
    if args.phase3:
        early_stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=15,
            min_evals=20,
        )
        logger.info("early_stopping_enabled", max_no_improvement=15, min_evals=20)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir / "best"),
        log_path=str(ckpt_dir / "eval_logs"),
        eval_freq=max(rl.eval_freq // n_train_envs, 1),
        n_eval_episodes=rl.n_eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=early_stop_cb,  # None is fine for Phase 2
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(rl.save_freq // n_train_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="ppo_forex",
        save_vecnormalize=True,
    )

    logging_cb = LoggingCallback(log_every=2_000)

    callbacks = [eval_cb, ckpt_cb, logging_cb]

    if args.phase3:
        conf_cb = ConfidenceLoggingCallback(config=rl)
        callbacks.append(conf_cb)
        logger.info("confidence_logging_enabled", log_every=rl.entropy_log_freq)

    # ── Train ─────────────────────────────────────────────────────────
    logger.info(
        "training_start",
        phase="3" if args.phase3 else "2",
        total_timesteps=rl.total_timesteps,
        n_envs=n_train_envs,
        device=str(model.device),
    )

    model.learn(
        total_timesteps=rl.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
        reset_num_timesteps=not bool(args.resume),
    )

    # ── Save final ────────────────────────────────────────────────────
    final_path = ckpt_dir / "ppo_forex_final"
    model.save(str(final_path))
    train_env.save(str(ckpt_dir / "vecnormalize_final.pkl"))
    logger.info("training_complete", saved_to=str(final_path))


if __name__ == "__main__":
    main()
