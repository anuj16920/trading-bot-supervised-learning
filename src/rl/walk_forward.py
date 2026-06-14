"""Walk-Forward Adaptive Retraining for the PPO agent (Module 3 — Phase 3).

Implements rolling 6-month fine-tuning windows. Each window:
  1. Loads previous best checkpoint
  2. Attaches it to a new training environment (the new window's data)
  3. Fine-tunes for wf_finetune_timesteps steps
  4. Evaluates on the following test window
  5. Saves checkpoint to checkpoints/rl/walkforward/window_N/

This keeps the policy adapted to evolving market conditions rather than
freezing it at the end of Phase 2 training.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.environment import ForexTradingEnv
from src.utils.config import RLConfig, load_config

logger = structlog.get_logger(__name__)

# Approximate M1 bars per month (30 days * 24h * 60min, forex ~22 trading days)
BARS_PER_MONTH = 22 * 24 * 60  # ~31,680


class RLWalkForwardTrainer:
    """Walk-forward fine-tuning for the PPO RL agent.

    Args:
        base_model_path:    Path to best_model.zip from Phase 2 training.
        base_vecnorm_path:  Path to vecnormalize_final.pkl from Phase 2.
        data_dir:           Directory containing train_features.npy, train_prices.npy, etc.
        config:             RLConfig with walk-forward parameters.
        output_dir:         Where per-window checkpoints are saved.
        window_months:      Size of each fine-tune window in months.
        finetune_timesteps: Training steps per window.
    """

    def __init__(
        self,
        base_model_path: Path,
        base_vecnorm_path: Path,
        data_dir: Path,
        config: Optional[RLConfig] = None,
        output_dir: Path = Path("checkpoints/rl/walkforward"),
        window_months: Optional[int] = None,
        finetune_timesteps: Optional[int] = None,
    ):
        self.base_model_path  = Path(base_model_path)
        self.base_vecnorm_path = Path(base_vecnorm_path)
        self.data_dir         = Path(data_dir)
        self.cfg              = config or RLConfig()
        self.output_dir       = Path(output_dir)
        self.window_months    = window_months or self.cfg.wf_window_months
        self.finetune_steps   = finetune_timesteps or self.cfg.wf_finetune_timesteps

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load full training data once (memory-mapped)
        self._X: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None

    def _load_data(self) -> None:
        if self._X is not None:
            return
        logger.info("loading_train_data", dir=str(self.data_dir))
        self._X = np.load(str(self.data_dir / "train_features.npy"), mmap_mode="r")
        self._P = np.load(str(self.data_dir / "train_prices.npy"),   mmap_mode="r")
        logger.info("train_data_loaded", shape=self._X.shape)

    def _generate_splits(self) -> list[dict]:
        """Generate expanding-window splits.

        Each split:
          train: bars [0, window_end)
          test:  bars [window_end, window_end + window_size)

        Returns list of dicts with train_end, test_start, test_end bar indices.
        """
        self._load_data()
        total_bars  = len(self._X)
        window_size = self.window_months * BARS_PER_MONTH
        splits = []

        # Start first window at 50% of data (preserve enough history)
        start_bar = total_bars // 2
        cursor    = start_bar + window_size

        while cursor + window_size <= total_bars:
            splits.append({
                "train_end":   cursor,
                "test_start":  cursor,
                "test_end":    min(cursor + window_size, total_bars),
                "window_idx":  len(splits),
            })
            cursor += window_size

        logger.info("splits_generated", n_splits=len(splits), window_months=self.window_months)
        return splits

    def _build_env(
        self,
        X: np.ndarray,
        P: np.ndarray,
        vecnorm_path: Optional[Path] = None,
        is_eval: bool = False,
    ) -> VecNormalize:
        """Build DummyVecEnv + VecNormalize for a data window."""
        rl_cfg = self.cfg.model_copy(deep=True)
        if is_eval:
            rl_cfg.friction.randomize = False

        def make_env():
            return ForexTradingEnv(X, P, config=rl_cfg)

        env = DummyVecEnv([make_env])
        if vecnorm_path and Path(vecnorm_path).exists():
            env = VecNormalize.load(str(vecnorm_path), env)
            env.training = not is_eval
            env.norm_reward = not is_eval
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        return env

    def _finetune_window(
        self,
        model_path: Path,
        vecnorm_path: Path,
        X_train: np.ndarray,
        P_train: np.ndarray,
        X_val: np.ndarray,
        P_val: np.ndarray,
        window_idx: int,
    ) -> tuple[Path, Path]:
        """Fine-tune on one window. Returns (saved_model_path, saved_vecnorm_path)."""
        win_dir = self.output_dir / f"window_{window_idx:03d}"
        win_dir.mkdir(parents=True, exist_ok=True)

        logger.info("finetune_start", window=window_idx, train_bars=len(X_train))

        train_env = self._build_env(X_train, P_train, vecnorm_path=vecnorm_path)
        model = PPO.load(str(model_path), env=train_env)

        # Fine-tune — reset_num_timesteps=False preserves LR schedule continuity
        model.learn(
            total_timesteps=self.finetune_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        out_model  = win_dir / "model.zip"
        out_vecnorm = win_dir / "vecnormalize.pkl"
        model.save(str(out_model))
        train_env.save(str(out_vecnorm))

        logger.info("finetune_saved", window=window_idx, path=str(out_model))

        # Quick eval on val window
        metrics = self._eval_window(out_model, X_val, P_val)
        logger.info("window_eval", window=window_idx, **metrics)

        # Cleanup
        del model, train_env
        gc.collect()

        return out_model, out_vecnorm

    def _eval_window(
        self,
        model_path: Path,
        X: np.ndarray,
        P: np.ndarray,
        n_episodes: int = 10,
    ) -> dict:
        """Quick eval on a data window. Returns mean return%, win rate, mean Sharpe."""
        rl_cfg = self.cfg.model_copy(deep=True)
        rl_cfg.friction.randomize = False
        env  = ForexTradingEnv(X, P, config=rl_cfg)
        model = PPO.load(str(model_path))

        pnls, wins, trades_list = [], [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(action))
                done = term or trunc
            pnls.append(env.capital - rl_cfg.initial_capital)
            wins.append(env.winning)
            trades_list.append(env.total_trades)

        del model
        mean_ret  = float(np.mean(pnls) / rl_cfg.initial_capital * 100)
        total_t   = sum(trades_list)
        win_rate  = sum(wins) / total_t if total_t > 0 else 0.0
        return {
            "mean_return_pct": round(mean_ret, 3),
            "win_rate":        round(win_rate, 3),
            "n_episodes":      n_episodes,
        }

    def run(self) -> list[dict]:
        """Run all walk-forward windows. Returns list of per-window metric dicts."""
        self._load_data()
        splits  = self._generate_splits()
        results = []

        current_model  = self.base_model_path
        current_vecnorm = self.base_vecnorm_path

        for split in splits:
            idx        = split["window_idx"]
            X_train    = self._X[:split["train_end"]]
            P_train    = self._P[:split["train_end"]]
            X_val      = self._X[split["test_start"]:split["test_end"]]
            P_val      = self._P[split["test_start"]:split["test_end"]]

            if len(X_val) < self.cfg.episode_bars * 2:
                logger.warning("val_window_too_small", window=idx, bars=len(X_val))
                continue

            new_model, new_vecnorm = self._finetune_window(
                current_model, current_vecnorm,
                X_train, P_train,
                X_val,   P_val,
                window_idx=idx,
            )

            metrics = self._eval_window(new_model, X_val, P_val,
                                        n_episodes=self.cfg.wf_n_eval_episodes)
            metrics["window_idx"]   = idx
            metrics["train_bars"]   = split["train_end"]
            metrics["test_bars"]    = split["test_end"] - split["test_start"]
            metrics["model_path"]   = str(new_model)
            results.append(metrics)

            # Chain: next window starts from this window's checkpoint
            current_model   = new_model
            current_vecnorm = new_vecnorm

        logger.info("walkforward_complete", n_windows=len(results))
        return results
