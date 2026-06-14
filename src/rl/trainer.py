"""RL training orchestrator for AQRF.

Vectorized environments, PPO training, checkpointing.
"""
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import structlog

from src.rl.environment import ForexTradingEnv
from src.rl.agent import make_env, create_ppo_agent
from src.rl.callbacks import StopTrainingOnNoModelImprovement, LoggingCallback
from src.utils.config import RLConfig
from src.utils.gpu import release_vram

logger = structlog.get_logger(__name__)


class RLTrainer:
    """Orchestrates RL training with vectorized environments."""

    def __init__(
        self,
        train_data: np.ndarray,
        train_prices: np.ndarray,
        val_data: np.ndarray,
        val_prices: np.ndarray,
        config: Optional[RLConfig] = None,
        output_dir: Path = Path("./checkpoints"),
    ):
        self.config = config or RLConfig()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_data = train_data
        self.train_prices = train_prices
        self.val_data = val_data
        self.val_prices = val_prices

        self.vec_env = None
        self.eval_env = None
        self.model = None

    def setup(self) -> None:
        """Setup vectorized training and eval environments."""
        # Training envs
        train_env_fns = [
            make_env(self.train_data, self.train_prices, self.config, i)
            for i in range(self.config.n_envs)
        ]

        self.vec_env = SubprocVecEnv(train_env_fns)
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
        )

        # Eval env (single)
        eval_env_fns = [
            make_env(self.val_data, self.val_prices, self.config, 999)
        ]
        self.eval_env = SubprocVecEnv(eval_env_fns)
        self.eval_env = VecNormalize(
            self.eval_env,
            norm_obs=True,
            norm_reward=True,
        )

        logger.info(
            "rl_setup_complete",
            n_envs=self.config.n_envs,
            train_samples=len(self.train_data),
            val_samples=len(self.val_data),
        )

    def train(self) -> None:
        """Run PPO training."""
        self.setup()

        # Create agent
        self.model = create_ppo_agent(
            self.vec_env,
            self.config,
            tensorboard_dir=str(self.output_dir / "tensorboard"),
        )

        # Callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.output_dir),
            log_path=str(self.output_dir / "eval_logs"),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(self.output_dir),
            name_prefix="ppo_forex",
        )

        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=20,
            min_evals=50,
        )

        log_callback = LoggingCallback(log_every=10_000)

        # Train
        logger.info("rl_training_start", total_timesteps=self.config.total_timesteps)

        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=[
                eval_callback,
                checkpoint_callback,
                stop_callback,
                log_callback,
            ],
            progress_bar=True,
        )

        logger.info("rl_training_complete")

    def save(self, path: Optional[Path] = None) -> None:
        """Save trained model and normalization stats."""
        if path is None:
            path = self.output_dir / "ppo_final"

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        self.vec_env.save(str(path) + "_vecnormalize.pkl")

        logger.info("rl_model_saved", path=str(path))

    def cleanup(self) -> None:
        """Release all resources."""
        if self.vec_env is not None:
            self.vec_env.close()
        if self.eval_env is not None:
            self.eval_env.close()

        del self.model
        release_vram()
        gc.collect()

        logger.info("rl_trainer_cleanup_complete")
