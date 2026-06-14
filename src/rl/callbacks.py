"""Custom callbacks for RL training.

Early stopping on no improvement, logging, and Phase 3 confidence/entropy monitoring.
"""
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
import structlog

from src.utils.config import RLConfig

logger = structlog.get_logger(__name__)


class StopTrainingOnNoModelImprovement(BaseCallback):
    """Stop Phase 3 training if eval reward doesn't improve for N consecutive evals.

    Must be passed as callback_after_eval= to EvalCallback so it fires after
    every evaluation. Reads last_mean_reward from the parent EvalCallback.
    """

    def __init__(
        self,
        max_no_improvement_evals: int = 20,
        min_evals: int = 50,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.max_no_improvement = max_no_improvement_evals
        self.min_evals = min_evals
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.eval_count = 0

    def _on_step(self) -> bool:
        # Called by EvalCallback after each evaluation via callback_after_eval.
        # parent is the EvalCallback instance.
        mean_reward = self.parent.last_mean_reward  # type: ignore[attr-defined]

        self.eval_count += 1

        if self.eval_count < self.min_evals:
            return True

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.no_improvement_count = 0
            logger.info("early_stop_new_best", reward=round(mean_reward, 2))
        else:
            self.no_improvement_count += 1
            logger.info(
                "early_stop_no_improvement",
                count=self.no_improvement_count,
                max=self.max_no_improvement,
                best=round(self.best_mean_reward, 2),
                current=round(mean_reward, 2),
            )

        if self.no_improvement_count >= self.max_no_improvement:
            logger.info(
                "early_stop_triggered",
                evals=self.eval_count,
                best_reward=round(self.best_mean_reward, 2),
            )
            return False  # signals SB3 to stop training

        return True


class LoggingCallback(BaseCallback):
    """Log training metrics every N steps."""

    def __init__(self, log_every: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            info = self.locals.get("infos", [{}])[0]
            # Use capital-derived pnl, not VecNormalize-clipped reward (which is near 0)
            capital = info.get("capital", 10000.0)
            pnl_total = round(capital - 10000.0, 2)
            logger.info(
                "rl_step",
                step=self.n_calls,
                pnl_total=pnl_total,
                capital=round(capital, 2),
                drawdown=round(info.get("drawdown", 0), 4),
                trades=info.get("trades", 0),
                daily_trades=info.get("daily_trades", 0),
                sl_hits=info.get("sl_hits", 0),
                tp_hits=info.get("tp_hits", 0),
            )
        return True


class ConfidenceLoggingCallback(BaseCallback):
    """Phase 3 (Module 2) — Log policy entropy and mean confidence to TensorBoard.

    Samples a small batch of recent observations from the rollout buffer,
    computes action probability distributions, and logs:
      - mean policy entropy (high = uncertain, low = confident)
      - mean max(action_probs) (confidence proxy)
      - friction diagnostics (mean spread/slippage if available in infos)
    """

    def __init__(self, config: RLConfig, log_every: Optional[int] = None, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = config
        self.log_every = log_every or config.entropy_log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every != 0:
            return True

        try:
            # Sample last observation from the vectorised env
            obs = self.locals.get("obs_tensor")
            if obs is None:
                return True

            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs)
                probs = dist.distribution.probs.cpu().numpy()  # (n_envs, 4)

            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1).mean())
            mean_confidence = float(np.max(probs, axis=1).mean())

            self.logger.record("phase3/policy_entropy", entropy)
            self.logger.record("phase3/mean_confidence", mean_confidence)

            # Log friction diagnostics from env infos
            infos = self.locals.get("infos", [{}])
            if infos and "ep_spread_pips" in infos[0]:
                spreads = [i.get("ep_spread_pips", 0.0) for i in infos]
                slippages = [i.get("ep_slippage_pips", 0.0) for i in infos]
                self.logger.record("phase3/ep_spread_pips", float(np.mean(spreads)))
                self.logger.record("phase3/ep_slippage_pips", float(np.mean(slippages)))

        except Exception as e:
            logger.warning("confidence_logging_error", error=str(e))

        return True
