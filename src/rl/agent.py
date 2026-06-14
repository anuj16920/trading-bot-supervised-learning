"""PPO agent with a TCN feature extractor built into the policy.

The TCN extracts temporal patterns from the (seq_len, n_features) observation
directly — no separate supervised pre-training needed. The RL gradient
trains the whole network end-to-end.

Architecture:
  obs (seq_len, n_features) → TCN → flat latent → MLP → action / value
"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym
import numpy as np

from src.rl.environment import ForexTradingEnv
from src.utils.config import RLConfig


# ── TCN block (same as src/models/tcn.py but self-contained here) ────

class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)[:, :, :x.shape[-1]]   # causal: trim right padding
        out = self.act(self.norm(out))
        out = self.drop(out)
        return out + self.res(x)


class TCNExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor: (batch, seq_len, n_feat) → (batch, latent_dim)."""

    def __init__(self, observation_space: gym.spaces.Box, latent_dim: int = 256):
        super().__init__(observation_space, features_dim=latent_dim)

        seq_len, n_feat = observation_space.shape

        channels = [n_feat, 64, 128, 128, 256]
        kernel   = 3
        dropout  = 0.1

        blocks = []
        for i in range(len(channels) - 1):
            dilation = 2 ** i
            blocks.append(_TCNBlock(channels[i], channels[i + 1], kernel, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], latent_dim),
            nn.GELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, seq_len, n_feat) → (batch, n_feat, seq_len) for Conv1d
        x = obs.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)   # (batch, 256)
        return self.proj(x)


# ── Factory helpers ───────────────────────────────────────────────────

def make_env(features: np.ndarray, prices: np.ndarray, config: RLConfig, rank: int = 0, monitor: bool = False):
    """Return a thunk that creates a seeded ForexTradingEnv."""
    from stable_baselines3.common.monitor import Monitor
    def _init():
        env = ForexTradingEnv(features, prices, config)
        env.reset(seed=config.random_seed + rank)
        if monitor:
            env = Monitor(env)
        return env
    return _init


def create_ppo_agent(
    vec_env,
    config: RLConfig | None = None,
    tensorboard_dir: str | None = None,
) -> PPO:
    """Create PPO with TCN feature extractor.

    Args:
        vec_env: Already-created VecEnv (with VecNormalize wrapper recommended).
        config: RL configuration.
        tensorboard_dir: TensorBoard log path.
    """
    cfg = config or RLConfig()

    policy_kwargs = dict(
        features_extractor_class=TCNExtractor,
        features_extractor_kwargs=dict(latent_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.GELU,
    )

    def lr_schedule(progress: float) -> float:
        return cfg.learning_rate * max(progress, 0.05)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        use_sde=False,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_dir,
        device=device,
        verbose=1,
    )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[agent] PPO created | params={n_params:,} | device={model.device}")
    return model
