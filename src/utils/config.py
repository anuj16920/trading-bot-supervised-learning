"""Configuration management for AQRF.

Uses Pydantic v2 for validation and type safety.
All hyperparameters live here — zero magic numbers in code.
"""
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# ── Data Configuration ──────────────────────────────────────────────

class DataConfig(BaseModel):
    """Data pipeline configuration."""

    symbol: str = "EURUSD"
    pip_size: float = 0.0001

    # Paths
    raw_tick_dir: Path = Path("./data/EURUSD/tick")
    m1_dir: Path = Path("./data/EURUSD/M1")
    h1_dir: Path = Path("./data/EURUSD/H1")

    # Processing
    chunk_size: int = 500_000
    float_dtype: Literal["float32", "float64"] = "float32"

    # Validation thresholds
    max_spread_pips: float = 10.0
    max_price_change_pips: float = 500.0
    max_gap_hours: float = 1.0

    # Splits (strictly chronological)
    train_start: str = "2015-01-01"
    train_end: str = "2021-12-31"
    val_start: str = "2022-01-01"
    val_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2024-12-31"

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk(cls, v: int) -> int:
        if v < 10_000:
            raise ValueError("chunk_size must be >= 10,000")
        return v


# ── Feature Configuration ───────────────────────────────────────────

class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    # Window sizes
    seq_len: int = 60  # timesteps per sample
    seq_len_regime: int = 100  # regime detector needs longer context

    # Rolling windows
    vol_windows: list[int] = Field(default=[20, 60, 200])
    velocity_windows: list[int] = Field(default=[5, 20])
    autocorr_windows: list[int] = Field(default=[1, 5, 10])
    autocorr_rolling: int = 50

    # Normalization
    zscore_window: int = 500
    zscore_clip: float = 5.0

    # Multi-timeframe
    timeframes: list[str] = Field(default=["M1", "M5", "M15", "H1", "H4"])

    # Output
    n_features: int = 45  # total features after MTF

    @field_validator("seq_len")
    @classmethod
    def validate_seq(cls, v: int) -> int:
        if v < 10:
            raise ValueError("seq_len must be >= 10")
        return v


# ── Model Configuration ─────────────────────────────────────────────

class TCNConfig(BaseModel):
    """Temporal Convolutional Network config."""

    n_features: int = 12  # Must match processed data shape (train_features.npy dim 2)
    channels: list[int] = Field(default=[64, 64, 128, 128, 256])
    kernel_size: int = 3
    dropout: float = 0.2
    dilation_base: int = 2

    # Training
    batch_size: int = 512
    target_vram_gb: float = 1.2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    warmup_epochs: int = 2

    # Output
    direction_classes: int = 2


class TransformerConfig(BaseModel):
    """Transformer model config."""

    n_features: int = 12  # Must match processed data shape (train_features.npy dim 2)
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"

    # Training
    batch_size: int = 256
    target_vram_gb: float = 1.5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    warmup_epochs: int = 2
    use_gradient_checkpointing: bool = False  # enable if OOM

    # Output
    direction_classes: int = 2


class RegimeConfig(BaseModel):
    """Regime detector config."""

    n_features: int = 12  # Must match processed data shape (train_features.npy dim 2)
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    num_regimes: int = 4

    # Training
    batch_size: int = 256
    target_vram_gb: float = 0.8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    label_smoothing: float = 0.1
    warmup_epochs: int = 2

    # Labeling thresholds
    adx_trending: float = 25.0
    adx_ranging: float = 20.0
    vol_threshold_sigma: float = 2.0


class ModelConfig(BaseModel):
    """Unified model configuration."""

    tcn: TCNConfig = Field(default_factory=TCNConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)

    # Ensemble
    tcn_weight: float = 0.4
    transformer_weight: float = 0.4
    regime_weight: float = 0.2

    # Training
    epochs: int = 100
    early_stopping_patience: int = 15
    scheduler_t0: int = 10

    # Precision
    use_amp: bool = True
    use_8bit_adam: bool = True


# ── Phase 3 Friction Configuration ──────────────────────────────────

class FrictionConfig(BaseModel):
    """Market friction & execution randomization (Module 1 — Domain Randomization)."""

    # False = deterministic eval friction (Phase 2 training and all evals).
    # True  = domain randomization ON (Phase 3 only — set by apply_phase3_config()).
    randomize: bool = False

    # Spread range (pips) — sampled each episode when randomize=True
    spread_min_pips: float = 0.3
    spread_max_pips: float = 3.0

    # Slippage range (pips)
    slippage_min_pips: float = 0.1
    slippage_max_pips: float = 1.5

    # Execution delay (bars) — order fills N bars after signal
    delay_min_bars: int = 0
    delay_max_bars: int = 3

    # Fill quality: 1.0 = perfect fill, 0.7 = 30% of slippage worsens fill
    fill_quality_min: float = 0.7
    fill_quality_max: float = 1.0

    # Deterministic eval values (used when randomize=False)
    eval_spread_pips: float = 0.5
    eval_slippage_pips: float = 0.3
    # Minimum 1-bar delay: agent sees bar T, fills at bar T+1 price.
    # Prevents same-bar execution where decision and fill use the same close.
    eval_delay_bars: int = 1


# ── Phase 3 Stress Test Configuration ───────────────────────────────

class StressConfig(BaseModel):
    """Stress testing & chaos simulation parameters (Module 5)."""

    n_trials: int = 200
    n_episodes_per_trial: int = 20
    ruin_threshold: float = 0.15        # fraction of capital loss = "ruin"
    cvar_percentile: float = 0.05       # CVaR tail (worst 5%)

    # Flash crash
    flash_crash_pips_min: float = 50.0
    flash_crash_pips_max: float = 200.0
    flash_crash_probability: float = 0.002  # per bar
    flash_crash_duration_bars: int = 3
    flash_recovery_fraction: float = 0.65   # price recovers 65% of drop

    # Spread spikes
    spread_spike_pips_min: float = 5.0
    spread_spike_pips_max: float = 30.0
    spread_spike_probability: float = 0.005
    spread_spike_duration_bars: int = 1

    # Volatility explosion
    vol_explosion_multiplier_min: float = 2.0
    vol_explosion_multiplier_max: float = 8.0
    vol_explosion_duration_min: int = 20
    vol_explosion_duration_max: int = 200
    vol_explosion_probability: float = 0.001


# ── RL Configuration ────────────────────────────────────────────────

class RLConfig(BaseModel):
    """Reinforcement learning configuration."""

    # Environment
    # STRIDE=3 in process_data.py: each env step = 3 M1 bars.
    # 7,200 steps × 3 = 21,600 M1 bars = 15 trading days.
    # Long episodes teach the agent to wait for quality setups.
    episode_bars: int = 7_200
    initial_capital: float = 10_000.0
    lot_size: float = 10_000.0   # micro-lot: 1 pip = $1

    # Action space: 0=hold, 1=buy, 2=sell, 3=close
    n_actions: int = 4

    # PPO
    learning_rate: float = 1e-4
    n_steps: int = 8192
    batch_size: int = 1024        # bigger minibatch → more GPU work per update, fewer syncs
    n_epochs: int = 5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Vectorised envs (Windows: DummyVecEnv only — SubprocVecEnv hangs with large arrays)
    # n_envs=2: going higher hurts FPS on Windows because DummyVecEnv inference is serial
    n_envs: int = 2

    # Training
    total_timesteps: int = 2_000_000
    eval_freq: int = 250_000
    save_freq: int = 500_000
    n_eval_episodes: int = 1

    # Reward shaping
    drawdown_penalty: float = 3.0
    drawdown_threshold: float = 0.02
    # overtrade_threshold counts ROUND-TRIPS (open+close = 1 trade).
    # Reward now in pips: TP=+20, SL=-10. Penalty=-25 pips makes excess trades net-negative.
    # Target 3-4 trades/day × 15 days = 45-60 round-trips per episode.
    overtrade_penalty: float = -25.0   # pips equivalent — exceeds a TP win
    overtrade_threshold: int = 60      # round-trips per 15-day episode (STRIDE=3 invariant)

    # Cooldown: steps the agent must wait (flat) after closing a trade.
    # STRIDE=3: 80 steps × 3 = 240 M1 bars = 4 hours. Targets 3-4 trades/day.
    trade_cooldown_bars: int = 80

    # Strict daily trade cap: hard-blocks new entries once max_trades_per_day is reached.
    # STRIDE=3: 480 steps × 3 = 1440 M1 bars = 1 trading day.
    # This is a HARD environment rule — the agent cannot override it regardless of reward.
    max_trades_per_day: int = 4        # absolute cap: 4 round-trips per calendar day
    bars_per_day: int = 480            # STRIDE=3: 480 steps = 1440 M1 bars = 24 hours

    # Hold penalty: DISABLED (0.0) — cooldown already prevents scalping.
    # A positive hold penalty forces the agent to trade constantly to avoid punishment.
    # With 21,600-bar episodes, even a tiny per-interval penalty dominates the reward.
    hold_penalty: float = 0.0
    hold_penalty_bars: int = 500
    hold_penalty_interval: int = 50

    # Termination
    max_drawdown: float = 0.10
    min_account: float = 8_000.0

    # 1:2 R:R enforcement
    stop_loss_pips: float = 10.0    # hard stop — auto-close if loss exceeds this
    take_profit_pips: float = 20.0  # hard TP  — auto-close if gain exceeds this (1:2 R:R)
    reward_scale_win: float = 1.5   # TP bonus multiplier
    reward_scale_loss: float = 1.0  # SL penalty multiplier (1.0 = no extra punishment beyond pip loss)

    # Reward: plain pip-based, no Sharpe normalization.
    # Sharpe normalization divided by vol caused reward magnitude to swing wildly
    # (low-vol periods → huge rewards, high-vol → near-zero), making the value
    # function impossible to fit. Plain pips are stable and interpretable.
    vol_window: int = 20            # kept for realized_vol diagnostics in _info
    sharpe_scale: float = 1.0      # effectively disabled (was 10.0)

    # Reproducibility
    random_seed: int = 42

    # Phase 3 — Market friction & domain randomization (Module 1)
    friction: FrictionConfig = Field(default_factory=FrictionConfig)

    # Phase 3 — Confidence-aware trading (Module 2)
    confidence_threshold: float = 0.65   # min max(action_probs) to act; else force HOLD
    use_confidence_filter: bool = True    # set False to disable during ablation
    entropy_log_freq: int = 1000          # log policy entropy every N steps

    # Phase 3 — Walk-forward retraining (Module 3)
    wf_window_months: int = 6
    wf_finetune_timesteps: int = 500_000
    wf_n_eval_episodes: int = 10

    # Phase 3 — Domain-randomized training overrides (activated with --phase3 flag)
    p3_total_timesteps: int = 4_000_000
    p3_ent_coef: float = 0.03
    p3_n_steps: int = 4096          # match Phase 2 — shorter rollouts hurt value estimation with long episodes
    p3_learning_rate: float = 5e-5
    p3_clip_range: float = 0.15


# ── Risk Configuration ──────────────────────────────────────────────

class RiskConfig(BaseModel):
    """Risk management configuration."""

    # Kelly
    max_kelly_fraction: float = 0.02
    kelly_fraction_multiplier: float = 0.5  # half-Kelly

    # Stops
    stop_vol_multiplier: float = 2.0
    min_rr_ratio: float = 1.5

    # Guard
    min_model_confidence: float = 0.60
    min_kelly_size: float = 0.005
    max_daily_trades: int = 15
    max_daily_drawdown: float = 0.02

    # Position sizing
    max_position_risk: float = 0.02  # 2% of capital
    lot_size: float = 100_000.0  # standard forex lot


# ── Unified Config ──────────────────────────────────────────────────

class AQRFConfig(BaseSettings):
    """Master configuration for AQRF system."""

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    stress: StressConfig = Field(default_factory=StressConfig)

    # System
    random_seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    class Config:
        env_prefix = "AQRF_"
        case_sensitive = False


def load_config(path: Optional[Path] = None) -> AQRFConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config. If None, uses defaults.

    Returns:
        Validated AQRFConfig instance.
    """
    if path is None or not path.exists():
        return AQRFConfig()

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return AQRFConfig(**raw)


def save_config(config: AQRFConfig, path: Path) -> None:
    """Save configuration to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
