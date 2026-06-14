# AQRF System Architecture

## Overview

AQRF (Adaptive Quantitative Reinforcement Framework) is a single end-to-end RL trading system.
No prediction layer. The agent observes raw market features and learns to take profitable trades directly.

---

## System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                             │
│                                                                     │
│   EURUSD/ohlcv/1min/    EURUSD/ohlcv/5min/    EURUSD/ohlcv/15min/  │
│   EURUSD/ohlcv/1hour/   EURUSD/ohlcv/4hour/                        │
│                         2016 – 2024                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       process_data.py                               │
│                                                                     │
│   Multi-Timeframe Feature Engineering                               │
│   ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ │
│   │  M1 (9)  │ │  M5 (5)  │ │ M15 (5)   │ │  H1 (5)  │ │ H4 (4) │ │
│   │ return_1 │ │m5_return │ │m15_return │ │h1_return │ │h4_ret  │ │
│   │ return_5 │ │m5_range  │ │m15_range  │ │h1_range  │ │h4_range│ │
│   │return_20 │ │m5_body   │ │m15_body   │ │h1_body   │ │h4_vol  │ │
│   │range_pct │ │m5_vol    │ │m15_vol    │ │h1_vol    │ │h4_ma20 │ │
│   │body_pct  │ │m5_ma20   │ │m15_ma20   │ │h1_ma20   │ └────────┘ │
│   │vol_ratio │ └──────────┘ └───────────┘ └──────────┘            │
│   │ma_ratio  │                                                     │
│   │vol_20    │  Session (4): hour_sin, hour_cos, dow_sin, dow_cos  │
│   │spread_pct│                                                     │
│   └──────────┘                                                     │
│                                                                     │
│   Total: 32 features per M1 bar                                    │
│   Output shape: (N, 60 bars, 32 features)                          │
│                                                                     │
│   Splits:  Train 2016-2021 → 581,291 sequences                     │
│            Val   2022      →  98,129 sequences                     │
│            Test  2023-2024 → 199,939 sequences                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    data/processed/                                  │
│                                                                     │
│   train_features.npy  (581291, 60, 32)  float32   4,257 MB         │
│   train_prices.npy    (581291, 2)        [bid, ask]                 │
│   val_features.npy    ( 98129, 60, 32)   718 MB                    │
│   val_prices.npy      ( 98129, 2)                                   │
│   test_features.npy   (199939, 60, 32)  1,464 MB                   │
│   test_prices.npy     (199939, 2)                                   │
│   norm_stats.npz      mean/std from train split                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       train_rl.py                                   │
│                  (single training entry point)                      │
│                                                                     │
│   ┌─────────────────────────┐   ┌─────────────────────────────┐    │
│   │   ForexTradingEnv ×2    │   │    ForexTradingEnv ×1       │    │
│   │   (train, DummyVecEnv)  │   │    (eval, DummyVecEnv)      │    │
│   │   VecNormalize wrapper  │   │    VecNormalize wrapper      │    │
│   └────────────┬────────────┘   └──────────────┬──────────────┘    │
│                │                               │                   │
│                └──────────────┬────────────────┘                   │
│                               │                                    │
│                               ▼                                    │
│              ┌────────────────────────────┐                        │
│              │         PPO Agent          │                        │
│              │    stable-baselines3       │                        │
│              │                           │                        │
│              │  n_steps      = 4096      │                        │
│              │  batch_size   = 256       │                        │
│              │  n_epochs     = 5         │                        │
│              │  clip_range   = 0.1       │                        │
│              │  lr           = 1e-4      │                        │
│              │  gamma        = 0.99      │                        │
│              └────────────┬───────────────┘                       │
└───────────────────────────┼─────────────────────────────────────── ┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PPO Policy Network                              │
│                                                                     │
│  Input: observation (60, 36)                                        │
│         = 32 market features + 4 portfolio features                 │
│         [position, unrealised_pnl%, drawdown, bars_held_norm]       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   TCNExtractor  (372K params)                 │  │
│  │                                                              │  │
│  │  Input (batch, 60, 36)                                       │  │
│  │       │                                                      │  │
│  │       ▼  permute → (batch, 36, 60)                           │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  TCNBlock 0:  36 → 64ch,  dilation=1,  kernel=3     │    │  │
│  │  │  TCNBlock 1:  64 → 128ch, dilation=2,  kernel=3     │    │  │
│  │  │  TCNBlock 2: 128 → 128ch, dilation=4,  kernel=3     │    │  │
│  │  │  TCNBlock 3: 128 → 256ch, dilation=8,  kernel=3     │    │  │
│  │  │                                                     │    │  │
│  │  │  Each block: DilatedCausalConv → GroupNorm → GELU   │    │  │
│  │  │              + residual connection                  │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  │       │                                                      │  │
│  │       ▼  AdaptiveAvgPool1d → (batch, 256)                    │  │
│  │       ▼  Linear(256→256) + GELU                              │  │
│  │                                                              │  │
│  │  Output: latent vector (batch, 256)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                         │
│              ┌────────────┴────────────┐                           │
│              ▼                         ▼                           │
│   ┌──────────────────┐     ┌──────────────────────┐               │
│   │  Policy Head     │     │   Value Head          │               │
│   │  MLP [128, 64]   │     │   MLP [128, 64]       │               │
│   │  → 4 actions     │     │   → 1 (state value)   │               │
│   └──────────────────┘     └──────────────────────┘               │
│                                                                     │
│  Actions:  0 = hold                                                 │
│            1 = buy  (open long / close short)                       │
│            2 = sell (open short / close long)                       │
│            3 = close (exit current position)                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  action
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ForexTradingEnv                                  │
│                    src/rl/environment.py                            │
│                                                                     │
│  State                                                              │
│  ├── capital          starts at $10,000                             │
│  ├── position         +1 long / -1 short / 0 flat                  │
│  ├── entry_price      price at which position was opened            │
│  ├── bars_held        how long current position has been open       │
│  └── peak_capital     for drawdown tracking                         │
│                                                                     │
│  Execution (realistic costs)                                        │
│  ├── spread           actual bid/ask from data                      │
│  └── slippage         0.3 pips per fill                             │
│                                                                     │
│  Reward (per step)                                                  │
│  ├── on trade close:  pips gained/lost  (clipped ±50 pips)         │
│  ├── holding too long (>200 bars):  -0.01 per bar                  │
│  └── blowup (drawdown >10%):        -10.0 terminal penalty         │
│                                                                     │
│  Episode                                                            │
│  ├── length:  2,880 bars  (2 trading days of M1)                   │
│  └── start:   random position in dataset                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  (obs, reward, done)
                               └──────────────► back to PPO Agent
```

---

## Training Progress (actual results)

| Timesteps | Eval Reward (pips/2-day episode) |
|-----------|----------------------------------|
| 20K       | 0                                |
| 40K       | +1.5                             |
| 600K      | +766                             |
| 660K      | +884                             |
| 720K      | +1,018                           |
| 760K      | +1,075                           |
| 800K      | +1,113  ← best so far            |

---

## File Structure

```
aqrf_project/
│
├── process_data.py          ← Step 1: build feature arrays from raw CSVs
├── train_rl.py              ← Step 2: train PPO agent (single entry point)
│
├── EURUSD/
│   ├── ohlcv/
│   │   ├── 1min/            M1 OHLCV CSVs  (2016-2025)
│   │   ├── 5min/            M5
│   │   ├── 15min/           M15
│   │   ├── 1hour/           H1
│   │   └── 4hour/           H4
│   └── tick/
│       └── daily/           tick-level bid/ask/volume CSVs
│
├── data/processed/
│   ├── train_features.npy   (581291, 60, 32)
│   ├── train_prices.npy     (581291, 2)
│   ├── val_features.npy     ( 98129, 60, 32)
│   ├── val_prices.npy       ( 98129, 2)
│   ├── test_features.npy    (199939, 60, 32)
│   ├── test_prices.npy      (199939, 2)
│   └── norm_stats.npz       z-score mean/std from train split
│
├── src/
│   ├── rl/
│   │   ├── environment.py   ForexTradingEnv (Gymnasium)
│   │   └── agent.py         TCNExtractor + PPO factory
│   └── utils/
│       └── config.py        AQRFConfig (all hyperparameters)
│
└── checkpoints/rl/
    ├── best/                best eval checkpoint (auto-saved)
    └── ppo_forex_*_steps.zip  periodic checkpoints
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| No supervised prediction layer | TCN at 52% direction accuracy = coin flip. RL trading on a coin flip cannot learn. |
| Reward = pips per closed trade | Cumulative P&L over 15-day episodes drowned the signal. Sparse per-trade reward is clean. |
| Episode = 2 days (2,880 bars) | Short episodes → more resets → 7× more gradient signal per hour of training. |
| Multi-timeframe features (M1→H4) | Single timeframe has ~2% correlation with next bar. MTF adds trend context from H1/H4. |
| TCN inside PPO policy | TCN extracts temporal patterns end-to-end with RL gradient — no separate pre-training needed. |
| GroupNorm not BatchNorm | Financial sequences are non-stationary. GroupNorm normalises per-sample, stable without large batches. |
| Micro-lot (10,000 units) | Standard lot (100,000) = $10/pip — too large for $10K account, caused runaway losses early in training. |
