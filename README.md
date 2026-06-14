# AQRF — Autonomous Quant Research Firm

Production-grade Deep Reinforcement Learning Forex Trading System for EUR/USD.

## Architecture

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | Data Pipeline | Parse, validate, store, load Dukascopy tick data |
| 2 | Feature Engineering | Pure math features from price/volume (no indicators) |
| 3 | AI Models | TCN + Transformer + Regime Detector ensemble |
| 4 | RL Agent | PPO-trained autonomous trading agent |
| 5 | Risk Engine | Kelly sizing, stops, confluence guard |

## Hardware Optimized For

- **CPU:** Ryzen 7 7435HS (8C/16T)
- **RAM:** 16GB DDR5
- **GPU:** NVIDIA RTX 3050 Laptop (4GB VRAM, CUDA 11.8)
- **OS:** Windows 11 + WSL2 Ubuntu 22.04

## Quick Start

```bash
# 1. Infrastructure
docker-compose -f infrastructure/docker-compose.yml up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 4. Run data pipeline
python -m src.data.parser
python -m src.data.validator
python -m src.data.storage

# 5. Train models (Phase 4)
python -m src.models.trainer --model tcn
python -m src.models.trainer --model transformer
python -m src.models.trainer --model regime

# 6. Train RL agent (Phase 5)
python -m src.rl.trainer

# 7. Backtest (Phase 7)
python -m src.backtest.engine
```

## Project Structure

```
aqrf/
├── data/               # Raw and processed forex data
├── src/
│   ├── data/           # Layer 1: Data Pipeline
│   ├── features/       # Layer 2: Feature Engineering
│   ├── models/         # Layer 3: DL Models
│   ├── rl/             # Layer 4: RL Environment + Agent
│   ├── risk/           # Layer 5: Risk Engine
│   ├── backtest/       # Backtesting engine
│   ├── execution/      # Paper trading
│   └── utils/          # Config, logging, GPU, DB
├── configs/            # YAML hyperparameters
├── infrastructure/     # Docker Compose, DB init
├── checkpoints/        # Model weights
└── mlruns/             # MLflow experiments
```

## Key Design Decisions

- **No traditional indicators** — AlphaGo-style learning from raw price/volume
- **Strict chronological splits** — zero lookahead bias
- **4GB VRAM budget** — AMP, 8-bit Adam, gradient accumulation, memory mapping
- **TimescaleDB** — compressed hypertables for tick data
- **Polars** — 3-5x faster than pandas for CSV parsing

## License

Proprietary — AQRF Internal Use Only
