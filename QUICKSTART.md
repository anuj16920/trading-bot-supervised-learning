# AQRF Quick Start Guide

## 🚀 Setup (One-Time)

### 1. Install Dependencies
```cmd
install_gpu.bat
```
**Note:** PyTorch download is 2.7GB and takes 15-20 minutes. Let it run!

### 2. Verify GPU
```cmd
venv\Scripts\activate
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
Should output: `NVIDIA GeForce RTX 3050 Laptop GPU`

### 3. Prepare Data (Copy CSVs)
```cmd
python scripts/prepare_data.py
```
Copies 1-minute and 1-hour data from `EURUSD/ohlcv/` to `data/EURUSD/`

### 4. Process Features (IMPORTANT - Prevents Memory Crash!)
```cmd
python prepare_features.py --stride 5
```
**This processes data in chunks to avoid crashing your system!**
- Reads CSV files one year at a time
- Computes features in batches
- Creates memory-mapped files for efficient training
- Takes ~10-20 minutes depending on stride

**Stride options:**
- `--stride 1`: Maximum data (slow, large files)
- `--stride 5`: Balanced (recommended, ~5GB)
- `--stride 10`: Fast training (smaller dataset)

---

## 🎯 Training Workflow

### Phase 1: Train Deep Learning Models

**Train TCN (smallest, ~270K params)**
```cmd
python train.py --model tcn
```
**Time:** ~30-60 minutes (with stride=5)

**Train Transformer (~1.2M params)**
```cmd
python train.py --model transformer
```
**Time:** ~1-2 hours

**Train Regime Detector**
```cmd
python train.py --model regime
```
**Time:** ~30-45 minutes

### Phase 2: Train RL Agent (Coming Soon)
```cmd
python -m src.rl.trainer
```
Trains PPO agent using the ensemble of models

### Phase 3: Backtest (Coming Soon)
```cmd
python -m src.backtest.engine
```

---

## 📊 Monitor Training

**TensorBoard:**
```cmd
tensorboard --logdir=tensorboard_logs
```
Open: http://localhost:6006

**MLflow:**
```cmd
# Already running from docker-compose
```
Open: http://localhost:5000

---

## 🗂️ Data Structure

Your data is in: `EURUSD/ohlcv/`
- `1min/` - 1-minute OHLCV bars (2016-2026)
- `1hour/` - 1-hour OHLCV bars
- `15min/`, `5min/`, `4hour/`, `1day/` - Other timeframes

**Format:** `datetime,open,high,low,close,volume,spread_avg,tick_count`

---

## ⚙️ Configuration

Edit YAML files in `configs/`:
- `data_config.yaml` - Data splits, validation thresholds
- `model_config.yaml` - Model architectures, training params
- `risk_config.yaml` - Kelly sizing, stop loss, confluence rules
- `rl_config.yaml` - PPO hyperparameters, reward shaping

---

## 🔧 Troubleshooting

**Out of Memory (OOM)?**
- Reduce `batch_size` in `configs/model_config.yaml`
- Enable `use_gradient_checkpointing: true` for Transformer

**CUDA not found?**
```cmd
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Docker issues?**
- Make sure Docker Desktop is running
- Check ports 5432, 6379, 5000 are not in use

---

## 📈 Expected Timeline

- **Data Preparation:** 5-10 minutes
- **TCN Training:** 2-4 hours (100 epochs)
- **Transformer Training:** 3-6 hours
- **Regime Training:** 1-2 hours
- **RL Training:** 8-12 hours (10M timesteps)
- **Backtesting:** 10-30 minutes

**Total:** ~1-2 days for full pipeline

---

## 🎓 Project Structure

```
aqrf_project/
├── data/EURUSD/          # Processed data (M1, H1)
├── EURUSD/ohlcv/         # Raw data (source)
├── src/
│   ├── data/             # Data pipeline
│   ├── features/         # Feature engineering
│   ├── models/           # DL models (TCN, Transformer, Regime)
│   ├── rl/               # RL agent (PPO)
│   ├── risk/             # Risk management
│   ├── backtest/         # Backtesting engine
│   └── utils/            # Config, logging, GPU utils
├── configs/              # YAML configurations
├── checkpoints/          # Saved model weights
├── mlruns/               # MLflow experiments
└── tensorboard_logs/     # TensorBoard logs
```

---

## 💡 Tips

1. **Start small:** Train TCN first to verify everything works
2. **Monitor VRAM:** Use `nvidia-smi` to check GPU usage
3. **Save checkpoints:** Models auto-save every 100k steps
4. **Use MLflow:** Track all experiments automatically
5. **Adjust configs:** Tune hyperparameters in YAML files

---

## 🆘 Need Help?

Check the logs:
- Training logs: Console output
- MLflow: http://localhost:5000
- TensorBoard: http://localhost:6006
- Docker logs: `docker-compose logs`
