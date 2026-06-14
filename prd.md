I am building a production-grade Deep Reinforcement Learning 
Forex Trading System for EUR/USD from scratch. This is inspired 
by Marshall Chang's TEDx talk on AI traders and AlphaGo's 
reinforcement learning approach applied to financial markets.

════════════════════════════════════════════════
PROJECT NAME: AQRF (Autonomous Quant Research Firm)
════════════════════════════════════════════════

GOAL:
Build a fully autonomous AI trading agent that learns to trade 
EUR/USD profitably using Deep Reinforcement Learning trained on 
10 years of real Dukascopy tick data. The agent must learn 
entirely from price and volume numbers — no traditional 
indicators. Like AlphaGo learned Go from scratch, this agent 
learns trading from scratch through millions of simulated trades.

════════════════════════════════════════════════
HARDWARE (optimize everything for this exactly):
════════════════════════════════════════════════
CPU: Ryzen 7 7435HS (8 cores / 16 threads)
RAM: 16GB DDR5
GPU: NVIDIA RTX 3050 Laptop (4GB VRAM, CUDA 11.8)
SSD: 512GB NVMe
OS: Windows 11 with WSL2 Ubuntu 22.04

════════════════════════════════════════════════
GPU TRAINING SPECIFICATIONS:
════════════════════════════════════════════════

DEVICE SETUP:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  Always verify at startup:
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"VRAM Free: {torch.cuda.memory_reserved(0) / 1e9:.1f}GB")

CUDA OPTIMIZATIONS (apply all):
  torch.backends.cudnn.benchmark = True
    (auto-find fastest conv algorithms for fixed input sizes)
  
  torch.backends.cudnn.deterministic = False
    (faster, slight non-determinism is acceptable)
  
  torch.backends.cuda.matmul.allow_tf32 = True
    (TF32 for matrix multiply, faster on Ampere GPUs)
  
  torch.backends.cudnn.allow_tf32 = True
    (TF32 for convolutions)

AUTOMATIC MIXED PRECISION (AMP) — mandatory:
  Use torch.cuda.amp for all DL model training
  
  scaler = torch.cuda.amp.GradScaler()
  
  Training step pattern:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(batch)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
  
  Benefits:
    FP16 forward pass (2x faster, half VRAM)
    FP32 gradients (numerical stability)
    Fits larger batches in 4GB VRAM

BATCH SIZE TARGETS FOR RTX 3050 4GB:
  TCN model:          batch_size=256  (target ~1.2GB VRAM)
  Transformer model:  batch_size=128  (target ~1.5GB VRAM)
  Regime detector:    batch_size=256  (target ~0.8GB VRAM)
  RL policy network:  batch_size=64   (target ~0.5GB VRAM)
  
  Always leave ~800MB VRAM headroom for:
    CUDA kernels overhead
    cuDNN workspace
    Gradient buffers
  
  Dynamic batch size finder:
    If OOM error occurs: halve batch_size and retry
    Log final batch_size used to MLflow

DATALOADER GPU OPTIMIZATION (mandatory on all loaders):
  DataLoader(
      dataset,
      batch_size=batch_size,
      num_workers=4,           # 4 CPU cores for data prep
      pin_memory=True,         # page-locked RAM → faster GPU transfer
      prefetch_factor=2,       # pre-load 2 batches per worker
      persistent_workers=True, # keep workers alive between epochs
      shuffle=True,            # shuffle windows (not raw sequence)
  )

════════════════════════════════════════════════
MEMORY MANAGEMENT SPECIFICATIONS:
════════════════════════════════════════════════

RAM MANAGEMENT (16GB total budget):
  
  Budget allocation:
    OS + system processes:    ~3GB
    TimescaleDB:              ~2GB
    Redis:                    ~1GB
    Python data pipeline:     ~4GB
    PyTorch training:         ~4GB
    Headroom:                 ~2GB
    Total:                    16GB
  
  CSV chunked loading (never load full file):
    chunk_size = 500_000 rows
    Estimated RAM per chunk: ~50MB for tick data
    Process chunk → insert DB → del chunk → gc.collect()
    
    Pattern:
      import gc
      for chunk in pl.read_csv_batched(file, batch_size=500_000):
          df = chunk.to_pandas()
          await storage.insert_batch(df)
          del df, chunk
          gc.collect()
  
  Training dataset RAM management:
    Never load full dataset into RAM
    Use PyTorch Dataset with lazy loading:
      __init__: store only metadata and DB connection
      __getitem__: query single window from TimescaleDB on demand
      Cache last 10,000 windows in RAM (LRU cache)
      
    Alternative for speed: 
      Pre-extract windows to memory-mapped numpy arrays
      np.memmap: lives on disk, accessed like RAM
      Zero RAM overhead, fast random access
      
    Implementation:
      windows_mmap = np.memmap(
          'data/windows.npy',
          dtype='float32',
          mode='r',
          shape=(n_samples, seq_len, n_features)
      )

VRAM MANAGEMENT (4GB budget):
  
  Budget allocation per training run:
    Model weights:      ~200-500MB depending on model
    Activations:        ~800MB-1.2GB (forward pass)
    Gradients:          ~200-500MB (same size as weights)
    Optimizer states:   ~400MB-1GB (Adam: 2x weight size)
    Batch data:         ~50-200MB
    cuDNN workspace:    ~200-400MB
    Total target:       stay under 3.5GB (leave 500MB buffer)
  
  VRAM monitoring (run every epoch):
    def log_vram():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"VRAM allocated: {allocated:.2f}GB / reserved: {reserved:.2f}GB")
        if allocated > 3.5:
            warnings.warn("VRAM usage > 3.5GB, risk of OOM")
  
  Gradient accumulation (simulate large batches):
    If batch_size=256 causes OOM:
      Use batch_size=64 with gradient_accumulation_steps=4
      Mathematically equivalent to batch_size=256
      
      Pattern:
        optimizer.zero_grad()
        for i, (x, y) in enumerate(loader):
            with autocast():
                loss = model(x, y) / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
  
  Explicit VRAM release after each phase:
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    (call this after training each model before loading next)
  
  Gradient checkpointing for Transformer:
    from torch.utils.checkpoint import checkpoint
    Use on transformer layers to trade compute for memory
    Reduces activation memory by ~60% at cost of ~30% slower
    Only use if needed for Transformer to fit in 4GB
  
  In-place operations where safe:
    Use relu_(x) instead of relu(x)
    Use add_(x) instead of + x
    Saves activation memory

OPTIMIZER MEMORY OPTIMIZATION:
  Use AdamW (not Adam) — same performance, better weight decay
  
  For memory-critical training use 8-bit Adam:
    pip install bitsandbytes
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    Reduces optimizer state memory by 75%
    (from ~1GB to ~250MB for typical model)

MODEL WEIGHT PRECISION:
  Store weights in FP32 (training stability)
  Inference in FP16 (2x faster, half VRAM):
    model.half()  # convert to FP16 for inference only
    x = x.half().to(device)
  
  After training save both:
    torch.save(model.state_dict(), 'model_fp32.pth')
    torch.save(model.half().state_dict(), 'model_fp16.pth')
    (use fp16 for backtesting/paper trading inference)

SSD MANAGEMENT:
  Total data budget on 512GB SSD:
    Raw CSV data:           ~6GB
    TimescaleDB data:       ~15GB (indexed, compressed)
    Numpy memory maps:      ~8GB (pre-computed windows)
    Model checkpoints:      ~2GB (all experiments)
    MLflow artifacts:       ~3GB
    Redis AOF persistence:  ~1GB
    Total:                  ~35GB (well within 512GB)
  
  TimescaleDB compression:
    Enable chunk compression after 7 days
    Typical compression ratio: 10:1 for tick data
    15GB uncompressed → ~1.5GB compressed older data

════════════════════════════════════════════════
DATA:
════════════════════════════════════════════════
Source: Dukascopy historical EUR/USD
Format: CSV files
Size: ~5.6GB

Tick data structure:
  timestamp, bid, ask, volume
  2015-01-02 00:00:01.234, 1.21034, 1.21036, 0.5

OHLCV structure:
  timestamp, open, high, low, close, volume

Timeframes available: M1, H1

Folder structure:
  data/
    EURUSD/
      tick/
        2015/
          EURUSD_2015_01.csv
      M1/
        EURUSD_M1_2015.csv
      H1/
        EURUSD_H1_2015.csv

Data split (strictly chronological — never random):
  Training:   2015-2021 (70%)
  Validation: 2022 (10%)
  Test:       2023-2024 (20%)

════════════════════════════════════════════════
COMPLETE SYSTEM ARCHITECTURE (5 layers):
════════════════════════════════════════════════

LAYER 1 — DATA PIPELINE

  parser.py:
    Use Polars for all CSV reading (3-5x faster than pandas)
    chunk_size = 500_000 rows
    Parse timestamp as datetime with millisecond precision
    Cast bid/ask/volume to float32 (not float64 — saves RAM)
    Validate on read: drop rows where bid <= 0 or ask <= bid
    Progress bar via tqdm
    Log rows processed, rows dropped, time taken
  
  validator.py:
    Gap detection: flag gaps > 1 hour in tick data
    Spread validation: drop ticks where spread > 10 pips
    Volume validation: drop ticks where volume <= 0
    Duplicate detection: drop duplicate timestamps
    Price sanity: drop ticks where price change > 500 pips
    Generate validation report: 
      total_rows, dropped_rows, gap_count, gap_locations
  
  storage.py:
    Async writes using asyncpg connection pool
    Pool size: min=2, max=10
    Batch insert: 10,000 rows per INSERT statement
    Use COPY protocol for bulk loading (fastest method)
    Retry logic: 3 retries with exponential backoff
  
  loader.py:
    query_tick_range(start, end, columns) → pl.DataFrame
    query_ohlcv_range(start, end, timeframe) → pl.DataFrame
    Returns Polars DataFrames (lazy evaluation)
    Connection pool: reuse across training
  
  splitter.py:
    split_chronological(df, train=0.70, val=0.10, test=0.20)
    Returns: (train_df, val_df, test_df)
    Verify no time overlap between splits
    Log split statistics: rows, date ranges, % of total

LAYER 2 — FEATURE ENGINEERING

  ALL features are pure math from price/volume only.
  Zero traditional indicators.
  Zero lookahead bias (all rolling windows use only past data).

  tick_features.py:
    mid_price = (bid + ask) / 2
    spread_pips = (ask - bid) / pip_size
    log_return = np.log(mid_t / mid_t-1)
    tick_direction = np.sign(log_return)
    volume_raw = volume
    
  ohlcv_features.py:
    log_return_open_close = log(close / open)
    log_return_close_close = log(close_t / close_t-1)
    high_low_range = (high - low) / pip_size
    realized_vol_20 = rolling_std(log_returns, 20)
    realized_vol_60 = rolling_std(log_returns, 60)
    realized_vol_200 = rolling_std(log_returns, 200)
    price_velocity_5 = (close - close_5_ago) / 5
    price_velocity_20 = (close - close_20_ago) / 20
    price_acceleration = velocity_5 - velocity_5_one_bar_ago
    order_flow_imbalance = (up_vol - down_vol) / total_vol
    autocorr_lag1 = rolling_corr(returns_t, returns_t-1, 50)
    autocorr_lag5 = rolling_corr(returns_t, returns_t-5, 50)
    autocorr_lag10 = rolling_corr(returns_t, returns_t-10, 50)
    rolling_mean_spread = rolling_mean(spread, 20)
    vwap_deviation = (close - vwap) / realized_vol
    
  mtf_features.py:
    Compute all ohlcv_features at: M1, M5, M15, H1, H4
    Resample M1 data to get M5, M15, H4 (no separate data needed)
    Align all timeframes to M1 index (forward fill higher TFs)
    Total features: ~45 across all timeframes
    
  pipeline.py:
    Normalization: rolling z-score with window=500 bars
      z = (x - rolling_mean(x, 500)) / rolling_std(x, 500)
      This prevents lookahead bias (uses only past data)
    Clip outliers: z-score > 5 → clip to 5, < -5 → clip to -5
    Output: numpy array shape (n_bars, 45) float32
    Save as memory-mapped file for fast training access

LAYER 3 — AI MODELS

  tcn.py — Temporal Convolutional Network:
    
    class TCNBlock(nn.Module):
      Conv1d(in_ch, out_ch, kernel=3, dilation=d, padding=d)
      BatchNorm1d
      GELU activation (better than ReLU for sequences)
      Dropout(0.2)
      Residual connection with 1x1 conv if channels differ
    
    class TCN(nn.Module):
      Input: (batch, seq=60, features=45)
      Permute to (batch, features=45, seq=60) for Conv1d
      
      Block 1: channels=64,  dilation=1
      Block 2: channels=64,  dilation=2
      Block 3: channels=128, dilation=4
      Block 4: channels=128, dilation=8
      Block 5: channels=256, dilation=16
      
      Global average pooling → (batch, 256)
      Dense(256, 128) + GELU + Dropout(0.2)
      
      Two output heads:
        direction_head: Dense(128, 2) + Softmax
          output: [p_up, p_down]
        magnitude_head: Dense(128, 1)
          output: expected pip move (regression)
    
    Total parameters: ~850K (fits easily in 4GB VRAM)
    
  transformer.py — Attention Model:
    
    class ForexTransformer(nn.Module):
      Input: (batch, seq=60, features=45)
      
      Input projection: Linear(45, d_model=128)
      Learnable positional encoding: nn.Embedding(60, 128)
      
      4x TransformerEncoderLayer:
        d_model=128
        nhead=8
        dim_feedforward=512
        dropout=0.1
        activation='gelu'
        batch_first=True
      
      CLS token: prepended learnable token
      Take CLS output: (batch, 128)
      Dense(128, 64) + GELU + Dropout(0.1)
      
      Two output heads (same as TCN):
        direction_head: Dense(64, 2) + Softmax
        magnitude_head: Dense(64, 1)
    
    Total parameters: ~1.2M
    Use gradient checkpointing if VRAM pressure
    
  regime.py — Regime Detector:
    
    class RegimeDetector(nn.Module):
      Input: (batch, seq=100, features=45)
      
      LSTM(input=45, hidden=128, layers=2, dropout=0.2)
      Take last hidden state: (batch, 128)
      Dense(128, 64) + ReLU
      Dense(64, 4) + Softmax
      Output: [p_trending_up, p_trending_down, p_ranging, p_volatile]
    
    Regime labels for training (rule-based):
      Use ADX + price vs MA only for LABELING training data
      (indicators used here for label generation only, not features)
      trending_up:   ADX > 25 AND close > MA50
      trending_down: ADX > 25 AND close < MA50
      ranging:       ADX < 20
      volatile:      realized_vol > mean_vol + 2*std_vol
    
  ensemble.py:
    
    class ModelEnsemble:
      Loads all 3 trained models
      Moves all to device (cuda)
      Sets all to eval() mode
      
      predict(features_tensor):
        with torch.no_grad():
          with torch.cuda.amp.autocast():
            tcn_out = tcn_model(features)
            tf_out = transformer_model(features)
            regime_out = regime_model(features_100)
        
        regime = argmax(regime_out)
        regime_weight = {
          trending_up:   {'up': 1.2, 'down': 0.8},
          trending_down: {'up': 0.8, 'down': 1.2},
          ranging:       {'up': 1.0, 'down': 1.0},
          volatile:      {'up': 0.5, 'down': 0.5}
        }
        
        final_up = (0.4*tcn_up + 0.4*tf_up) * regime_weight
        confidence = max(final_up, 1-final_up)
        
        return {
          'direction': 'up' if final_up > 0.5 else 'down',
          'confidence': confidence,
          'magnitude': 0.5*(tcn_mag + tf_mag),
          'regime': regime
        }
  
  trainer.py — Unified DL Training Loop:
    
    class DLTrainer:
      
      setup():
        model.to(device)
        scaler = GradScaler()
        optimizer = AdamW8bit(lr=1e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingWarmRestarts(T_0=10)
        
      train_epoch():
        model.train()
        for batch_x, batch_y in dataloader:
          batch_x = batch_x.to(device, non_blocking=True)
          batch_y = batch_y.to(device, non_blocking=True)
          
          optimizer.zero_grad(set_to_none=True)
          
          with autocast():
            direction_pred, magnitude_pred = model(batch_x)
            dir_loss = cross_entropy(direction_pred, batch_y_dir)
            mag_loss = mse_loss(magnitude_pred, batch_y_mag)
            loss = dir_loss + 0.3 * mag_loss
          
          scaler.scale(loss).backward()
          scaler.unscale_(optimizer)
          clip_grad_norm_(model.parameters(), 1.0)
          scaler.step(optimizer)
          scaler.update()
          
          del batch_x, batch_y
          
        log_vram()
        scheduler.step()
        
      after_training():
        del optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()
        
      MLflow logging every epoch:
        train_loss, val_loss, val_accuracy
        learning_rate, vram_usage, epoch_time

LAYER 4 — RL ENVIRONMENT + AGENT

  environment.py — ForexTradingEnv(gym.Env):
    
    Observation space:
      Box(low=-5, high=5, shape=(60, 50), dtype=np.float32)
      60 timesteps × (45 market features + 5 portfolio features)
      
      Portfolio features (appended each step):
        position:          -1, 0, or 1
        unrealized_pnl:    normalized by account size
        current_drawdown:  0 to 1
        bars_in_trade:     normalized 0 to 1
        available_capital: normalized 0 to 1
    
    Action space: Discrete(6)
      0: Hold / Do nothing
      1: Buy full size (Kelly)
      2: Buy half size (half Kelly)
      3: Sell full size (Kelly)
      4: Sell half size (half Kelly)
      5: Close current position
    
    reset():
      Sample random episode start from training period
      Episode length: 30 days of M1 data (~43,200 steps)
      Initialize portfolio: capital=10000, position=flat
      Return initial observation
    
    step(action):
      Get current bid/ask from tick data
      Execute action with realistic fills:
        Buy:  fill at ask price (pay spread)
        Sell: fill at bid price (pay spread)
        Slippage: add 0.5 pip fixed
      
      Calculate reward:
        pnl = (current_price - entry_price) × position × lot_size
        vol = realized_vol_20 at current step
        sharpe_reward = pnl / max(vol, 1e-8)
        
        drawdown_penalty = -2.0 if drawdown > 0.02 else 0
        overtrade_penalty = -0.1 if trades_today > 10 else 0
        hold_penalty = -0.01 if bars_in_trade > 100 else 0
        
        total_reward = sharpe_reward + drawdown_penalty + 
                       overtrade_penalty + hold_penalty
      
      Termination conditions:
        done = True if:
          end of episode data
          drawdown > 5%
          account < 8000 (lost 20%)
      
      Return: obs, reward, done, truncated, info
    
    info dict (for logging):
      current_pnl, drawdown, n_trades, win_rate,
      current_position, current_price, regime
    
  agent.py — PPO Agent:
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],
            vf=[256, 256, 128]
        ),
        activation_fn=nn.GELU,
    )
    
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=linear_schedule(3e-4),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        policy_kwargs=policy_kwargs,
        tensorboard_log='./tensorboard_logs/',
        device='cuda',
        verbose=1,
    )
    
  trainer.py — RL Training:
    
    Vectorized environments:
      n_envs = 4 (one per CPU core pair)
      vec_env = SubprocVecEnv([make_env(i) for i in range(4)])
      vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
      (VecNormalize: normalizes observations and rewards automatically)
    
    Callbacks:
      EvalCallback:
        eval_env = separate validation environment
        eval_freq = 10_000 steps
        n_eval_episodes = 10
        best_model_save_path = './checkpoints/'
        deterministic = True
      
      CheckpointCallback:
        save_freq = 100_000 steps
        save_path = './checkpoints/'
        name_prefix = 'ppo_forex'
      
      StopTrainingOnNoModelImprovement:
        max_no_improvement_evals = 20
        min_evals = 50
    
    Training:
      model.learn(
          total_timesteps=10_000_000,
          callback=[eval_cb, checkpoint_cb, stop_cb],
          progress_bar=True,
      )
    
    After training:
      vec_env.close()
      torch.cuda.empty_cache()
      gc.collect()

LAYER 5 — RISK ENGINE

  kelly.py:
    def kelly_fraction(win_prob, rr_ratio):
      q = 1 - win_prob
      f = (win_prob * rr_ratio - q) / rr_ratio
      half_kelly = f * 0.5
      return max(0, min(half_kelly, 0.02))
      (never risk more than 2% of capital per trade)
    
  stops.py:
    def calculate_stops(entry, direction, realized_vol, rr=1.5):
      stop_distance = 2 * realized_vol * pip_value
      tp_distance = stop_distance * rr
      
      if direction == 'buy':
        stop_loss = entry - stop_distance
        take_profit = entry + tp_distance
      else:
        stop_loss = entry + stop_distance
        take_profit = entry - tp_distance
      
      actual_rr = tp_distance / stop_distance
      assert actual_rr >= 1.5, "R:R below minimum"
      return stop_loss, take_profit, actual_rr
    
  guard.py — Confluence Filter:
    def approve_trade(signal, portfolio):
      checks = {
        'tcn_confidence':    signal.tcn_conf > 0.60,
        'transformer_conf':  signal.tf_conf > 0.60,
        'direction_agree':   signal.tcn_dir == signal.tf_dir,
        'regime_ok':         signal.regime != 'volatile',
        'kelly_size':        signal.kelly_size > 0.005,
        'rr_minimum':        signal.rr >= 1.5,
        'daily_drawdown':    portfolio.daily_loss < 0.02,
        'daily_trades':      portfolio.trades_today < 15,
      }
      all_pass = all(checks.values())
      log_trade_decision(checks, all_pass)
      return all_pass

════════════════════════════════════════════════
BACKTESTING ENGINE:
════════════════════════════════════════════════
  Event-driven:
    MarketEvent → FeatureEvent → SignalEvent →
    RiskEvent → OrderEvent → FillEvent → PortfolioUpdate
  
  Tick-level fills:
    Long entry = ask price
    Short entry = bid price
    Spread included automatically
    Slippage = 0.5 pip fixed
  
  Walk-forward rounds:
    Round 1: Train 2015-2017, Test 2018
    Round 2: Train 2015-2018, Test 2019
    Round 3: Train 2015-2019, Test 2020
    Round 4: Train 2015-2020, Test 2021
    Round 5: Train 2015-2021, Test 2022
    Final:   Train 2015-2022, Test 2023-2024
  
  Required metrics output:
    Total return (%)
    Annualized return (%)
    Sharpe ratio (target > 1.5)
    Sortino ratio
    Max drawdown % (target < 10%)
    Win rate % (target > 52%)
    Profit factor (target > 1.3)
    Average R:R achieved
    Total number of trades
    Average trade duration
    Best month / worst month
    Monthly returns heatmap (12 × n_years grid)

════════════════════════════════════════════════
TECH STACK (exact versions):
════════════════════════════════════════════════
Python:              3.11
PyTorch:             2.1.0+cu118
Stable-Baselines3:   2.2.1
Gymnasium:           0.29.1
Polars:              0.20.0
Pandas:              2.1.0
NumPy:               1.26.0
bitsandbytes:        0.41.0  (8-bit optimizer)
SQLAlchemy:          2.0.0 async
asyncpg:             0.29.0
Redis:               5.0.0
FastAPI:             0.104.0
Uvicorn:             0.24.0
Pydantic:            2.5.0
Structlog:           23.2.0
MLflow:              2.9.0
tqdm:                4.66.0
pytest:              7.4.0
Docker Compose:      3.8
TimescaleDB:         latest (PostgreSQL 15)

════════════════════════════════════════════════
PROJECT STRUCTURE:
════════════════════════════════════════════════

aqrf/
├── data/
│   └── EURUSD/
│       ├── tick/
│       ├── M1/
│       └── H1/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── validator.py
│   │   ├── storage.py
│   │   ├── loader.py
│   │   └── splitter.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tick_features.py
│   │   ├── ohlcv_features.py
│   │   ├── mtf_features.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tcn.py
│   │   ├── transformer.py
│   │   ├── regime.py
│   │   ├── ensemble.py
│   │   └── trainer.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── environment.py
│   │   ├── agent.py
│   │   ├── callbacks.py
│   │   └── trainer.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── kelly.py
│   │   ├── stops.py
│   │   └── guard.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── simulator.py
│   │   └── analytics.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── paper_trader.py
│   │   └── order_manager.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── db.py
│       └── gpu.py
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── rl_config.yaml
│   └── risk_config.yaml
├── infrastructure/
│   ├── docker-compose.yml
│   ├── postgres/
│   │   └── init.sql
│   └── redis/
│       └── redis.conf
├── notebooks/
├── tests/
├── checkpoints/
├── mlruns/
├── tensorboard_logs/
├── .env
├── requirements.txt
└── README.md

════════════════════════════════════════════════
CODING STANDARDS:
════════════════════════════════════════════════
- Type hints on every single function
- Pydantic v2 models for all configs
- Async/await for all DB and I/O
- Structured JSON logging via structlog
- Docstring on every class and function
- Zero magic numbers (everything in config YAML)
- Error handling + retry on every external call
- Unit tests for data pipeline and features
- Zero lookahead bias (enforced, not assumed)
- GPU memory logged every epoch
- del + empty_cache() after every training phase
- set_to_none=True on zero_grad() always
- non_blocking=True on all .to(device) calls
- float32 for training, float16 for inference

════════════════════════════════════════════════
BUILD ORDER:
════════════════════════════════════════════════
Phase 1: Infrastructure
  docker-compose.yml, init.sql, redis.conf,
  .env template, requirements.txt, gpu.py

Phase 2: Data Pipeline
  parser.py, validator.py, storage.py,
  loader.py, splitter.py

Phase 3: Feature Engineering
  tick_features.py, ohlcv_features.py,
  mtf_features.py, pipeline.py

Phase 4: DL Models
  tcn.py, transformer.py, regime.py,
  ensemble.py, trainer.py

Phase 5: RL System
  environment.py, agent.py,
  callbacks.py, trainer.py

Phase 6: Risk Engine
  kelly.py, stops.py, guard.py

Phase 7: Backtesting
  engine.py, simulator.py, analytics.py

Phase 8: Paper Trading
  paper_trader.py, order_manager.py

════════════════════════════════════════════════
RULES FOR CLAUDE CODE:
════════════════════════════════════════════════
- Build one phase at a time
- Each file must be complete and runnable
- No placeholder functions or TODO comments
- Test each phase before moving to next
- Ask me before starting each new phase
- If a design decision is needed, ask me
- All GPU code must handle OOM gracefully
- All async code must handle connection errors
- Production grade only — no toy implementations