-- AQRF Database Initialization
-- TimescaleDB hypertable setup for forex tick data

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Tick data table
CREATE TABLE IF NOT EXISTS tick_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bid DOUBLE PRECISION NOT NULL,
    ask DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    spread_pips DOUBLE PRECISION GENERATED ALWAYS AS ((ask - bid) / 0.0001) STORED
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable('tick_data', 'time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

-- OHLCV table
CREATE TABLE IF NOT EXISTS ohlcv_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('ohlcv_data', 'time', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Features table (pre-computed windows)
CREATE TABLE IF NOT EXISTS feature_windows (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    window_id BIGINT NOT NULL,
    features JSONB NOT NULL,
    regime TEXT,
    target_direction INTEGER,
    target_magnitude DOUBLE PRECISION
);

SELECT create_hypertable('feature_windows', 'time', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Training metadata
CREATE TABLE IF NOT EXISTS training_runs (
    run_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    phase TEXT NOT NULL,
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    config JSONB,
    metrics JSONB,
    checkpoint_path TEXT,
    status TEXT DEFAULT 'running'
);

-- Trade logs for backtesting
CREATE TABLE IF NOT EXISTS trade_logs (
    trade_id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES training_runs(run_id),
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    size DOUBLE PRECISION NOT NULL,
    pnl DOUBLE PRECISION,
    pnl_pips DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    exit_reason TEXT,
    regime TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_tick_symbol_time ON tick_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv_data (symbol, timeframe, time DESC);
CREATE INDEX IF NOT EXISTS idx_features_window ON feature_windows (window_id);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trade_logs (run_id);

-- Enable compression on older chunks
ALTER TABLE tick_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'time DESC'
);

ALTER TABLE ohlcv_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe',
    timescaledb.compress_orderby = 'time DESC'
);

-- Compression policy: compress chunks older than 7 days
SELECT add_compression_policy('tick_data', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('ohlcv_data', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy: drop raw tick data older than 2 years (keep OHLCV)
SELECT add_retention_policy('tick_data', INTERVAL '2 years', if_not_exists => TRUE);
