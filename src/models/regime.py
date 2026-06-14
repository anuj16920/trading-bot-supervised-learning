"""Regime detection model for AQRF.

LSTM-based classifier: trending_up, trending_down, ranging, volatile.
Uses ADX + MA for labeling training data only (not as features).
"""
from typing import Optional

import torch
import torch.nn as nn
import structlog

from src.utils.config import RegimeConfig

logger = structlog.get_logger(__name__)


class RegimeDetector(nn.Module):
    """LSTM-based market regime classifier."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        super().__init__()

        config = config or RegimeConfig()
        self.config = config

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_regimes),
        )

        self._init_weights()
        self._log_params()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _log_params(self) -> None:
        """Log parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("regime_detector_initialized", total_params=total, trainable_params=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len=100, n_features=45)

        Returns:
            logits: (batch, num_regimes=4)
        """
        # LSTM
        _, (hidden, _) = self.lstm(x)

        # Take last layer hidden state
        last_hidden = hidden[-1]  # (batch, hidden_size)

        # Classify
        logits = self.classifier(last_hidden)

        return logits


def generate_regime_labels(
    df,
    config: Optional[RegimeConfig] = None,
):
    """Generate regime labels using ADX + MA (for training data only).

    Args:
        df: DataFrame with OHLCV data
        config: Regime configuration

    Returns:
        Series of regime labels (0-3)
    """
    import numpy as np
    import polars as pl

    config = config or RegimeConfig()

    # Calculate ADX (simplified)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    # +DM / -DM
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0), 0)

    # Smoothed (14-period)
    period = 14
    atr = np.convolve(tr, np.ones(period)/period, mode='valid')
    plus_di = 100 * np.convolve(plus_dm, np.ones(period)/period, mode='valid') / atr
    minus_di = 100 * np.convolve(minus_dm, np.ones(period)/period, mode='valid') / atr

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx = np.convolve(dx, np.ones(period)/period, mode='valid')

    # Pad to original length
    pad_len = len(close) - len(adx)
    adx = np.concatenate([np.full(pad_len, adx[0]), adx])

    # Moving average
    ma50 = df["close"].rolling_mean(window_size=50, min_periods=50).to_numpy()

    # Realized volatility
    returns = df["close"].pct_change().to_numpy()
    vol = df["close"].rolling_std(window_size=20, min_periods=20).to_numpy()
    vol_mean = np.nanmean(vol)
    vol_std = np.nanstd(vol)

    # Classify
    labels = []
    for i in range(len(close)):
        if np.isnan(adx[i]) or np.isnan(ma50[i]):
            labels.append(2)  # ranging as default
            continue

        if adx[i] > config.adx_trending:
            if close[i] > ma50[i]:
                labels.append(0)  # trending_up
            else:
                labels.append(1)  # trending_down
        elif adx[i] < config.adx_ranging:
            labels.append(2)  # ranging
        elif not np.isnan(vol[i]) and vol[i] > vol_mean + config.vol_threshold_sigma * vol_std:
            labels.append(3)  # volatile
        else:
            labels.append(2)  # default ranging

    return pl.Series("regime", labels)
