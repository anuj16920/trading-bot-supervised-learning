"""Paper trading module for AQRF.

Simulates live trading without real capital at risk.
Connects to data feed, runs ensemble + risk engine, logs trades.
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path

import numpy as np
import torch
import structlog

from src.models.ensemble import ModelEnsemble
from src.risk.kelly import kelly_fraction
from src.risk.stops import calculate_stops
from src.risk.guard import approve_trade, TradeSignal, PortfolioState
from src.utils.config import RiskConfig, DataConfig

logger = structlog.get_logger(__name__)


class PaperTrader:
    """Paper trading system for live simulation."""

    def __init__(
        self,
        ensemble: ModelEnsemble,
        risk_config: Optional[RiskConfig] = None,
        initial_capital: float = 10000.0,
        log_dir: Path = Path("./paper_trades"),
    ):
        self.ensemble = ensemble
        self.risk_config = risk_config or RiskConfig()
        self.initial_capital = initial_capital
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Portfolio state
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.peak_capital = initial_capital

        # Current trade
        self.current_trade: Optional[Dict] = None

        # Logging
        self.trade_log: list[Dict] = []

    def on_tick(
        self,
        timestamp: datetime,
        bid: float,
        ask: float,
        features: np.ndarray,
    ) -> Optional[Dict]:
        """Process new tick data.

        Args:
            timestamp: Current timestamp
            bid: Bid price
            ask: Ask price
            features: (seq_len, n_features) array

        Returns:
            Trade dict if trade executed, None otherwise
        """
        # Check stops for open position
        if self.current_trade is not None:
            exit_reason = self._check_stops(bid, ask)
            if exit_reason:
                return self._close_position(timestamp, bid, ask, exit_reason)

        # Get prediction
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        signal = self.ensemble.predict(features_tensor)

        if isinstance(signal, list):
            signal = signal[0]

        # Build trade signal for guard
        portfolio = PortfolioState(
            daily_loss=abs(self.daily_pnl) / self.initial_capital,
            trades_today=self.trades_today,
            current_position=self.position,
            capital=self.capital,
        )

        trade_signal = TradeSignal(
            direction=signal["direction"],
            tcn_conf=signal["tcn_conf"],
            tf_conf=signal["tf_conf"],
            tcn_dir=signal["direction"],
            tf_dir=signal["direction"],
            regime=signal["regime"],
            kelly_size=0.01,
            rr=1.5,
            magnitude=signal["magnitude"],
        )

        # Check approval
        approved = approve_trade(trade_signal, portfolio, self.risk_config)

        if approved and self.position == 0:
            return self._open_position(timestamp, bid, ask, signal)

        return None

    def _check_stops(self, bid: float, ask: float) -> Optional[str]:
        """Check if stop loss or take profit hit."""
        if self.current_trade is None:
            return None

        if self.position > 0:
            if bid <= self.current_trade["stop_loss"]:
                return "stop_loss"
            if bid >= self.current_trade["take_profit"]:
                return "take_profit"
        elif self.position < 0:
            if ask >= self.current_trade["stop_loss"]:
                return "stop_loss"
            if ask <= self.current_trade["take_profit"]:
                return "take_profit"

        return None

    def _open_position(
        self,
        timestamp: datetime,
        bid: float,
        ask: float,
        signal: Dict,
    ) -> Dict:
        """Open new position."""
        direction = signal["direction"]
        vol = 0.001

        if direction == "buy":
            entry = ask
            self.position = 1
        else:
            entry = bid
            self.position = -1

        stop, tp, rr = calculate_stops(entry, direction, vol, config=self.risk_config)

        trade = {
            "timestamp": timestamp,
            "direction": direction,
            "entry": entry,
            "stop_loss": stop,
            "take_profit": tp,
            "rr": rr,
            "regime": signal["regime"],
            "confidence": signal["confidence"],
        }

        self.current_trade = trade
        self.entry_price = entry
        self.trades_today += 1
        self.total_trades += 1

        logger.info("paper_trade_opened", **trade)
        return trade

    def _close_position(
        self,
        timestamp: datetime,
        bid: float,
        ask: float,
        reason: str,
    ) -> Dict:
        """Close current position."""
        if self.current_trade is None:
            return {}

        if self.position > 0:
            exit_price = bid
        else:
            exit_price = ask

        pnl = (exit_price - self.entry_price) * self.position * 100000 * 0.01
        pnl_pips = (exit_price - self.entry_price) * self.position / 0.0001

        self.capital += pnl
        self.daily_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

        trade_result = {
            "timestamp": timestamp,
            "direction": self.current_trade["direction"],
            "entry": self.entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "pnl_pips": pnl_pips,
            "reason": reason,
            "capital": self.capital,
        }

        self.trade_log.append(trade_result)
        self.current_trade = None
        self.position = 0
        self.entry_price = 0.0

        logger.info("paper_trade_closed", **trade_result)
        return trade_result

    def get_stats(self) -> Dict:
        """Get current trading statistics."""
        if not self.trade_log:
            return {"capital": self.capital, "trades": 0}

        pnls = [t["pnl"] for t in self.trade_log]
        wins = [p for p in pnls if p > 0]

        return {
            "capital": round(self.capital, 2),
            "peak_capital": round(self.peak_capital, 2),
            "drawdown_pct": round((self.peak_capital - self.capital) / self.peak_capital * 100, 2),
            "total_trades": len(self.trade_log),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 2) if pnls else 0,
            "total_pnl": round(sum(pnls), 2),
            "avg_trade_pnl": round(np.mean(pnls), 2),
        }
