"""Event-driven backtesting engine for AQRF.

Processes: MarketEvent -> FeatureEvent -> SignalEvent -> RiskEvent -> OrderEvent -> FillEvent -> PortfolioUpdate
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Callable
from enum import Enum

import numpy as np
import torch
import polars as pl
import structlog

from src.utils.config import DataConfig, RiskConfig
from src.models.ensemble import ModelEnsemble
from src.risk.kelly import kelly_fraction
from src.risk.stops import calculate_stops
from src.risk.guard import approve_trade, TradeSignal, PortfolioState

logger = structlog.get_logger(__name__)


class EventType(Enum):
    MARKET = "market"
    FEATURE = "feature"
    SIGNAL = "signal"
    RISK = "risk"
    ORDER = "order"
    FILL = "fill"
    PORTFOLIO = "portfolio"


@dataclass
class Event:
    event_type: EventType
    timestamp: datetime
    data: Dict = field(default_factory=dict)


@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    pnl_pips: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_reason: str = ""
    regime: str = ""


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime],
        ensemble: ModelEnsemble,
        risk_config: Optional[RiskConfig] = None,
        initial_capital: float = 10000.0,
    ):
        self.data = data
        self.prices = prices  # [bid, ask]
        self.timestamps = timestamps
        self.ensemble = ensemble
        self.risk_config = risk_config or RiskConfig()
        self.initial_capital = initial_capital

        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_date: Optional[datetime] = None

        self.equity_curve: List[float] = [initial_capital]
        self.events: List[Event] = []

    def reset_daily(self, current_time: datetime) -> None:
        """Reset daily counters."""
        if self.last_date is not None and current_time.date() != self.last_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
        self.last_date = current_time.date()

    def run(self) -> Dict:
        """Run full backtest."""
        logger.info("backtest_start", bars=len(self.data), capital=self.initial_capital)

        for i in range(len(self.data)):
            timestamp = self.timestamps[i]
            self.reset_daily(timestamp)

            # Market event
            bid, ask = self.prices[i]

            # Check stop/tp for open position
            if self.current_trade is not None:
                exit_reason = None
                if self.position > 0:
                    if bid <= self.current_trade.stop_loss:
                        exit_reason = "stop_loss"
                    elif bid >= self.current_trade.take_profit:
                        exit_reason = "take_profit"
                elif self.position < 0:
                    if ask >= self.current_trade.stop_loss:
                        exit_reason = "stop_loss"
                    elif ask <= self.current_trade.take_profit:
                        exit_reason = "take_profit"

                if exit_reason:
                    self._close_position(timestamp, bid, ask, exit_reason)

            # Feature event
            features = self.data[i]
            features_tensor = torch.from_numpy(features).float().unsqueeze(0)

            # Signal event
            signal = self.ensemble.predict(features_tensor)

            if isinstance(signal, list):
                signal = signal[0]

            # Risk event / Confluence
            portfolio = PortfolioState(
                daily_loss=abs(self.daily_pnl) / self.initial_capital,
                trades_today=self.daily_trades,
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
                kelly_size=0.01,  # Placeholder
                rr=1.5,  # Placeholder
                magnitude=signal["magnitude"],
            )

            approved = approve_trade(trade_signal, portfolio, self.risk_config)

            # Order event
            if approved and self.position == 0:
                self._open_position(timestamp, bid, ask, signal)

            # Update equity
            unrealized = self._get_unrealized_pnl(bid, ask)
            self.equity_curve.append(self.capital + unrealized)
            self.peak_capital = max(self.peak_capital, self.equity_curve[-1])

        # Close any open position at end
        if self.current_trade is not None:
            self._close_position(
                self.timestamps[-1],
                self.prices[-1][0],
                self.prices[-1][1],
                "end_of_data",
            )

        results = self._calculate_metrics()
        logger.info("backtest_complete", **results)
        return results

    def _open_position(self, timestamp: datetime, bid: float, ask: float, signal: Dict) -> None:
        """Open new position."""
        direction = signal["direction"]
        vol = 0.001  # Placeholder realized vol

        if direction == "buy":
            entry = ask
            self.position = 1
        else:
            entry = bid
            self.position = -1

        size = 0.01  # Simplified sizing

        stop, tp, rr = calculate_stops(entry, direction, vol, config=self.risk_config)

        self.current_trade = Trade(
            entry_time=timestamp,
            direction=direction,
            entry_price=entry,
            size=size,
            stop_loss=stop,
            take_profit=tp,
            regime=signal["regime"],
        )

        self.entry_price = entry
        self.daily_trades += 1

        logger.info("position_opened", direction=direction, entry=entry, size=size)

    def _close_position(self, timestamp: datetime, bid: float, ask: float, reason: str) -> None:
        """Close current position."""
        if self.current_trade is None:
            return

        if self.position > 0:
            exit_price = bid
        else:
            exit_price = ask

        pnl = (exit_price - self.entry_price) * self.position * 100000 * self.current_trade.size
        pnl_pips = (exit_price - self.entry_price) * self.position / 0.0001

        self.current_trade.exit_time = timestamp
        self.current_trade.exit_price = exit_price
        self.current_trade.pnl = pnl
        self.current_trade.pnl_pips = pnl_pips
        self.current_trade.exit_reason = reason

        self.capital += pnl
        self.daily_pnl += pnl
        self.trades.append(self.current_trade)

        logger.info(
            "position_closed",
            reason=reason,
            pnl=pnl,
            pnl_pips=pnl_pips,
            capital=self.capital,
        )

        self.position = 0
        self.entry_price = 0.0
        self.current_trade = None

    def _get_unrealized_pnl(self, bid: float, ask: float) -> float:
        """Get unrealized PnL."""
        if self.position == 0:
            return 0.0
        price = bid if self.position > 0 else ask
        return (price - self.entry_price) * self.position * 100000 * 0.01

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {"error": "no_trades"}

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        total_return = (equity[-1] - self.initial_capital) / self.initial_capital

        # Annualized (assuming ~252 trading days, ~43,200 M1 bars per 30 days)
        n_bars = len(self.data)
        n_years = n_bars / (43_200 * 12)
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        # Sharpe
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)
        else:
            sharpe = 0.0

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252 * 24 * 60)
        else:
            sortino = 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(equity)
        drawdowns = (cummax - equity) / cummax
        max_drawdown = np.max(drawdowns)

        # Win rate
        pnls = [t.pnl for t in self.trades]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Avg R:R
        avg_rr = np.mean([t.take_profit / abs(t.stop_loss - t.entry_price) 
                         for t in self.trades if t.stop_loss != t.entry_price])

        return {
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(annualized_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_rr": round(avg_rr, 2),
            "total_trades": len(self.trades),
            "avg_trade_duration_bars": 0,  # Calculate from timestamps
            "final_capital": round(self.capital, 2),
        }
