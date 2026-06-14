"""Confluence filter / trade approval for AQRF.

All checks must pass for trade execution.
"""
from dataclasses import dataclass
from typing import Optional, Dict

import structlog

from src.utils.config import RiskConfig

logger = structlog.get_logger(__name__)


@dataclass
class TradeSignal:
    """Signal from model ensemble."""
    direction: str
    tcn_conf: float
    tf_conf: float
    tcn_dir: str
    tf_dir: str
    regime: str
    kelly_size: float
    rr: float
    magnitude: float


@dataclass
class PortfolioState:
    """Current portfolio state."""
    daily_loss: float
    trades_today: int
    current_position: int
    capital: float


def approve_trade(
    signal: TradeSignal,
    portfolio: PortfolioState,
    config: Optional[RiskConfig] = None,
) -> bool:
    """Approve or reject trade based on confluence rules.

    Args:
        signal: Trade signal from ensemble
        portfolio: Current portfolio state
        config: Risk configuration

    Returns:
        True if trade approved
    """
    config = config or RiskConfig()

    checks = {
        "tcn_confidence": signal.tcn_conf >= config.min_model_confidence,
        "transformer_confidence": signal.tf_conf >= config.min_model_confidence,
        "direction_agree": signal.tcn_dir == signal.tf_dir,
        "regime_ok": signal.regime != "volatile",
        "kelly_size": signal.kelly_size >= config.min_kelly_size,
        "rr_minimum": signal.rr >= config.min_rr_ratio,
        "daily_drawdown": portfolio.daily_loss < config.max_daily_drawdown,
        "daily_trades": portfolio.trades_today < config.max_daily_trades,
        "no_position": portfolio.current_position == 0,
    }

    all_pass = all(checks.values())

    logger.info(
        "trade_decision",
        approved=all_pass,
        direction=signal.direction,
        confidence=max(signal.tcn_conf, signal.tf_conf),
        regime=signal.regime,
        checks=checks,
    )

    return all_pass
