"""Stop loss and take profit calculation for AQRF.

Volatility-based stops with minimum R:R enforcement.
"""
from typing import Optional, Tuple

import structlog

from src.utils.config import RiskConfig, DataConfig

logger = structlog.get_logger(__name__)


def calculate_stops(
    entry: float,
    direction: str,
    realized_vol: float,
    pip_value: float = 0.0001,
    config: Optional[RiskConfig] = None,
) -> Tuple[float, float, float]:
    """Calculate stop loss and take profit levels.

    Args:
        entry: Entry price
        direction: 'buy' or 'sell'
        realized_vol: Current realized volatility
        pip_value: Pip value for the pair
        config: Risk configuration

    Returns:
        (stop_loss, take_profit, actual_rr_ratio)
    """
    config = config or RiskConfig()

    # Stop distance based on volatility
    stop_distance = config.stop_vol_multiplier * realized_vol

    # Take profit for minimum R:R
    tp_distance = stop_distance * config.min_rr_ratio

    if direction == "buy":
        stop_loss = entry - stop_distance
        take_profit = entry + tp_distance
    elif direction == "sell":
        stop_loss = entry + stop_distance
        take_profit = entry - tp_distance
    else:
        raise ValueError(f"Invalid direction: {direction}")

    actual_rr = tp_distance / stop_distance

    logger.info(
        "stops_calculated",
        entry=entry,
        direction=direction,
        stop_loss=stop_loss,
        take_profit=take_profit,
        stop_distance_pips=stop_distance / pip_value,
        actual_rr=actual_rr,
    )

    return stop_loss, take_profit, actual_rr
