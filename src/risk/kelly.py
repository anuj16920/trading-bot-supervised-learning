"""Kelly Criterion position sizing for AQRF.

Half-Kelly with maximum 2% risk per trade.
"""
from typing import Optional

import structlog

from src.utils.config import RiskConfig

logger = structlog.get_logger(__name__)


def kelly_fraction(
    win_prob: float,
    rr_ratio: float,
    config: Optional[RiskConfig] = None,
) -> float:
    """Calculate Kelly fraction for position sizing.

    Args:
        win_prob: Probability of winning (0-1)
        rr_ratio: Reward-to-risk ratio
        config: Risk configuration

    Returns:
        Kelly fraction (0 to max_kelly_fraction)
    """
    config = config or RiskConfig()

    if win_prob <= 0 or rr_ratio <= 0:
        return 0.0

    q = 1.0 - win_prob

    # Full Kelly
    f = (win_prob * rr_ratio - q) / rr_ratio

    # Half-Kelly for safety
    half_kelly = f * config.kelly_fraction_multiplier

    # Clamp to maximum risk
    result = max(0.0, min(half_kelly, config.max_kelly_fraction))

    logger.debug(
        "kelly_calculated",
        win_prob=win_prob,
        rr_ratio=rr_ratio,
        full_kelly=f,
        half_kelly=half_kelly,
        final=result,
    )

    return result
