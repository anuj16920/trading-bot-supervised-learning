"""Order management for AQRF.

Handles order lifecycle: pending -> filled -> closed.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    order_id: str
    timestamp: datetime
    symbol: str
    direction: str  # buy or sell
    size: float
    order_type: OrderType
    entry_price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    pnl: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None


class OrderManager:
    """Manages order lifecycle and position tracking."""

    def __init__(self):
        self.orders: List[Order] = []
        self.open_orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.order_counter = 0

    def generate_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"ORD_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.order_counter:04d}"

    def submit_order(
        self,
        symbol: str,
        direction: str,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Submit new order."""
        order = Order(
            order_id=self.generate_id(),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            size=size,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        self.orders.append(order)
        self.open_orders.append(order)

        logger.info(
            "order_submitted",
            order_id=order.order_id,
            symbol=symbol,
            direction=direction,
            size=size,
            type=order_type.value,
        )

        return order

    def fill_order(self, order_id: str, fill_price: float) -> Optional[Order]:
        """Mark order as filled."""
        for order in self.open_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.FILLED
                order.fill_price = fill_price
                order.fill_time = datetime.utcnow()

                logger.info(
                    "order_filled",
                    order_id=order_id,
                    fill_price=fill_price,
                )
                return order

        logger.warning("order_fill_not_found", order_id=order_id)
        return None

    def close_order(
        self,
        order_id: str,
        exit_price: float,
        reason: str,
    ) -> Optional[Order]:
        """Close filled order."""
        for order in self.open_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CLOSED
                order.exit_price = exit_price
                order.exit_time = datetime.utcnow()
                order.exit_reason = reason

                # Calculate PnL
                if order.direction == "buy":
                    order.pnl = (exit_price - order.fill_price) * order.size * 100000
                else:
                    order.pnl = (order.fill_price - exit_price) * order.size * 100000

                self.open_orders.remove(order)
                self.closed_orders.append(order)

                logger.info(
                    "order_closed",
                    order_id=order_id,
                    exit_price=exit_price,
                    pnl=order.pnl,
                    reason=reason,
                )
                return order

        logger.warning("order_close_not_found", order_id=order_id)
        return None

    def get_open_positions(self) -> List[Order]:
        """Get all open positions."""
        return [o for o in self.open_orders if o.status == OrderStatus.FILLED]

    def get_daily_stats(self) -> dict:
        """Get today's trading statistics."""
        today = datetime.utcnow().date()
        today_trades = [
            o for o in self.closed_orders
            if o.exit_time and o.exit_time.date() == today
        ]

        if not today_trades:
            return {"trades": 0, "pnl": 0}

        pnls = [t.pnl for t in today_trades if t.pnl is not None]
        return {
            "trades": len(today_trades),
            "pnl": round(sum(pnls), 2),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 2) if pnls else 0,
        }
