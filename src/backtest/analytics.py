"""Performance analytics for AQRF backtesting.

Metrics, visualizations, and reporting.
"""
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class PerformanceAnalytics:
    """Calculate and format performance metrics."""

    def __init__(self, trades: List, equity_curve: List[float], timestamps: List[datetime]):
        self.trades = trades
        self.equity = np.array(equity_curve)
        self.timestamps = timestamps
        self.returns = np.diff(self.equity) / self.equity[:-1]

    def calculate_all_metrics(self) -> Dict:
        """Calculate comprehensive metrics."""
        metrics = {
            "returns": self._return_metrics(),
            "risk": self._risk_metrics(),
            "trades": self._trade_metrics(),
            "distribution": self._distribution_metrics(),
        }
        return metrics

    def _return_metrics(self) -> Dict:
        """Return-based metrics."""
        total_return = (self.equity[-1] - self.equity[0]) / self.equity[0]

        # Annualized
        n_days = max(len(self.timestamps) / (24 * 60), 1)
        annualized = (1 + total_return) ** (365 / n_days) - 1

        # Monthly returns
        monthly_returns = self._calculate_monthly_returns()

        return {
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(annualized * 100, 2),
            "monthly_returns": monthly_returns,
            "best_month_pct": round(max(monthly_returns) * 100, 2) if monthly_returns else 0,
            "worst_month_pct": round(min(monthly_returns) * 100, 2) if monthly_returns else 0,
        }

    def _risk_metrics(self) -> Dict:
        """Risk metrics."""
        if len(self.returns) < 2:
            return {"sharpe": 0, "sortino": 0, "max_drawdown": 0}

        # Sharpe (assuming M1 bars, annualize by sqrt(252 * 24 * 60))
        sharpe = np.mean(self.returns) / np.std(self.returns) * np.sqrt(252 * 24 * 60)

        # Sortino
        downside = self.returns[self.returns < 0]
        sortino = np.mean(self.returns) / np.std(downside) * np.sqrt(252 * 24 * 60) if len(downside) > 0 else 0

        # Max drawdown
        cummax = np.maximum.accumulate(self.equity)
        drawdowns = (cummax - self.equity) / cummax
        max_dd = np.max(drawdowns)

        # Calmar
        annual_return = self._return_metrics()["annualized_return_pct"] / 100
        calmar = annual_return / max_dd if max_dd > 0 else 0

        return {
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "calmar_ratio": round(calmar, 2),
        }

    def _trade_metrics(self) -> Dict:
        """Trade statistics."""
        if not self.trades:
            return {}

        pnls = [t.pnl for t in self.trades]
        durations = []  # Calculate from entry/exit times

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        return {
            "total_trades": len(self.trades),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_trade_pnl": round(np.mean(pnls), 2),
        }

    def _distribution_metrics(self) -> Dict:
        """Return distribution metrics."""
        if len(self.returns) < 2:
            return {}

        return {
            "skewness": round(float(np.mean((self.returns - np.mean(self.returns))**3) / np.std(self.returns)**3), 2),
            "kurtosis": round(float(np.mean((self.returns - np.mean(self.returns))**4) / np.std(self.returns)**4), 2),
            "positive_days_pct": round(np.mean(self.returns > 0) * 100, 2),
        }

    def _calculate_monthly_returns(self) -> List[float]:
        """Calculate monthly returns from equity curve."""
        if not self.timestamps:
            return []

        # Group by month
        monthly = {}
        for i, ts in enumerate(self.timestamps):
            key = (ts.year, ts.month)
            if key not in monthly:
                monthly[key] = []
            monthly[key].append(self.equity[i])

        returns = []
        for key in sorted(monthly.keys()):
            values = monthly[key]
            if len(values) > 1:
                ret = (values[-1] - values[0]) / values[0]
                returns.append(ret)

        return returns

    def generate_report(self) -> str:
        """Generate formatted performance report."""
        metrics = self.calculate_all_metrics()

        report = """
═══════════════════════════════════════════════════════
           AQRF BACKTEST PERFORMANCE REPORT
═══════════════════════════════════════════════════════

RETURN METRICS
──────────────
Total Return:        {total_return_pct}%
Annualized Return:   {annualized_return_pct}%
Best Month:          {best_month}%
Worst Month:         {worst_month}%

RISK METRICS
────────────
Sharpe Ratio:        {sharpe}
Sortino Ratio:       {sortino}
Max Drawdown:        {max_dd}%
Calmar Ratio:        {calmar}

TRADE STATISTICS
────────────────
Total Trades:        {total_trades}
Win Rate:            {win_rate}%
Profit Factor:       {pf}
Avg Win:             ${avg_win}
Avg Loss:            ${avg_loss}

═══════════════════════════════════════════════════════
""".format(
            total_return_pct=metrics["returns"]["total_return_pct"],
            annualized_return_pct=metrics["returns"]["annualized_return_pct"],
            best_month=metrics["returns"]["best_month_pct"],
            worst_month=metrics["returns"]["worst_month_pct"],
            sharpe=metrics["risk"]["sharpe_ratio"],
            sortino=metrics["risk"]["sortino_ratio"],
            max_dd=metrics["risk"]["max_drawdown_pct"],
            calmar=metrics["risk"]["calmar_ratio"],
            total_trades=metrics["trades"]["total_trades"],
            win_rate=metrics["trades"]["win_rate_pct"],
            pf=metrics["trades"]["profit_factor"],
            avg_win=metrics["trades"]["avg_win"],
            avg_loss=metrics["trades"]["avg_loss"],
        )

        return report
