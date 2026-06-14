"""Walk-forward backtesting simulator for AQRF.

Expanding window training, fixed window testing.
"""
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import structlog

from src.backtest.engine import BacktestEngine
from src.models.ensemble import ModelEnsemble
from src.utils.config import DataConfig, RiskConfig

logger = structlog.get_logger(__name__)


class WalkForwardSimulator:
    """Walk-forward analysis with expanding training window."""

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
        self.prices = prices
        self.timestamps = timestamps
        self.ensemble = ensemble
        self.risk_config = risk_config or RiskConfig()
        self.initial_capital = initial_capital

        self.round_results: List[Dict] = []

    def run_round(
        self,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
        round_num: int,
    ) -> Dict:
        """Run single walk-forward round.

        Args:
            train_start: Training data start index
            train_end: Training data end index
            test_start: Test data start index
            test_end: Test data end index
            round_num: Round number for logging

        Returns:
            Backtest metrics dict
        """
        logger.info(
            "walk_forward_round_start",
            round=round_num,
            train_size=train_end - train_start,
            test_size=test_end - test_start,
        )

        # Test data only for backtest
        test_data = self.data[test_start:test_end]
        test_prices = self.prices[test_start:test_end]
        test_timestamps = self.timestamps[test_start:test_end]

        engine = BacktestEngine(
            data=test_data,
            prices=test_prices,
            timestamps=test_timestamps,
            ensemble=self.ensemble,
            risk_config=self.risk_config,
            initial_capital=self.initial_capital,
        )

        results = engine.run()
        results["round"] = round_num
        results["train_period"] = f"{self.timestamps[train_start]} to {self.timestamps[train_end]}"
        results["test_period"] = f"{self.timestamps[test_start]} to {self.timestamps[test_end]}"

        self.round_results.append(results)

        logger.info("walk_forward_round_complete", round=round_num, **results)
        return results

    def run_all_rounds(
        self,
        splits: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Run all walk-forward rounds.

        Default rounds:
            Round 1: Train 2015-2017, Test 2018
            Round 2: Train 2015-2018, Test 2019
            ...
            Final: Train 2015-2022, Test 2023-2024
        """
        if splits is None:
            # Generate default splits based on data length
            n = len(self.data)
            # Approximate year boundaries (assuming ~525,600 M1 bars/year)
            bars_per_year = 525_600

            splits = []
            for year in range(2018, 2023):
                train_end_idx = (year - 2015) * bars_per_year
                test_end_idx = train_end_idx + bars_per_year

                if test_end_idx > n:
                    break

                splits.append({
                    "train_start": 0,
                    "train_end": train_end_idx,
                    "test_start": train_end_idx,
                    "test_end": min(test_end_idx, n),
                })

            # Final round: all training data, test 2023-2024
            if len(self.data) > bars_per_year * 8:
                splits.append({
                    "train_start": 0,
                    "train_end": bars_per_year * 8,
                    "test_start": bars_per_year * 8,
                    "test_end": n,
                })

        for i, split in enumerate(splits):
            self.run_round(
                train_start=split["train_start"],
                train_end=split["train_end"],
                test_start=split["test_start"],
                test_end=split["test_end"],
                round_num=i + 1,
            )

        return self.round_results

    def aggregate_results(self) -> Dict:
        """Aggregate metrics across all rounds."""
        if not self.round_results:
            return {}

        metrics = [
            "total_return_pct",
            "annualized_return_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
            "avg_rr",
            "total_trades",
        ]

        aggregated = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in self.round_results if metric in r]
            if values:
                aggregated[f"{metric}_mean"] = round(np.mean(values), 2)
                aggregated[f"{metric}_std"] = round(np.std(values), 2)
                aggregated[f"{metric}_min"] = round(np.min(values), 2)
                aggregated[f"{metric}_max"] = round(np.max(values), 2)

        aggregated["total_rounds"] = len(self.round_results)

        logger.info("walk_forward_aggregated", **aggregated)
        return aggregated
