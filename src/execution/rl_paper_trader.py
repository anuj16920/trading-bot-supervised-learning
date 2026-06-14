"""RL Paper Trader — live simulation for the PPO agent (Module 4 — Phase 3).

Feeds historical or simulated bar data through the trained PPO agent bar by bar,
applies the confidence filter, and logs expected vs actual fill prices to measure
the simulation-to-reality gap before live deployment.

Unlike PaperTrader (which uses the ModelEnsemble), this class uses the raw RL
policy directly and tracks execution quality via fill_log.

Usage:
    trader = RLPaperTrader(model_path, config)
    results = trader.run_simulation(features, prices)
    analysis = trader.get_fill_analysis()
    trader.save_logs("paper_run_2024")
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from stable_baselines3 import PPO

from src.rl.confidence import filter_action, get_action_probs, compute_entropy
from src.rl.environment import ForexTradingEnv
from src.utils.config import RLConfig, RiskConfig

logger = structlog.get_logger(__name__)


class RLPaperTrader:
    """Paper trading driver for the PPO agent with fill quality tracking.

    Args:
        model_path:      Path to best_model.zip.
        config:          RLConfig (friction.randomize will be forced False).
        risk_config:     Optional RiskConfig for daily drawdown limits.
        initial_capital: Starting capital.
        log_dir:         Directory for trade and fill logs.
    """

    def __init__(
        self,
        model_path: Path,
        config:         Optional[RLConfig]   = None,
        risk_config:    Optional[RiskConfig] = None,
        initial_capital: float = 10_000.0,
        log_dir: Path = Path("./paper_trades_rl"),
    ):
        self.cfg            = (config or RLConfig()).model_copy(deep=True)
        self.cfg.friction.randomize = False   # always deterministic in paper mode
        self.risk_cfg       = risk_config or RiskConfig()
        self.initial_capital = initial_capital
        self.log_dir        = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("loading_model", path=str(model_path))
        self.model = PPO.load(str(model_path))

        # Portfolio state (mirrors ForexTradingEnv)
        self.capital      = initial_capital
        self.peak_capital = initial_capital
        self.position     = 0.0
        self.entry_price  = 0.0
        self.bars_held    = 0
        self.total_trades = 0
        self.winning      = 0
        self.sl_hits      = 0
        self.tp_hits      = 0

        # Delayed action queue (mirrors env execution delay logic)
        self._pending_action: Optional[int] = None
        self._pending_bars:   int           = 0

        # Logs
        self.trade_log: list[dict] = []
        self.fill_log:  list[dict] = []   # expected vs actual fills

        # Internal env for observation construction only
        self._env: Optional[ForexTradingEnv] = None
        self._features: Optional[np.ndarray] = None
        self._prices:   Optional[np.ndarray] = None

    # ── Initialisation ────────────────────────────────────────────────

    def initialize(self, features: np.ndarray, prices: np.ndarray) -> None:
        """Attach dataset. Must be called before run_simulation()."""
        self._features = features
        self._prices   = prices
        self._env      = ForexTradingEnv(features, prices, config=self.cfg)
        self._env.reset()
        logger.info("paper_trader_initialized", bars=len(features))

    # ── Core bar processor ────────────────────────────────────────────

    def on_bar(self, bar_idx: int) -> Optional[dict]:
        """Process one bar. Returns trade event dict or None."""
        if self._env is None:
            raise RuntimeError("Call initialize() before on_bar().")

        self._env.cur_idx = bar_idx

        # Build observation using env's _obs() with our portfolio state injected
        obs = self._get_obs(bar_idx)

        # Get action probabilities and apply confidence filter
        try:
            probs  = get_action_probs(self.model, obs)
            entropy = compute_entropy(probs)
        except Exception:
            probs, entropy = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.0

        raw_action, _ = self.model.predict(obs, deterministic=True)
        raw_action    = int(raw_action)

        if self.cfg.use_confidence_filter:
            action = filter_action(raw_action, probs, self.cfg.confidence_threshold)
        else:
            action = raw_action

        # Handle execution delay
        if self._pending_action is not None:
            self._pending_bars -= 1
            if self._pending_bars <= 0:
                action = self._pending_action
                self._pending_action = None
            else:
                action = 0  # hold while waiting
        elif self.cfg.friction.eval_delay_bars > 0 and action in (1, 2):
            self._pending_action = action
            self._pending_bars   = self.cfg.friction.eval_delay_bars
            action = 0

        # Check SL/TP before executing agent action
        sl_event = self._check_sl_tp_paper(bar_idx)
        if sl_event:
            return sl_event

        # Execute action
        return self._execute_paper(action, bar_idx, probs, entropy)

    def _get_obs(self, bar_idx: int) -> np.ndarray:
        """Build (seq_len, n_obs) observation using our portfolio state."""
        mkt      = self._features[bar_idx]   # (seq_len, n_mkt_feat)
        seq_len  = mkt.shape[0]
        upnl_pct = self._unrealised_pnl(bar_idx) / self.initial_capital
        dd       = self._drawdown()
        held_norm = min(self.bars_held / 100.0, 1.0)
        port = np.full((seq_len, 4), [self.position, upnl_pct, dd, held_norm], dtype=np.float32)
        return np.concatenate([mkt, port], axis=-1).astype(np.float32)

    def _effective_prices(self, bar_idx: int) -> tuple[float, float]:
        bid_raw, ask_raw = self._prices[bar_idx]
        half_spread = (self.cfg.friction.eval_spread_pips * 0.0001) / 2.0
        mid = (bid_raw + ask_raw) / 2.0
        return mid - half_spread, mid + half_spread

    def _apply_friction(self, price: float, direction: int) -> tuple[float, float]:
        """Returns (expected_price, actual_price). expected = mid, actual = with friction."""
        slip = self.cfg.friction.eval_slippage_pips * 0.0001
        actual = price + direction * slip
        return price, actual

    def _unrealised_pnl(self, bar_idx: int) -> float:
        if self.position == 0.0:
            return 0.0
        bid, ask = self._effective_prices(bar_idx)
        mid = (bid + ask) * 0.5
        return (mid - self.entry_price) * self.position * self.cfg.lot_size

    def _drawdown(self) -> float:
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - self.capital) / self.peak_capital)

    def _check_sl_tp_paper(self, bar_idx: int) -> Optional[dict]:
        """Check hard SL/TP. Returns close event dict or None."""
        if self.position == 0.0:
            return None

        bid, ask = self._effective_prices(bar_idx)
        mid = (bid + ask) * 0.5

        if self.position > 0:
            upnl_pips = (mid - self.entry_price) / 0.0001
            if upnl_pips <= -self.cfg.stop_loss_pips:
                return self._close_paper(bar_idx, bid, -1, reason="sl")
            if upnl_pips >= self.cfg.take_profit_pips:
                return self._close_paper(bar_idx, bid, -1, reason="tp")
        elif self.position < 0:
            upnl_pips = (self.entry_price - mid) / 0.0001
            if upnl_pips <= -self.cfg.stop_loss_pips:
                return self._close_paper(bar_idx, ask, +1, reason="sl")
            if upnl_pips >= self.cfg.take_profit_pips:
                return self._close_paper(bar_idx, ask, +1, reason="tp")
        return None

    def _execute_paper(
        self, action: int, bar_idx: int, probs: np.ndarray, entropy: float
    ) -> Optional[dict]:
        bid, ask = self._effective_prices(bar_idx)
        event    = None

        if action == 0:
            return None

        elif action == 1:  # buy
            if self.position < 0:
                event = self._close_paper(bar_idx, ask, +1, reason="signal_close")
            if self.position == 0.0:
                expected, actual = self._apply_friction(ask, +1)
                self.entry_price  = actual
                self.position     = 1.0
                self.bars_held    = 0
                self.total_trades += 1
                self._log_fill(bar_idx, "buy", expected, actual, probs, entropy)

        elif action == 2:  # sell
            if self.position > 0:
                event = self._close_paper(bar_idx, bid, -1, reason="signal_close")
            if self.position == 0.0:
                expected, actual = self._apply_friction(bid, -1)
                self.entry_price  = actual
                self.position     = -1.0
                self.bars_held    = 0
                self.total_trades += 1
                self._log_fill(bar_idx, "sell", expected, actual, probs, entropy)

        elif action == 3:  # close
            if self.position > 0:
                event = self._close_paper(bar_idx, bid, -1, reason="manual_close")
            elif self.position < 0:
                event = self._close_paper(bar_idx, ask, +1, reason="manual_close")

        if self.position != 0.0:
            self.bars_held += 1

        return event

    def _close_paper(self, bar_idx: int, exit_raw: float, direction: int, reason: str) -> dict:
        expected, actual = self._apply_friction(exit_raw, direction)
        if self.position > 0:
            pnl = (actual - self.entry_price) * self.position * self.cfg.lot_size
        else:
            pnl = (self.entry_price - actual) * abs(self.position) * self.cfg.lot_size

        self.capital      = max(0.0, self.capital + pnl)
        self.peak_capital = max(self.peak_capital, self.capital)
        if pnl > 0:
            self.winning += 1
        if reason == "sl":
            self.sl_hits += 1
        if reason == "tp":
            self.tp_hits += 1

        self._log_fill(bar_idx, f"close_{reason}", expected, actual,
                       np.zeros(4, dtype=np.float32), 0.0)

        event = {
            "bar_idx":     bar_idx,
            "reason":      reason,
            "pnl":         pnl,
            "capital":     self.capital,
            "position":    0.0,
            "entry_price": self.entry_price,
            "exit_price":  actual,
        }
        self.trade_log.append(event)

        self.position    = 0.0
        self.bars_held   = 0
        self.entry_price = 0.0
        return event

    def _log_fill(
        self,
        bar_idx: int,
        side: str,
        expected: float,
        actual: float,
        probs: np.ndarray,
        entropy: float,
    ) -> None:
        self.fill_log.append({
            "bar_idx":    bar_idx,
            "side":       side,
            "expected":   expected,
            "actual":     actual,
            "slip_pips":  abs(actual - expected) / 0.0001,
            "confidence": float(np.max(probs)),
            "entropy":    entropy,
        })

    # ── Simulation runner ─────────────────────────────────────────────

    def run_simulation(
        self,
        features: np.ndarray,
        prices:   np.ndarray,
        start_bar: int = 0,
        end_bar:   Optional[int] = None,
    ) -> dict:
        """Run full bar-by-bar simulation. Returns final metrics dict."""
        self.initialize(features, prices)
        end = end_bar or len(features)

        for bar_idx in range(start_bar, end):
            self.on_bar(bar_idx)

        # Force-close any open position at end
        if self.position != 0.0 and end > 0:
            bid, ask = self._effective_prices(end - 1)
            price    = bid if self.position > 0 else ask
            self._close_paper(end - 1, price,
                              -1 if self.position > 0 else +1,
                              reason="end_of_data")

        return self.get_stats()

    # ── Statistics ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Current portfolio statistics."""
        total_t  = self.total_trades
        win_rate = self.winning / total_t if total_t > 0 else 0.0
        pnls     = [t["pnl"] for t in self.trade_log]
        return {
            "capital":         self.capital,
            "peak_capital":    self.peak_capital,
            "drawdown_pct":    self._drawdown() * 100,
            "total_return_pct": (self.capital / self.initial_capital - 1) * 100,
            "total_trades":    total_t,
            "win_rate_pct":    win_rate * 100,
            "total_pnl":       self.capital - self.initial_capital,
            "avg_trade_pnl":   float(np.mean(pnls)) if pnls else 0.0,
            "sl_hits":         self.sl_hits,
            "tp_hits":         self.tp_hits,
        }

    def get_fill_analysis(self) -> dict:
        """Analyse fill_log: mean slippage, fill quality, confidence distribution."""
        if not self.fill_log:
            return {"error": "no fills recorded"}

        slippages   = [f["slip_pips"]   for f in self.fill_log]
        confidences = [f["confidence"]  for f in self.fill_log]
        entropies   = [f["entropy"]     for f in self.fill_log]

        return {
            "n_fills":           len(self.fill_log),
            "mean_slip_pips":    float(np.mean(slippages)),
            "max_slip_pips":     float(np.max(slippages)),
            "mean_confidence":   float(np.mean(confidences)),
            "min_confidence":    float(np.min(confidences)),
            "mean_entropy":      float(np.mean(entropies)),
            "filtered_pct":      0.0,   # placeholder — tracked in on_bar if needed
            "total_cost_pips":   float(np.sum(slippages)),
        }

    def save_logs(self, prefix: str = "rl_paper") -> None:
        """Save trade_log and fill_log as CSV files."""
        if self.trade_log:
            trade_path = self.log_dir / f"{prefix}_trades.csv"
            with open(trade_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.trade_log[0].keys())
                writer.writeheader()
                writer.writerows(self.trade_log)
            logger.info("trade_log_saved", path=str(trade_path))

        if self.fill_log:
            fill_path = self.log_dir / f"{prefix}_fills.csv"
            with open(fill_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fill_log[0].keys())
                writer.writeheader()
                writer.writerows(self.fill_log)
            logger.info("fill_log_saved", path=str(fill_path))
