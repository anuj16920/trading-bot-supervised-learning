"""Forex trading environment for AQRF.

No prediction layer. The RL agent sees raw multi-timeframe features directly
and learns WHEN to enter/exit for profit. Reward = realised P&L after costs.

Observation: (seq_len, n_features + 4 portfolio features)
Actions: 0=hold, 1=buy, 2=sell, 3=close

Phase 3: Domain randomization via FrictionConfig — spread, slippage, execution
delay, and fill quality are randomized each episode during training, and fixed
to deterministic eval values during evaluation.
"""
from collections import deque
from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.utils.config import RLConfig


class ForexTradingEnv(gym.Env):
    """EUR/USD trading environment driven by pre-computed feature sequences."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,   # (N, seq_len, n_features)
        prices:   np.ndarray,   # (N, 2)  [bid, ask] at bar close
        config:   Optional[RLConfig] = None,
    ):
        super().__init__()
        self.cfg   = config or RLConfig()
        self.data  = features
        self.prices = prices

        self.n_bars     = len(features)
        self.seq_len    = features.shape[1]
        self.n_mkt_feat = features.shape[2]
        self.ep_len     = min(self.cfg.episode_bars, self.n_bars - 1)

        # Portfolio state (also initialises Phase 3 friction state)
        self._reset_portfolio()

        # Spaces
        n_obs = self.n_mkt_feat + 5   # market + [position, upnl_pct, drawdown, bars_held_norm, cooldown_norm]
        self.observation_space = spaces.Box(
            low=-6.0, high=6.0,
            shape=(self.seq_len, n_obs),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)  # hold, buy, sell, close

    # ── Portfolio helpers ─────────────────────────────────────────────

    def _reset_portfolio(self):
        self.capital      = self.cfg.initial_capital
        self.peak_capital = self.cfg.initial_capital
        self.position     = 0.0   # +1 long, -1 short, 0 flat
        self.entry_price  = 0.0
        self.bars_held    = 0
        self.total_trades = 0     # counts ROUND-TRIPS only (one open+close = 1)
        self.winning      = 0
        self.cur_idx      = 0
        self.ep_start     = 0
        self.sl_hits      = 0
        self.tp_hits      = 0
        self.cooldown_bars_remaining = 0  # bars left before next entry allowed
        # Daily trade cap: deque of bar indices when each round-trip was closed.
        # maxlen=max_trades_per_day+1 — we only need to know if the oldest close
        # is within the last bars_per_day, so we never store more than needed.
        self._trade_close_bars: deque = deque(maxlen=self.cfg.max_trades_per_day + 1)
        # Phase 3: per-episode friction state (overwritten by _sample_friction at reset)
        self._ep_spread_pips:     float          = self.cfg.friction.eval_spread_pips
        self._ep_slippage_pips:   float          = self.cfg.friction.eval_slippage_pips
        self._ep_delay_bars:      int            = self.cfg.friction.eval_delay_bars
        self._ep_fill_quality:    float          = 1.0
        self._pending_action:     Optional[int]  = None
        self._pending_bars_remaining: int        = 0

    def _daily_trades_used(self) -> int:
        """Count round-trips closed within the last bars_per_day bars.

        The deque has maxlen=max_trades_per_day+1, so this is O(max_trades_per_day) — constant time.
        """
        window_start = self.cur_idx - self.cfg.bars_per_day
        return sum(1 for b in self._trade_close_bars if b >= window_start)

    def _daily_limit_reached(self) -> bool:
        return self._daily_trades_used() >= self.cfg.max_trades_per_day

    def _record_close_bar(self):
        """Record that a trade was closed at the current bar (for daily cap tracking)."""
        self._trade_close_bars.append(self.cur_idx)

    def _unrealised_pnl(self) -> float:
        if self.position == 0.0:
            return 0.0
        bid, ask = self._effective_prices()
        mid = (bid + ask) * 0.5
        return (mid - self.entry_price) * self.position * self.cfg.lot_size

    def _equity(self) -> float:
        """Realized capital plus unrealized P&L — true account equity."""
        return self.capital + self._unrealised_pnl()

    def _drawdown(self) -> float:
        """Drawdown based on equity (includes floating loss), not just realized capital."""
        equity = self._equity()
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - equity) / self.peak_capital)

    # ── Phase 3: Friction helpers ─────────────────────────────────────

    def _sample_friction(self) -> None:
        """Sample per-episode execution friction. Called at the top of reset()."""
        fc = self.cfg.friction
        if fc.randomize:
            rng = self.np_random
            self._ep_spread_pips   = float(rng.uniform(fc.spread_min_pips, fc.spread_max_pips))
            self._ep_slippage_pips = float(rng.uniform(fc.slippage_min_pips, fc.slippage_max_pips))
            self._ep_delay_bars    = int(rng.integers(fc.delay_min_bars, fc.delay_max_bars + 1))
            self._ep_fill_quality  = float(rng.uniform(fc.fill_quality_min, fc.fill_quality_max))
        else:
            self._ep_spread_pips   = fc.eval_spread_pips
            self._ep_slippage_pips = fc.eval_slippage_pips
            self._ep_delay_bars    = fc.eval_delay_bars
            self._ep_fill_quality  = 1.0
        self._pending_action          = None
        self._pending_bars_remaining  = 0

    def _effective_prices(self) -> Tuple[float, float]:
        """Return (bid, ask) adjusted for episode spread."""
        safe_idx = min(self.cur_idx, self.n_bars - 1)
        bid_raw, ask_raw = self.prices[safe_idx]
        half_spread = (self._ep_spread_pips * 0.0001) / 2.0
        mid = (bid_raw + ask_raw) / 2.0
        return mid - half_spread, mid + half_spread

    def _apply_fill_quality(self, price: float, direction: int) -> float:
        """Worsen fill price based on episode slippage.

        direction: +1 = buying (price moves against us upward),
                   -1 = selling (price moves against us downward).

        In deterministic eval mode (randomize=False) the full configured
        slippage is always applied — fill_quality discount is training-only.
        In randomized training mode fill_quality scales how much of the
        sampled slippage actually hits the fill.
        """
        slip = self._ep_slippage_pips * 0.0001
        if not self.cfg.friction.randomize:
            # Audit/eval: apply full slippage unconditionally
            return price + direction * slip
        # Training: fill_quality in [0.7, 1.0] per config comment:
        # fill_quality=1.0 -> perfect fill (0% extra slippage worsens fill).
        # fill_quality=0.7 -> 30% of slippage worsens fill.
        # So the degradation fraction = (1 - fill_quality).
        worst = price + direction * slip
        return price + direction * (1.0 - self._ep_fill_quality) * abs(worst - price)

    # ── Observation ───────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        # Clamp so a post-increment cur_idx at episode end never goes out of bounds.
        safe_idx = min(self.cur_idx, self.n_bars - 1)
        mkt = self.data[safe_idx]   # (seq_len, n_mkt_feat)

        upnl_pct      = self._unrealised_pnl() / self.cfg.initial_capital
        dd            = self._drawdown()
        held_norm     = min(self.bars_held / 100.0, 1.0)
        cooldown_norm = min(self.cooldown_bars_remaining / self.cfg.trade_cooldown_bars, 1.0)

        # Broadcast 5 portfolio scalars across sequence axis
        port = np.full((self.seq_len, 5), [self.position, upnl_pct, dd, held_norm, cooldown_norm], dtype=np.float32)
        return np.concatenate([mkt, port], axis=-1).astype(np.float32)

    # ── Execution ─────────────────────────────────────────────────────

    def _execute(self, action: int) -> Tuple[float, bool]:
        """Return (realised PnL, trade_closed). Uses episode friction for fills.

        total_trades counts ROUND-TRIPS only. One open+close = 1 trade.
        Cooldown enforced after every close: agent must wait trade_cooldown_bars
        before opening a new position, preventing immediate flip-flopping.
        """
        bid, ask = self._effective_prices()
        pnl = 0.0
        trade_closed = False

        if action == 0:    # hold
            return 0.0, False

        elif action == 1:  # buy
            if self.position < 0:
                # Close short — counts as 1 completed round-trip
                exit_price = self._apply_fill_quality(ask, +1)
                pnl = (self.entry_price - exit_price) * abs(self.position) * self.cfg.lot_size
                self._record_trade(pnl)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0
                self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                trade_closed = True
            elif self.position == 0.0 and self.cooldown_bars_remaining <= 0 and not self._daily_limit_reached():
                # Open long — blocked by cooldown OR daily cap
                self.entry_price = self._apply_fill_quality(ask, +1)
                self.position = 1.0
                self.bars_held = 0

        elif action == 2:  # sell
            if self.position > 0:
                # Close long — counts as 1 completed round-trip
                exit_price = self._apply_fill_quality(bid, -1)
                pnl = (exit_price - self.entry_price) * self.position * self.cfg.lot_size
                self._record_trade(pnl)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0
                self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                trade_closed = True
            elif self.position == 0.0 and self.cooldown_bars_remaining <= 0 and not self._daily_limit_reached():
                # Open short — blocked by cooldown OR daily cap
                self.entry_price = self._apply_fill_quality(bid, -1)
                self.position = -1.0
                self.bars_held = 0

        elif action == 3:  # explicit close
            if self.position > 0:
                exit_price = self._apply_fill_quality(bid, -1)
                pnl = (exit_price - self.entry_price) * self.position * self.cfg.lot_size
                self._record_trade(pnl)
                self._record_close_bar()
                self.total_trades += 1
                trade_closed = True
            elif self.position < 0:
                exit_price = self._apply_fill_quality(ask, +1)
                pnl = (self.entry_price - exit_price) * abs(self.position) * self.cfg.lot_size
                self._record_trade(pnl)
                self._record_close_bar()
                self.total_trades += 1
                trade_closed = True
            if trade_closed:
                self.position = 0.0
                self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars

        return float(pnl), trade_closed

    def _check_sl_tp(self) -> Tuple[float, bool, bool, bool]:
        """Check SL/TP levels. Returns (pnl, trade_closed, sl_hit, tp_hit)."""
        if self.position == 0.0:
            return 0.0, False, False, False

        bid, ask = self._effective_prices()
        mid  = (bid + ask) * 0.5

        if self.position > 0:
            upnl_pips = (mid - self.entry_price) / 0.0001
            if upnl_pips <= -self.cfg.stop_loss_pips:
                exit_price = self._apply_fill_quality(bid, -1)
                pnl = (exit_price - self.entry_price) * self.position * self.cfg.lot_size
                self._record_trade(pnl, sl_hit=True)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0; self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                return float(pnl), True, True, False
            if upnl_pips >= self.cfg.take_profit_pips:
                exit_price = self._apply_fill_quality(bid, -1)
                pnl = (exit_price - self.entry_price) * self.position * self.cfg.lot_size
                self._record_trade(pnl, tp_hit=True)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0; self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                return float(pnl), True, False, True

        elif self.position < 0:
            upnl_pips = (self.entry_price - mid) / 0.0001
            if upnl_pips <= -self.cfg.stop_loss_pips:
                exit_price = self._apply_fill_quality(ask, +1)
                pnl = (self.entry_price - exit_price) * abs(self.position) * self.cfg.lot_size
                self._record_trade(pnl, sl_hit=True)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0; self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                return float(pnl), True, True, False
            if upnl_pips >= self.cfg.take_profit_pips:
                exit_price = self._apply_fill_quality(ask, +1)
                pnl = (self.entry_price - exit_price) * abs(self.position) * self.cfg.lot_size
                self._record_trade(pnl, tp_hit=True)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0; self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                return float(pnl), True, False, True

        return 0.0, False, False, False

    def _record_trade(self, pnl: float, sl_hit: bool = False, tp_hit: bool = False):
        if pnl > 0:
            self.winning += 1
        if sl_hit:
            self.sl_hits += 1
        if tp_hit:
            self.tp_hits += 1

    # ── Reward ────────────────────────────────────────────────────────

    def _realized_vol(self) -> float:
        """Rolling realized volatility over the last vol_window bars (in pip units)."""
        start = max(0, self.cur_idx - self.cfg.vol_window)
        end   = max(1, self.cur_idx)
        if end - start < 2:
            return 1.0
        mids = (self.prices[start:end, 0] + self.prices[start:end, 1]) * 0.5
        log_rets = np.diff(np.log(np.maximum(mids, 1e-8)))
        vol_price = float(np.std(log_rets)) * mids[-1]
        vol_pips  = vol_price / 0.0001
        return max(vol_pips, 0.1)   # floor at 0.1 pip to prevent division explosion

    def _reward(self, pnl: float, trade_closed: bool, sl_hit: bool = False, tp_hit: bool = False) -> float:
        if not trade_closed:
            return 0.0

        # Plain pip reward: stable magnitude, interpretable, easy for value fn to fit.
        pips = pnl / (self.cfg.lot_size * 0.0001)

        if tp_hit:
            r = pips * self.cfg.reward_scale_win
        elif sl_hit:
            r = pips * self.cfg.reward_scale_loss
        else:
            r = pips * self.cfg.reward_scale_win if pips > 0 else pips * self.cfg.reward_scale_loss

        # Overtrade penalty: fires per-trade beyond threshold so scalping is always
        # net-negative. With TP=+30 pips and penalty=-20, excess trades cost money.
        if self.total_trades > self.cfg.overtrade_threshold:
            r += self.cfg.overtrade_penalty

        return float(r)

    # ── Gym interface ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_portfolio()
        self._sample_friction()   # Phase 3: sample new friction parameters each episode

        max_start = max(0, self.n_bars - self.ep_len - 1)
        self.ep_start = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self.cur_idx  = self.ep_start

        return self._obs(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Phase 3: execution delay — queue action, execute HOLD while waiting
        if self._pending_action is not None:
            self._pending_bars_remaining -= 1
            if self._pending_bars_remaining <= 0:
                action_to_execute = self._pending_action
                self._pending_action = None
            else:
                action_to_execute = 0  # hold while waiting for delayed fill
        elif self._ep_delay_bars > 0 and action in (1, 2):
            self._pending_action = action
            self._pending_bars_remaining = self._ep_delay_bars
            action_to_execute = 0
        else:
            action_to_execute = action

        # 1. Check SL/TP first — overrides agent action if triggered
        sl_pnl, sl_closed, sl_hit, tp_hit = self._check_sl_tp()

        if sl_closed:
            pnl, trade_closed = sl_pnl, True
        else:
            pnl, trade_closed = self._execute(action_to_execute)
            sl_hit = tp_hit = False

        self.capital += pnl
        # Peak tracks equity (capital + unrealized) so floating losses count
        self.peak_capital = max(self.peak_capital, self._equity())

        if self.position != 0.0:
            self.bars_held += 1

        if self.cooldown_bars_remaining > 0:
            self.cooldown_bars_remaining -= 1

        reward = self._reward(pnl, trade_closed, sl_hit=sl_hit, tp_hit=tp_hit)

        self.cur_idx += 1

        terminated = False
        truncated  = False

        if self.cur_idx >= self.ep_start + self.ep_len:
            # Force-close any open position at episode end so unrealized P&L
            # is always realized before the episode terminates.
            if self.position != 0.0:
                # cur_idx was already incremented; clamp before accessing prices.
                safe = min(self.cur_idx, self.n_bars - 1)
                bid_raw, ask_raw = self.prices[safe]
                half_spread = (self._ep_spread_pips * 0.0001) / 2.0
                mid = (bid_raw + ask_raw) / 2.0
                bid, ask = mid - half_spread, mid + half_spread
                slip = self._ep_slippage_pips * 0.0001
                if self.position > 0:
                    exit_price = bid - slip
                    close_pnl  = (exit_price - self.entry_price) * self.position * self.cfg.lot_size
                else:
                    exit_price = ask + slip
                    close_pnl  = (self.entry_price - exit_price) * abs(self.position) * self.cfg.lot_size
                self._record_trade(close_pnl)
                self.total_trades += 1
                self.capital  += close_pnl
                self.position  = 0.0
                self.bars_held = 0
                reward += self._reward(close_pnl, True)
            truncated = True

        if self._drawdown() > self.cfg.max_drawdown or self.capital < self.cfg.min_account:
            # Force-close any open position before terminating so unrealized P&L
            # is fully realized and the terminal state is clean.
            if self.position != 0.0:
                safe = min(self.cur_idx, self.n_bars - 1)
                bid_raw, ask_raw = self.prices[safe]
                half_spread = (self._ep_spread_pips * 0.0001) / 2.0
                mid = (bid_raw + ask_raw) / 2.0
                bid_t, ask_t = mid - half_spread, mid + half_spread
                slip = self._ep_slippage_pips * 0.0001
                if self.position > 0:
                    exit_price = bid_t - slip
                    close_pnl  = (exit_price - self.entry_price) * self.position * self.cfg.lot_size
                else:
                    exit_price = ask_t + slip
                    close_pnl  = (self.entry_price - exit_price) * abs(self.position) * self.cfg.lot_size
                self._record_trade(close_pnl)
                self.total_trades += 1
                self.capital  += close_pnl
                self.position  = 0.0
                self.bars_held = 0
                reward += self._reward(close_pnl, True)
            terminated = True
            reward -= 10.0

        return self._obs(), reward, terminated, truncated, self._info()

    def _info(self) -> dict:
        idx = min(self.cur_idx, self.n_bars - 1)
        wr  = self.winning / self.total_trades if self.total_trades > 0 else 0.0
        return {
            "capital":          self.capital,
            "equity":           self._equity(),
            "drawdown":         self._drawdown(),
            "position":         self.position,
            "trades":           self.total_trades,
            "win_rate":         wr,
            "unrealised_pnl":   self._unrealised_pnl(),
            "price":            float(self.prices[idx].mean()),
            "sl_hits":          self.sl_hits,
            "tp_hits":          self.tp_hits,
            "realized_vol_pips": self._realized_vol(),
            "bars_held":        self.bars_held,
            "daily_trades":     self._daily_trades_used(),
            "daily_limit_reached": self._daily_limit_reached(),
            # Phase 3: friction diagnostics
            "ep_spread_pips":   self._ep_spread_pips,
            "ep_slippage_pips": self._ep_slippage_pips,
            "ep_delay_bars":    self._ep_delay_bars,
            "cooldown_remaining": self.cooldown_bars_remaining,
        }
