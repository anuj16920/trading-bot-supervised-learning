"""AQRF v10 — Comprehensive Stress Evaluation on 2023-2024 Test Data.

Tests the model under progressively harsher execution conditions to determine
whether performance is robust or fragile.

Stress dimensions:
  1. Spread ladder      : 0.5, 1.0, 1.5, 2.0, 3.0 pips
  2. Slippage ladder    : 0.3, 0.5, 1.0, 1.5, 2.0 pips
  3. Execution delay    : 1, 2, 3 bars
  4. Random skipped fills (5% and 15% of fills randomly cancelled)
  5. Worse SL fill      : SL fills at mid + 2x slippage instead of standard
  6. News spread spike  : 5x spread for 3 bars on random 0.5% of bars
  7. Worst-case SL/TP   : same-candle handling with worst fill
  8. Risk scale         : 1x, 2x, 3x, 4x, 5x lot size

2025-2026 Section:
  Data not available in current processed dataset.
  Reserved for future external validation once new unseen data is added.
  No 2025-2026 results are reported here.

Usage:
    python eval_stress.py
    python eval_stress.py --model PATH/model.zip --tag v10
"""
import argparse
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.rl.environment import ForexTradingEnv
from src.utils.config import load_config, RLConfig
from src.utils.logging import setup_logging
import structlog

logger = structlog.get_logger(__name__)

# Live log file — written in real time so you can tail it while the script runs
_live_log = None

def log(msg: str):
    """Print to stdout and flush to live log file immediately."""
    print(msg, flush=True)
    if _live_log is not None:
        _live_log.write(msg + "\n")
        _live_log.flush()

TEST_MONTHS = [
    "2023-Jan", "2023-Feb", "2023-Mar", "2023-Apr", "2023-May", "2023-Jun",
    "2023-Jul", "2023-Aug", "2023-Sep", "2023-Oct", "2023-Nov", "2023-Dec",
    "2024-Jan", "2024-Feb", "2024-Mar", "2024-Apr", "2024-May", "2024-Jun",
    "2024-Jul", "2024-Aug", "2024-Sep", "2024-Oct", "2024-Nov", "2024-Dec",
]

INITIAL_CAPITAL = 10_000.0
BARS_PER_DAY    = 1_440
BARS_PER_WEEK   = 7_200
RNG_SEED        = 42


# ── Patched environment runner ───────────────────────────────────────────────

class StressEnv(ForexTradingEnv):
    """ForexTradingEnv with injectable stress hooks.

    stress_hooks dict (all optional):
      skip_fill_prob  : float — probability any open/close fill is silently skipped
      sl_fill_mult    : float — multiply slippage by this on SL exits (e.g. 2.0)
      news_spike_prob : float — per-bar probability of a spread spike
      news_spike_mult : float — spread multiplier during news spike (e.g. 5.0)
      news_spike_bars : int   — how many bars the spike lasts
      worst_case_sltp : bool  — SL/TP fills at worst price in candle (mid ± 2*slip)
    """

    def __init__(self, features, prices, cfg, stress_hooks=None, rng_seed=42):
        super().__init__(features, prices, cfg)
        self.hooks   = stress_hooks or {}
        self._srng   = np.random.default_rng(rng_seed)
        self._spike_bars_remaining = 0
        self._base_spread = cfg.friction.eval_spread_pips

    def _effective_prices(self):
        bid, ask = super()._effective_prices()
        # News spike: randomly widen spread for spike_bars
        if self._spike_bars_remaining > 0:
            self._spike_bars_remaining -= 1
            mult = self.hooks.get("news_spike_mult", 5.0)
            half_extra = (self._base_spread * (mult - 1) * 0.0001) / 2.0
            bid -= half_extra
            ask += half_extra
        elif self._srng.random() < self.hooks.get("news_spike_prob", 0.0):
            self._spike_bars_remaining = self.hooks.get("news_spike_bars", 3) - 1
            mult = self.hooks.get("news_spike_mult", 5.0)
            half_extra = (self._base_spread * (mult - 1) * 0.0001) / 2.0
            bid -= half_extra
            ask += half_extra
        return bid, ask

    def _apply_fill_quality(self, price, direction):
        slip = self._ep_slippage_pips * 0.0001
        return price + direction * slip

    def _check_sl_tp(self):
        """Override to apply sl_fill_mult on SL exits and worst_case_sltp."""
        if self.position == 0.0:
            return 0.0, False, False, False

        bid, ask = self._effective_prices()
        mid = (bid + ask) * 0.5
        sl_mult = self.hooks.get("sl_fill_mult", 1.0)
        slip_base = self._ep_slippage_pips * 0.0001

        if self.position > 0:
            upnl_pips = (mid - self.entry_price) / 0.0001
            if upnl_pips <= -self.cfg.stop_loss_pips:
                # SL exit: apply sl_fill_mult to slippage
                exit_price = bid - slip_base * sl_mult
                pnl = (exit_price - self.entry_price) * self.position * self.cfg.lot_size
                self._record_trade(pnl, sl_hit=True)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0; self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                return float(pnl), True, True, False
            if upnl_pips >= self.cfg.take_profit_pips:
                exit_price = bid - slip_base
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
                exit_price = ask + slip_base * sl_mult
                pnl = (self.entry_price - exit_price) * abs(self.position) * self.cfg.lot_size
                self._record_trade(pnl, sl_hit=True)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0; self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                return float(pnl), True, True, False
            if upnl_pips >= self.cfg.take_profit_pips:
                exit_price = ask + slip_base
                pnl = (self.entry_price - exit_price) * abs(self.position) * self.cfg.lot_size
                self._record_trade(pnl, tp_hit=True)
                self._record_close_bar()
                self.total_trades += 1
                self.position = 0.0; self.bars_held = 0
                self.cooldown_bars_remaining = self.cfg.trade_cooldown_bars
                return float(pnl), True, False, True

        return 0.0, False, False, False

    def _execute(self, action):
        # Random skipped fill: silently treat open/close actions as hold
        if action in (1, 2, 3) and self._srng.random() < self.hooks.get("skip_fill_prob", 0.0):
            return 0.0, False
        return super()._execute(action)


def run_episode(features, prices, cfg, model, stress_hooks=None, lot_scale=1.0, seed=RNG_SEED):
    """Run full 2023-2024 test data as one continuous episode. Returns result dict."""
    scaled_cfg = cfg.model_copy(deep=True)
    scaled_cfg.lot_size       = cfg.lot_size * lot_scale
    scaled_cfg.episode_bars   = len(features) - 1
    scaled_cfg.friction.randomize = False

    if stress_hooks:
        env = StressEnv(features, prices, scaled_cfg, stress_hooks=stress_hooks, rng_seed=seed)
    else:
        env = ForexTradingEnv(features, prices, scaled_cfg)

    trades      = []
    equity_bars = []

    orig_record = env._record_trade
    def patched_record(pnl, sl_hit=False, tp_hit=False):
        trades.append({"bar": env.cur_idx, "pnl": pnl, "sl": sl_hit, "tp": tp_hit, "win": pnl > 0})
        orig_record(pnl, sl_hit=sl_hit, tp_hit=tp_hit)
    env._record_trade = patched_record

    obs, _ = env.reset(seed=seed)
    env.ep_start = 0
    env.cur_idx  = 0
    obs = env._obs()

    done = False
    while not done:
        equity_bars.append(env._equity())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    equity_bars.append(env._equity())

    equity = np.array(equity_bars, dtype=np.float32)
    peak   = np.maximum.accumulate(equity)
    dd_ser = (peak - equity) / np.maximum(peak, 1e-8)

    wins   = sum(1 for t in trades if t["win"])
    n_tr   = len(trades)
    sl_hits = sum(1 for t in trades if t["sl"])
    tp_hits = sum(1 for t in trades if t["tp"])

    return {
        "pnl":       float(equity[-1] - equity[0]),
        "ret_pct":   float((equity[-1] - equity[0]) / equity[0] * 100),
        "max_dd":    float(dd_ser.max()),
        "worst_day": float(_worst_window_dd(equity, BARS_PER_DAY)),
        "worst_week":float(_worst_window_dd(equity, BARS_PER_WEEK)),
        "trades":    n_tr,
        "win_pct":   wins / n_tr * 100 if n_tr else 0.0,
        "sl_hits":   sl_hits,
        "tp_hits":   tp_hits,
        "equity":    equity,
        "dd_series": dd_ser,
        "trades_list": trades,
    }


def _worst_window_dd(equity, window):
    worst = 0.0
    step  = max(window // 4, 1)
    for s in range(0, len(equity) - window, step):
        chunk = equity[s:s + window]
        pk    = np.maximum.accumulate(chunk)
        worst = max(worst, float(((pk - chunk) / np.maximum(pk, 1e-8)).max()))
    return worst


def monthly_breakdown(result, initial_capital):
    """Split equity curve into 24 monthly slices, return per-month stats."""
    equity = result["equity"]
    trades = result["trades_list"]
    n      = len(equity)
    bpm    = n // 24
    rows   = []
    running = initial_capital
    for i, label in enumerate(TEST_MONTHS):
        s = i * bpm
        e = s + bpm if i < 23 else n
        chunk = equity[s:e]
        pnl   = float(chunk[-1] - chunk[0])
        pk    = np.maximum.accumulate(chunk)
        dd    = float(((pk - chunk) / np.maximum(pk, 1e-8)).max())
        month_trades = [t for t in trades if s <= t["bar"] < e]
        nt    = len(month_trades)
        wins  = sum(1 for t in month_trades if t["win"])
        sl    = sum(1 for t in month_trades if t["sl"])
        tp    = sum(1 for t in month_trades if t["tp"])
        ret   = pnl / running * 100 if running > 0 else 0.0
        running += pnl
        rows.append({"month": label, "pnl": pnl, "ret_pct": ret,
                     "trades": nt, "win_pct": wins / nt * 100 if nt else 0.0,
                     "sl": sl, "tp": tp, "capital": running, "month_dd": dd})
    return rows


# ── Report builder ───────────────────────────────────────────────────────────

SEP  = "=" * 78
DASH = "-" * 78

def fmt_row(label, r, show_dd=False):
    dd_col = f"  {r['max_dd']*100:>6.2f}%" if show_dd else ""
    ruin   = "  RUIN" if r.get("ruin") else ""
    return (f"  {label:<28}  {r['pnl']:>+9.2f}  {r['ret_pct']:>6.2f}%"
            f"  {r['trades']:>6}  {r['win_pct']:>5.1f}%"
            f"  {r['worst_day']*100:>6.2f}%  {r['worst_week']*100:>6.2f}%"
            f"{dd_col}{ruin}")

HDR = (f"  {'Scenario':<28}  {'P&L ($)':>9}  {'Ret%':>6}"
       f"  {'Trades':>6}  {'Win%':>5}  {'WrstDay':>7}  {'WrstWk':>6}"
       f"  {'MaxDD%':>7}")


def write_report(path, model_path, baseline, spread_rows, slip_rows,
                 delay_rows, skip_rows, sl_fill_rows, news_rows,
                 worstcase_rows, scale_rows, monthly_rows):

    L = []

    def sec(title):
        L.extend(["", SEP, f"  {title}", SEP])

    L += [
        "AQRF v10 — Comprehensive Stress Evaluation Report",
        f"Model  : {model_path}",
        f"Data   : data/processed/test_features.npy  (2023-01-01 to 2024-12-31)",
        f"Eval friction (baseline): spread=0.5 pip, slip=0.3 pip, delay=1 bar",
        f"R:R    : SL=10 pips / TP=20 pips (1:2)  |  Break-even win rate = 33.3%",
        f"Lot    : $10,000 micro-lot (1 pip = $1.00)",
        "",
        "All tests run on the SAME untouched test data — no retraining, no tuning.",
        "Capital resets to $10,000 at the start of each scenario for clean comparison.",
    ]

    # ── Baseline ──────────────────────────────────────────────────────────────
    sec("BASELINE  (spread=0.5, slip=0.3, delay=1)")
    L.append(HDR)
    L.append(DASH)
    L.append(fmt_row("Baseline", baseline, show_dd=True))

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    sec("MONTHLY BREAKDOWN — BASELINE (2023-2024 continuous)")
    L.append(f"  {'Month':<12}  {'P&L ($)':>9}  {'Ret%':>6}  {'Trades':>7}"
             f"  {'Win%':>6}  {'SL':>4}  {'TP':>4}  {'MthDD%':>7}  {'Cumul$':>10}")
    L.append(DASH)
    for r in monthly_rows:
        mk = "+" if r["pnl"] > 0 else ("-" if r["pnl"] < 0 else " ")
        L.append(f"  {r['month']:<12}  {r['pnl']:>+9.2f}  {r['ret_pct']:>+6.2f}%"
                 f"  {r['trades']:>7}  {r['win_pct']:>5.1f}%"
                 f"  {r['sl']:>4}  {r['tp']:>4}"
                 f"  {r['month_dd']*100:>6.2f}%  {r['capital']:>10.2f} {mk}")
    total_pnl = sum(r["pnl"] for r in monthly_rows)
    total_tr  = sum(r["trades"] for r in monthly_rows)
    total_wins= sum(int(r["trades"] * r["win_pct"] / 100 + 0.5) for r in monthly_rows)
    total_sl  = sum(r["sl"] for r in monthly_rows)
    total_tp  = sum(r["tp"] for r in monthly_rows)
    total_ret = total_pnl / INITIAL_CAPITAL * 100
    total_wp  = total_wins / total_tr * 100 if total_tr else 0.0
    final_cap = monthly_rows[-1]["capital"]
    profitable = sum(1 for r in monthly_rows if r["pnl"] > 0)
    L.append(DASH)
    L.append(f"  {'TOTAL':<12}  {total_pnl:>+9.2f}  {total_ret:>+6.2f}%"
             f"  {total_tr:>7}  {total_wp:>5.1f}%"
             f"  {total_sl:>4}  {total_tp:>4}  {'':>7}  {final_cap:>10.2f}")
    L += ["",
          f"  Profitable months : {profitable} / 24  ({profitable/24*100:.1f}%)",
          f"  Best month        : {max(monthly_rows, key=lambda r: r['pnl'])['month']}  "
          f"${max(r['pnl'] for r in monthly_rows):>+.2f}",
          f"  Worst month       : {min(monthly_rows, key=lambda r: r['pnl'])['month']}  "
          f"${min(r['pnl'] for r in monthly_rows):>+.2f}",
          f"  Avg trades/month  : {total_tr / 24:.1f}",
          f"  Overall win rate  : {total_wp:.1f}%",
          f"  Net P&L (2 years) : ${total_pnl:>+.2f}  ({total_ret:>+.2f}% on $10,000)",
    ]

    def stress_section(title, rows):
        sec(title)
        L.append(HDR)
        L.append(DASH)
        for label, r in rows:
            L.append(fmt_row(label, r, show_dd=True))

    # ── Spread ladder ─────────────────────────────────────────────────────────
    stress_section("STRESS 1 — SPREAD LADDER  (slip=0.3, delay=1)", spread_rows)

    # ── Slippage ladder ───────────────────────────────────────────────────────
    stress_section("STRESS 2 — SLIPPAGE LADDER  (spread=0.5, delay=1)", slip_rows)

    # ── Delay ladder ──────────────────────────────────────────────────────────
    stress_section("STRESS 3 — EXECUTION DELAY  (spread=0.5, slip=0.3)", delay_rows)

    # ── Skipped fills ─────────────────────────────────────────────────────────
    stress_section("STRESS 4 — RANDOM SKIPPED FILLS  (spread=0.5, slip=0.3, delay=1)", skip_rows)

    # ── Worse SL fill ─────────────────────────────────────────────────────────
    stress_section("STRESS 5 — WORSE SL FILL  (SL slippage multiplied)", sl_fill_rows)

    # ── News spike ────────────────────────────────────────────────────────────
    stress_section("STRESS 6 — NEWS SPREAD SPIKE  (5x spread, 3-bar, 0.5% of bars)", news_rows)

    # ── Combined worst-case ───────────────────────────────────────────────────
    stress_section("STRESS 7 — COMBINED WORST-CASE  (wide spread + high slip + 2x SL fill)", worstcase_rows)

    # ── Risk scale ────────────────────────────────────────────────────────────
    sec("STRESS 8 — RISK SCALE TEST  (baseline friction, lot size x1 to x5)")
    L.append(f"  {'Scale':<10}  {'Lot ($)':>9}  {'P&L ($)':>10}  {'Ret%':>7}"
             f"  {'MaxDD%':>7}  {'WrstDay':>8}  {'WrstWk':>7}  {'Win%':>6}  {'Ruin?':>6}")
    L.append(DASH)
    for r in scale_rows:
        ruin = "YES ⚠" if r["ruin"] else "no"
        L.append(f"  {r['label']:<10}  {r['lot']:>9,.0f}  {r['pnl']:>+10.2f}"
                 f"  {r['ret_pct']:>6.2f}%  {r['max_dd']*100:>6.2f}%"
                 f"  {r['worst_day']*100:>7.2f}%  {r['worst_week']*100:>6.2f}%"
                 f"  {r['win_pct']:>5.1f}%  {ruin:>6}")

    # ── 2025-2026 placeholder ─────────────────────────────────────────────────
    sec("SECTION: 2025-2026 OUT-OF-SAMPLE VALIDATION")
    L += [
        "  2025-2026 data not available in current processed dataset.",
        "  This section is reserved for future external validation once new",
        "  unseen data is added.",
        "",
        "  Protocol when data becomes available:",
        "    - Do NOT retrain on 2025-2026 data.",
        "    - Do NOT tune any hyperparameter using 2025-2026 data.",
        "    - Run eval_stress.py once on it as final untouched validation.",
        "    - That single run becomes the definitive proof of generalization.",
        "",
        "  No 2025-2026 results are reported here.",
    ]

    L += ["", SEP, "  END OF REPORT", SEP]
    path.write_text("\n".join(L))
    print(f"\n  Report saved: {path}")


# ── Chart ────────────────────────────────────────────────────────────────────

def save_chart(baseline, spread_rows, slip_rows, scale_rows,
               monthly_rows, out_dir, tag):

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle(f"AQRF {tag} — Stress Evaluation (2023-2024)",
                 fontsize=13, fontweight="bold")

    # 1. Baseline equity curve
    ax = axes[0, 0]
    eq = baseline["equity"]
    ax.plot(eq, color="navy", linewidth=0.8)
    ax.axhline(INITIAL_CAPITAL, color="red", linewidth=0.8, linestyle="--", label="$10,000 start")
    ax.fill_between(range(len(eq)), INITIAL_CAPITAL, eq,
                    where=eq >= INITIAL_CAPITAL, alpha=0.15, color="steelblue")
    ax.fill_between(range(len(eq)), INITIAL_CAPITAL, eq,
                    where=eq < INITIAL_CAPITAL, alpha=0.2, color="tomato")
    ax.set_title("Baseline Equity Curve (floating, 2023-2024)")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Baseline drawdown
    ax = axes[0, 1]
    ax.fill_between(range(len(baseline["dd_series"])),
                    0, baseline["dd_series"] * 100, color="tomato", alpha=0.7)
    ax.set_title("Baseline Drawdown %")
    ax.set_ylabel("Drawdown (%)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

    # 3. Spread ladder — P&L
    ax = axes[1, 0]
    labels = [l for l, _ in spread_rows]
    pnls   = [r["pnl"] for _, r in spread_rows]
    colors = ["steelblue" if p >= 0 else "tomato" for p in pnls]
    ax.bar(labels, pnls, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Spread Ladder — Net P&L ($)")
    ax.set_ylabel("P&L ($)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 4. Slippage ladder — P&L
    ax = axes[1, 1]
    labels = [l for l, _ in slip_rows]
    pnls   = [r["pnl"] for _, r in slip_rows]
    colors = ["steelblue" if p >= 0 else "tomato" for p in pnls]
    ax.bar(labels, pnls, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Slippage Ladder — Net P&L ($)")
    ax.set_ylabel("P&L ($)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 5. Risk scale — P&L
    ax = axes[2, 0]
    labels = [r["label"] for r in scale_rows]
    pnls   = [r["pnl"] for r in scale_rows]
    colors = ["steelblue" if p >= 0 else "tomato" for p in pnls]
    ax.bar(labels, pnls, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Risk Scale — Net P&L ($)")
    ax.set_ylabel("P&L ($)")
    ax.grid(axis="y", alpha=0.3)

    # 6. Risk scale — Max DD
    ax = axes[2, 1]
    dds = [r["max_dd"] * 100 for r in scale_rows]
    colors = ["tomato" if d > 10.0 else "steelblue" for d in dds]
    ax.bar(labels, dds, color=colors, edgecolor="white")
    ax.axhline(10.0, color="red", linewidth=1.0, linestyle="--", label="10% ruin threshold")
    ax.set_title("Risk Scale — Max Drawdown %")
    ax.set_ylabel("Max DD (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"{tag}_stress_eval.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart  saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="checkpoints/rl/phase3/20260529_145135/best/best_model.zip")
    parser.add_argument("--tag", type=str, default="v10")
    args = parser.parse_args()

    global _live_log
    cfg     = load_config()
    rl      = cfg.rl
    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    live_log_path = out_dir / f"{args.tag}_stress_live.txt"
    _live_log = open(live_log_path, "w", buffering=1)  # line-buffered
    log(f"AQRF {args.tag} Stress Eval — Live Log")
    log(f"Model : {args.model}")
    log(f"Log   : {live_log_path}")
    log(SEP)

    features = np.array(np.load("data/processed/test_features.npy", mmap_mode="r"))
    prices   = np.array(np.load("data/processed/test_prices.npy",   mmap_mode="r"))
    logger.info("data_loaded", shape=features.shape)

    model = PPO.load(str(args.model))
    logger.info("model_loaded", path=args.model)

    def run(label, spread=0.5, slip=0.3, delay=1, hooks=None, lot_scale=1.0):
        log(f"  >> Running: {label} ...")
        c = rl.model_copy(deep=True)
        c.friction.eval_spread_pips   = spread
        c.friction.eval_slippage_pips = slip
        c.friction.eval_delay_bars    = delay
        r = run_episode(features, prices, c, model, stress_hooks=hooks, lot_scale=lot_scale)
        r["ruin"] = r["equity"].min() < INITIAL_CAPITAL * (1 - rl.max_drawdown)
        log(f"  {label:<32}  P&L={r['pnl']:>+8.2f}  MaxDD={r['max_dd']*100:.2f}%"
            f"  Trades={r['trades']}  Win={r['win_pct']:.1f}%"
            + ("  *** RUIN ***" if r["ruin"] else "  OK"))
        return r

    # ── Baseline ──────────────────────────────────────────────────────────────
    log("\n" + SEP)
    log("  BASELINE")
    log(DASH)
    baseline = run("Baseline (0.5 sp / 0.3 sl / 1 bar)")
    monthly_rows = monthly_breakdown(baseline, INITIAL_CAPITAL)
    log("\n  Monthly breakdown:")
    for r in monthly_rows:
        mk = "+" if r["pnl"] > 0 else "-"
        log(f"    {r['month']}  P&L={r['pnl']:>+8.2f}  Trades={r['trades']}  Win={r['win_pct']:.1f}%  MthDD={r['month_dd']*100:.2f}%  {mk}")

    # ── Stress 1: Spread ladder ───────────────────────────────────────────────
    log("\n" + SEP)
    log("  STRESS 1 — SPREAD LADDER")
    log(DASH)
    spread_rows = []
    for sp in [0.5, 1.0, 1.5, 2.0, 3.0]:
        r = run(f"Spread {sp} pips", spread=sp)
        spread_rows.append((f"Spread={sp} pip", r))

    # ── Stress 2: Slippage ladder ─────────────────────────────────────────────
    log("\n" + SEP)
    log("  STRESS 2 — SLIPPAGE LADDER")
    log(DASH)
    slip_rows = []
    for sl in [0.3, 0.5, 1.0, 1.5, 2.0]:
        r = run(f"Slippage {sl} pips", slip=sl)
        slip_rows.append((f"Slip={sl} pip", r))

    # ── Stress 3: Execution delay ─────────────────────────────────────────────
    log("\n" + SEP)
    log("  STRESS 3 — EXECUTION DELAY")
    log(DASH)
    delay_rows = []
    for d in [1, 2, 3]:
        r = run(f"Delay {d} bars", delay=d)
        delay_rows.append((f"Delay={d} bar{'s' if d > 1 else ''}", r))

    # ── Stress 4: Random skipped fills ───────────────────────────────────────
    log("\n" + SEP)
    log("  STRESS 4 — RANDOM SKIPPED FILLS")
    log(DASH)
    skip_rows = []
    for prob in [0.05, 0.15]:
        r = run(f"Skip {int(prob*100)}% fills",
                hooks={"skip_fill_prob": prob})
        skip_rows.append((f"Skip {int(prob*100)}% fills", r))

    # ── Stress 5: Worse SL fill ───────────────────────────────────────────────
    log("\n" + SEP)
    log("  STRESS 5 — WORSE SL FILL")
    log(DASH)
    sl_fill_rows = []
    for mult in [1.5, 2.0, 3.0]:
        r = run(f"SL fill {mult}x slip",
                hooks={"sl_fill_mult": mult})
        sl_fill_rows.append((f"SL slip x{mult}", r))

    # ── Stress 6: News spread spike ───────────────────────────────────────────
    log("\n" + SEP)
    log("  STRESS 6 — NEWS SPREAD SPIKE")
    log(DASH)
    news_rows = []
    for prob, mult in [(0.005, 5.0), (0.01, 5.0), (0.005, 10.0)]:
        label = f"Spike {int(prob*100*10)/10}% prob x{mult}"
        r = run(label,
                hooks={"news_spike_prob": prob,
                       "news_spike_mult": mult,
                       "news_spike_bars": 3})
        news_rows.append((label, r))

    # ── Stress 7: Combined worst-case (high spread + high slip + 2x SL fill) ────
    # Note: env only has close prices, not intra-bar OHLC, so true same-candle
    # SL+TP detection is not possible. This tests the combined impact of wide
    # spread, high slippage, and 2x SL fill slippage simultaneously.
    log("\n" + SEP)
    log("  STRESS 7 — COMBINED WORST-CASE (wide spread + high slip + 2x SL fill)")
    log(DASH)
    worstcase_rows = []
    for spread, slip in [(1.0, 1.0), (2.0, 2.0), (3.0, 2.0)]:
        label = f"Combined sp={spread} sl={slip} SL×2"
        r = run(label, spread=spread, slip=slip,
                hooks={"sl_fill_mult": 2.0})
        worstcase_rows.append((label, r))

    # ── Stress 8: Risk scale ──────────────────────────────────────────────────
    log("\n" + SEP)
    log("  STRESS 8 — RISK SCALE")
    log(DASH)
    scale_rows = []
    for scale in [1, 2, 3, 4, 5]:
        r = run(f"{scale}x lot size", lot_scale=scale)
        r["label"] = f"{scale}x"
        r["lot"]   = rl.lot_size * scale
        scale_rows.append(r)

    # ── Save outputs ──────────────────────────────────────────────────────────
    log("\n" + SEP)
    log("  SAVING CHARTS AND REPORT...")
    save_chart(baseline, spread_rows, slip_rows, scale_rows,
               monthly_rows, out_dir, args.tag)

    report_path = out_dir / f"{args.tag}_stress_report.txt"
    write_report(report_path, args.model,
                 baseline, spread_rows, slip_rows, delay_rows,
                 skip_rows, sl_fill_rows, news_rows, worstcase_rows,
                 scale_rows, monthly_rows)
    log("  STRESS EVAL COMPLETE")
    if _live_log:
        _live_log.close()


if __name__ == "__main__":
    main()
