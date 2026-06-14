"""Stress Testing & Chaos Simulation for the PPO agent (Module 5 — Phase 3).

Intentionally attempts to break the agent under hostile conditions:
  1. Execution chaos   — extreme spread, slippage, delays
  2. Flash crashes     — synthetic price drops injected into the price series
  3. Spread spikes     — sudden temporary spread explosions
  4. Volatility explosions — price moves scaled up by a random multiplier
  5. Combined chaos    — all stressors simultaneously

Metrics: ruin probability, CVaR (tail risk), worst-case drawdown,
         Sharpe degradation vs baseline, recovery rate.

Usage:
    tester = RLStressTester(model_path, features, prices, config)
    baseline = tester.run_baseline()
    result   = tester.run_execution_chaos()
    tester.save_report(Path("eval_results/stress_report.md"))
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import structlog
from stable_baselines3 import PPO

from src.rl.environment import ForexTradingEnv
from src.utils.config import RLConfig, StressConfig

logger = structlog.get_logger(__name__)


class RLStressTester:
    """Monte Carlo stress tester for the PPO RL agent.

    Args:
        model_path:          Path to best_model.zip.
        features:            Feature sequences (N, seq_len, n_features).
        prices:              Price array (N, 2) [bid, ask].
        base_config:         RLConfig with friction.randomize=False baseline.
        stress_config:       StressConfig with scenario parameters.
        n_trials:            Monte Carlo trials per scenario (overrides stress_config).
        n_episodes_per_trial: Episodes per trial.
        seed:                RNG seed for reproducibility.
    """

    def __init__(
        self,
        model_path:           Path,
        features:             np.ndarray,
        prices:               np.ndarray,
        base_config:          Optional[RLConfig]   = None,
        stress_config:        Optional[StressConfig] = None,
        n_trials:             Optional[int] = None,
        n_episodes_per_trial: Optional[int] = None,
        seed:                 int = 42,
    ):
        self.model_path   = Path(model_path)
        self.features     = features
        self.prices       = prices
        self.cfg          = (base_config or RLConfig()).model_copy(deep=True)
        self.cfg.friction.randomize = False  # baseline = deterministic
        self.stress_cfg   = stress_config or StressConfig()
        self.n_trials     = n_trials or self.stress_cfg.n_trials
        self.n_eps        = n_episodes_per_trial or self.stress_cfg.n_episodes_per_trial
        self.rng          = np.random.default_rng(seed)

        self._baseline: Optional[dict] = None
        self._scenario_results: dict[str, dict] = {}

        logger.info("stress_tester_init", n_trials=self.n_trials, n_eps=self.n_eps)

    # ── Baseline ──────────────────────────────────────────────────────

    def run_baseline(self) -> dict:
        """Run clean evaluation — no chaos injection."""
        logger.info("running_baseline")
        cfg = self.cfg.model_copy(deep=True)
        cfg.friction.randomize = False
        results = self._run_scenario(cfg, price_modifier=None, n_trials=min(50, self.n_trials))
        metrics = self._compute_scenario_metrics(results, baseline=None)
        self._baseline = metrics
        self._scenario_results["baseline"] = metrics
        logger.info("baseline_done", **{k: v for k, v in metrics.items() if isinstance(v, float)})
        return metrics

    # ── Execution chaos ───────────────────────────────────────────────

    def run_execution_chaos(
        self,
        spread_range:   tuple[float, float] = (1.0, 10.0),
        slippage_range: tuple[float, float] = (0.5, 5.0),
        delay_range:    tuple[int,   int]   = (0, 10),
    ) -> dict:
        """Randomize execution far beyond training distribution."""
        logger.info("running_execution_chaos")
        cfg = self.cfg.model_copy(deep=True)
        cfg.friction.randomize        = True
        cfg.friction.spread_min_pips  = spread_range[0]
        cfg.friction.spread_max_pips  = spread_range[1]
        cfg.friction.slippage_min_pips = slippage_range[0]
        cfg.friction.slippage_max_pips = slippage_range[1]
        cfg.friction.delay_min_bars   = delay_range[0]
        cfg.friction.delay_max_bars   = delay_range[1]
        cfg.friction.fill_quality_min = 0.3
        cfg.friction.fill_quality_max = 0.8

        results = self._run_scenario(cfg, price_modifier=None, n_trials=self.n_trials)
        metrics = self._compute_scenario_metrics(results, self._baseline)
        self._scenario_results["execution_chaos"] = metrics
        return metrics

    # ── Flash crashes ─────────────────────────────────────────────────

    def run_flash_crash(
        self,
        magnitude_range: tuple[float, float] = None,
        duration_bars:   int   = None,
        probability:     float = None,
    ) -> dict:
        """Inject synthetic flash crashes into the price series."""
        logger.info("running_flash_crash")
        sc  = self.stress_cfg
        mag_min  = (magnitude_range or (sc.flash_crash_pips_min, sc.flash_crash_pips_max))[0]
        mag_max  = (magnitude_range or (sc.flash_crash_pips_min, sc.flash_crash_pips_max))[1]
        dur      = duration_bars or sc.flash_crash_duration_bars
        prob     = probability   or sc.flash_crash_probability
        recovery = sc.flash_recovery_fraction

        def modifier(prices: np.ndarray) -> np.ndarray:
            p = prices.copy()
            for i in range(len(p)):
                if self.rng.random() < prob:
                    drop = self.rng.uniform(mag_min, mag_max) * 0.0001
                    for j in range(i, min(i + dur, len(p))):
                        frac = (j - i + 1) / dur
                        p[j] -= drop * frac
                    # partial recovery
                    recover_end = min(i + dur * 3, len(p))
                    for j in range(i + dur, recover_end):
                        p[j] += drop * recovery * ((j - i - dur) / (dur * 2))
            return p

        results = self._run_scenario(self.cfg, price_modifier=modifier, n_trials=self.n_trials)
        metrics = self._compute_scenario_metrics(results, self._baseline)
        self._scenario_results["flash_crash"] = metrics
        return metrics

    # ── Spread spikes ─────────────────────────────────────────────────

    def run_spread_spike(
        self,
        spike_range: tuple[float, float] = None,
        probability: float = None,
        duration:    int   = 1,
    ) -> dict:
        """Inject sudden spread spikes (e.g. news events)."""
        logger.info("running_spread_spike")
        sc       = self.stress_cfg
        sp_min   = (spike_range or (sc.spread_spike_pips_min, sc.spread_spike_pips_max))[0]
        sp_max   = (spike_range or (sc.spread_spike_pips_min, sc.spread_spike_pips_max))[1]
        prob     = probability or sc.spread_spike_probability

        cfg = self.cfg.model_copy(deep=True)
        cfg.friction.randomize = True
        # Base spread is normal but we inject spikes via price modifier
        cfg.friction.spread_min_pips = self.cfg.friction.eval_spread_pips
        cfg.friction.spread_max_pips = self.cfg.friction.eval_spread_pips

        def modifier(prices: np.ndarray) -> np.ndarray:
            p = prices.copy()
            for i in range(len(p)):
                if self.rng.random() < prob:
                    spike = self.rng.uniform(sp_min, sp_max) * 0.0001
                    for j in range(i, min(i + duration, len(p))):
                        # Widen bid-ask by spike amount
                        mid = p[j].mean()
                        p[j, 0] = mid - spike / 2   # bid
                        p[j, 1] = mid + spike / 2   # ask
            return p

        results = self._run_scenario(cfg, price_modifier=modifier, n_trials=self.n_trials)
        metrics = self._compute_scenario_metrics(results, self._baseline)
        self._scenario_results["spread_spike"] = metrics
        return metrics

    # ── Volatility explosion ──────────────────────────────────────────

    def run_volatility_explosion(
        self,
        multiplier_range: tuple[float, float] = None,
        duration_range:   tuple[int,   int]   = None,
        probability:      float = None,
    ) -> dict:
        """Scale price moves by a random multiplier for burst periods."""
        logger.info("running_volatility_explosion")
        sc      = self.stress_cfg
        mul_min = (multiplier_range or (sc.vol_explosion_multiplier_min, sc.vol_explosion_multiplier_max))[0]
        mul_max = (multiplier_range or (sc.vol_explosion_multiplier_min, sc.vol_explosion_multiplier_max))[1]
        dur_min = (duration_range   or (sc.vol_explosion_duration_min,   sc.vol_explosion_duration_max))[0]
        dur_max = (duration_range   or (sc.vol_explosion_duration_min,   sc.vol_explosion_duration_max))[1]
        prob    = probability or sc.vol_explosion_probability

        def modifier(prices: np.ndarray) -> np.ndarray:
            p    = prices.copy()
            i    = 0
            while i < len(p):
                if self.rng.random() < prob:
                    multiplier = self.rng.uniform(mul_min, mul_max)
                    duration   = int(self.rng.integers(dur_min, dur_max + 1))
                    end        = min(i + duration, len(p))
                    mid_base   = p[i].mean()
                    for j in range(i, end):
                        mid   = p[j].mean()
                        delta = mid - mid_base
                        new_mid = mid_base + delta * multiplier
                        half_spread = (p[j, 1] - p[j, 0]) / 2
                        p[j, 0] = new_mid - half_spread
                        p[j, 1] = new_mid + half_spread
                        mid_base = new_mid
                    i = end
                else:
                    i += 1
            return p

        results = self._run_scenario(self.cfg, price_modifier=modifier, n_trials=self.n_trials)
        metrics = self._compute_scenario_metrics(results, self._baseline)
        self._scenario_results["volatility_explosion"] = metrics
        return metrics

    # ── Combined chaos ────────────────────────────────────────────────

    def run_combined_chaos(self) -> dict:
        """All stressors simultaneously — worst-case scenario."""
        logger.info("running_combined_chaos")
        cfg = self.cfg.model_copy(deep=True)
        cfg.friction.randomize        = True
        cfg.friction.spread_min_pips  = 1.0
        cfg.friction.spread_max_pips  = 8.0
        cfg.friction.slippage_min_pips = 0.5
        cfg.friction.slippage_max_pips = 4.0
        cfg.friction.delay_min_bars   = 0
        cfg.friction.delay_max_bars   = 5
        cfg.friction.fill_quality_min = 0.4
        cfg.friction.fill_quality_max = 0.9
        sc = self.stress_cfg

        def modifier(prices: np.ndarray) -> np.ndarray:
            p = prices.copy()
            # Flash crashes
            for i in range(len(p)):
                if self.rng.random() < sc.flash_crash_probability:
                    drop = self.rng.uniform(sc.flash_crash_pips_min, sc.flash_crash_pips_max) * 0.0001
                    for j in range(i, min(i + sc.flash_crash_duration_bars, len(p))):
                        p[j] -= drop * 0.5
            # Volatility explosions
            i = 0
            while i < len(p):
                if self.rng.random() < sc.vol_explosion_probability:
                    mul = self.rng.uniform(sc.vol_explosion_multiplier_min, sc.vol_explosion_multiplier_max)
                    dur = int(self.rng.integers(sc.vol_explosion_duration_min, sc.vol_explosion_duration_max + 1))
                    end = min(i + dur, len(p))
                    mid_base = p[i].mean()
                    for j in range(i, end):
                        delta = p[j].mean() - mid_base
                        new_mid = mid_base + delta * mul
                        hs = (p[j, 1] - p[j, 0]) / 2
                        p[j] = [new_mid - hs, new_mid + hs]
                        mid_base = new_mid
                    i = end
                else:
                    i += 1
            return p

        results = self._run_scenario(cfg, price_modifier=modifier, n_trials=self.n_trials)
        metrics = self._compute_scenario_metrics(results, self._baseline)
        self._scenario_results["combined_chaos"] = metrics
        return metrics

    # ── Core Monte Carlo loop ─────────────────────────────────────────

    def _run_scenario(
        self,
        scenario_cfg:    RLConfig,
        price_modifier:  Optional[Callable[[np.ndarray], np.ndarray]],
        n_trials:        int,
    ) -> list[dict]:
        """Run n_trials episodes under scenario conditions."""
        model   = PPO.load(str(self.model_path))
        results = []

        n_bars  = len(self.features)
        ep_len  = scenario_cfg.episode_bars

        for trial in range(n_trials):
            # Random start position
            max_start = max(0, n_bars - ep_len - 1)
            start     = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0

            prices_slice   = self.prices[start:start + ep_len + 1].copy()
            features_slice = self.features[start:start + ep_len + 1]

            if price_modifier is not None:
                prices_slice = price_modifier(prices_slice)

            env = ForexTradingEnv(features_slice, prices_slice, config=scenario_cfg)
            obs, _ = env.reset(seed=trial)
            done   = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(action))
                done = term or trunc

            final_pnl = env.capital - scenario_cfg.initial_capital
            results.append({
                "pnl":        final_pnl,
                "return_pct": final_pnl / scenario_cfg.initial_capital * 100,
                "capital":    env.capital,
                "drawdown":   env._drawdown(),
                "win_rate":   env.winning / env.total_trades if env.total_trades > 0 else 0.0,
                "trades":     env.total_trades,
                "ruin":       env.capital < scenario_cfg.initial_capital * (1 - self.stress_cfg.ruin_threshold),
            })

        del model
        return results

    # ── Metrics aggregation ───────────────────────────────────────────

    def _compute_scenario_metrics(
        self, trial_results: list[dict], baseline: Optional[dict]
    ) -> dict:
        returns    = np.array([r["return_pct"] for r in trial_results])
        drawdowns  = np.array([r["drawdown"]   for r in trial_results])
        ruin_flags = np.array([r["ruin"]        for r in trial_results])

        tail_cutoff = int(len(returns) * self.stress_cfg.cvar_percentile)
        sorted_rets = np.sort(returns)
        cvar        = float(np.mean(sorted_rets[:max(1, tail_cutoff)]))

        sharpe = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0

        metrics = {
            "mean_return_pct":    float(np.mean(returns)),
            "std_return_pct":     float(np.std(returns)),
            "min_return_pct":     float(np.min(returns)),
            "max_return_pct":     float(np.max(returns)),
            "cvar_5":             cvar,
            "worst_drawdown":     float(np.max(drawdowns)),
            "mean_drawdown":      float(np.mean(drawdowns)),
            "ruin_probability":   float(np.mean(ruin_flags)),
            "sharpe":             sharpe,
            "n_trials":           len(trial_results),
        }

        if baseline:
            base_sharpe = baseline.get("sharpe", 1e-8)
            metrics["sharpe_degradation"] = (
                (base_sharpe - sharpe) / abs(base_sharpe) if base_sharpe != 0 else 0.0
            )
            metrics["return_degradation"] = (
                baseline["mean_return_pct"] - metrics["mean_return_pct"]
            )

        return metrics

    # ── Reporting ─────────────────────────────────────────────────────

    def generate_report(self) -> str:
        lines = [
            "# AQRF Phase 3 — Stress Test Report",
            "",
            f"Model: `{self.model_path}`  ",
            f"Trials per scenario: {self.n_trials}  ",
            f"Episodes per trial: {self.n_eps}  ",
            "",
            "---",
            "",
        ]

        scenario_order = ["baseline", "execution_chaos", "flash_crash",
                          "spread_spike", "volatility_explosion", "combined_chaos"]

        for name in scenario_order:
            if name not in self._scenario_results:
                continue
            m = self._scenario_results[name]
            lines.append(f"## {name.replace('_', ' ').title()}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean Return% | {m.get('mean_return_pct', 0):.2f}% |")
            lines.append(f"| Std Return%  | {m.get('std_return_pct',  0):.2f}% |")
            lines.append(f"| CVaR 5%      | {m.get('cvar_5',          0):.2f}% |")
            lines.append(f"| Worst DD%    | {m.get('worst_drawdown',  0)*100:.2f}% |")
            lines.append(f"| Ruin Prob    | {m.get('ruin_probability', 0)*100:.1f}% |")
            lines.append(f"| Sharpe       | {m.get('sharpe',           0):.2f} |")
            if "sharpe_degradation" in m:
                lines.append(f"| Sharpe Degradation | {m['sharpe_degradation']*100:.1f}% |")
            if "return_degradation" in m:
                lines.append(f"| Return Degradation | {m['return_degradation']:.2f}% |")
            lines.append("")

        return "\n".join(lines)

    def save_report(self, path: Path) -> None:
        report = self.generate_report()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(report, encoding="utf-8")
        logger.info("stress_report_saved", path=str(path))
