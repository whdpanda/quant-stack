"""Demo: vbt backtest -> ExperimentRecord -> ExperimentTracker -> Markdown report.

Shows the full research workflow:
  1. Build synthetic price data
  2. Run signal pipeline (trend filter + momentum ranking)
  3. Run vbt backtest -> BacktestResult
  4. Wrap in ExperimentRecord with params, tags, notes
  5. Save via ExperimentTracker (writes record.json + report.md, updates registry)
  6. Browse registry and reload record
  7. Print the generated report

Run:
    python experiments/demo_tracking.py

Requires:
    pip install 'quant-stack[research]'
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date

from quant_stack.core.schemas import ExperimentRecord
from quant_stack.factors.momentum import momentum_63
from quant_stack.factors.trend import sma_200
from quant_stack.research.vbt_adapter import VbtRunConfig, run_vbt_backtest, signal_frame_to_weights
from quant_stack.signals.momentum import relative_momentum_ranking_signal
from quant_stack.signals.trend import as_eligibility_mask, trend_filter_signal
from quant_stack.tracking import ExperimentTracker, ReportGenerator

# ── 1. Synthetic universe ──────────────────────────────────────────────────────

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "IEF"]
rng = np.random.default_rng(42)
idx = pd.bdate_range("2018-01-02", periods=1_500)

annual_returns = {"SPY": 0.08, "QQQ": 0.12, "IWM": 0.07, "EFA": 0.05, "IEF": 0.03}
annual_vols    = {"SPY": 0.16, "QQQ": 0.20, "IWM": 0.18, "EFA": 0.18, "IEF": 0.06}

prices = {}
for sym in SYMBOLS:
    mu  = annual_returns[sym] / 252
    sig = annual_vols[sym]    / np.sqrt(252)
    log_ret = rng.normal(mu - 0.5 * sig**2, sig, 1_500)
    prices[sym] = 100.0 * np.exp(np.cumsum(log_ret))

close = pd.DataFrame(prices, index=idx)

# ── 2. Signal pipeline ────────────────────────────────────────────────────────

STRATEGY_PARAMS = {
    "momentum_window": 63,
    "trend_window": 200,
    "top_n": 2,
    "rebalance_freq": "ME",
    "commission_bps": 10,
}

mom      = momentum_63(close)
trend    = sma_200(close)
trend_sf = trend_filter_signal(close, trend)
eligible = as_eligibility_mask(trend_sf)
sf       = relative_momentum_ranking_signal(mom, top_n=STRATEGY_PARAMS["top_n"], eligible=eligible)
weights  = signal_frame_to_weights(sf)

# ── 3. vbt backtest ───────────────────────────────────────────────────────────

cfg = VbtRunConfig(
    commission=STRATEGY_PARAMS["commission_bps"] / 10_000,
    rebalance_freq=STRATEGY_PARAMS["rebalance_freq"],
)
result = run_vbt_backtest(
    close=close,
    weights=weights,
    config=cfg,
    benchmark_close=close["SPY"],
    strategy_name="trend_momentum_top2",
)

# ── 4. Build ExperimentRecord ─────────────────────────────────────────────────

record = ExperimentRecord(
    description=(
        "Dual-filter momentum strategy: SMA-200 trend eligibility + "
        "63-day cross-sectional momentum ranking. Top-2 assets, monthly rebalancing."
    ),
    strategy_params=STRATEGY_PARAMS,
    symbols=SYMBOLS,
    period_start=date(2018, 1, 2),
    period_end=date(2023, 10, 2),
    backtest_result=result,
    tags=["momentum", "trend-filter", "monthly", "synthetic"],
    notes=(
        "Outperforms SPY benchmark significantly on synthetic data. "
        "High turnover (~316%/yr) implies transaction costs are material; "
        "validate on real data before drawing conclusions. "
        "SMA-200 warm-up period (first ~200 bars) holds 100% cash."
    ),
)

# ── 5. Save via ExperimentTracker ─────────────────────────────────────────────

tracker = ExperimentTracker("./experiments")
exp_dir = tracker.save(record)

print(f"\n[SAVED] Experiment directory: {exp_dir}")
print(f"        record.json : {(exp_dir / 'record.json').stat().st_size:,} bytes")
print(f"        report.md   : {(exp_dir / 'report.md').stat().st_size:,} bytes")

# ── 6. Browse registry ────────────────────────────────────────────────────────

print("\n[REGISTRY] All experiments (newest first):")
for entry in tracker.list_experiments():
    m = entry["metrics"]
    sharpe = m.get("sharpe_ratio", float("nan"))
    total_r = m.get("total_return", float("nan"))
    print(
        f"  {entry['created_at']}  {entry['strategy_name']:<32}"
        f"  Sharpe={sharpe:.3f}  TotalReturn={total_r:.2%}"
        f"  tags={entry['tags']}"
    )

# ── 7. Reload and verify round-trip ───────────────────────────────────────────

reloaded = tracker.load(record.experiment_id)
assert reloaded.experiment_id == record.experiment_id
assert abs(reloaded.backtest_result.sharpe_ratio - result.sharpe_ratio) < 1e-9
print(f"\n[ROUNDTRIP] Loaded experiment {record.experiment_id[:8]}... OK")

# ── 8. Print the generated report ─────────────────────────────────────────────

report_path = exp_dir / "report.md"
print(f"\n{'=' * 60}")
print(f"  report.md ({report_path})")
print(f"{'=' * 60}\n")
print(report_path.read_text(encoding="utf-8"))
