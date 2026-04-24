"""Sector ETF Momentum — Formal Strategy (entry_margin hysteresis).

Strategy
--------
Universe  : 9 risk-on sector / style ETFs (see RISK_ON_UNIVERSE)
Momentum  : 210-day price ROC (~10 calendar months)
Selection : top-3 ETFs by cross-sectional momentum rank
Weighting : equal weight
Rebalance : bi-monthly (every 2 months, 2ME), executed next business day
Hysteresis: entry_margin=0.02 — a new asset displaces the worst-ranked held asset
            only when its 210-day ROC exceeds the displaced asset's ROC by ≥ 2 pp.
            Consolidated from hysteresis study (2026-04-24):
              entry_margin  Sharpe=1.091  CAGR=12.47%  MaxDD=22.17%  ← winner
              no_hysteresis Sharpe=1.048  CAGR=12.11%  MaxDD=22.17%
Benchmark : SPY buy-and-hold

Run
---
    python experiments/sector_momentum_experiment.py

Requires
--------
    pip install 'quant-stack[research]'   pip install yfinance
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger

from quant_stack.core.schemas import DataConfig, ExperimentRecord, PortfolioWeights
from quant_stack.data.providers.yahoo import YahooProvider
from quant_stack.research.strategies.sector_momentum import (
    HysteresisMode,
    RISK_ON_UNIVERSE,
    SectorMomentumStrategy,
    apply_hysteresis,
)
from quant_stack.research.vbt_adapter import (
    VbtRunConfig,
    run_vbt_backtest,
    signal_frame_to_weights,
)
from quant_stack.signals.base import SignalFrame
from quant_stack.tracking import ExperimentTracker

# ── Config ─────────────────────────────────────────────────────────────────────

BENCHMARK    = "SPY"
PERIOD_START = date(2010, 1, 1)
PERIOD_END   = date(2025, 12, 31)

STRATEGY_PARAMS = {
    "momentum_window":  210,
    "top_n":            3,
    "rebalance_freq":   "2ME",
    "commission_bps":   10,
    "hysteresis_mode":  "entry_margin",
    "entry_margin":     0.02,          # new asset must beat displaced by ≥ 2pp ROC
}

# ── 1. Fetch data ──────────────────────────────────────────────────────────────

print("=" * 60)
print("  Sector ETF Momentum — Formal Strategy (entry_margin)")
print("=" * 60)

all_syms = list(dict.fromkeys(RISK_ON_UNIVERSE + [BENCHMARK]))
cfg = DataConfig(
    symbols=all_syms, start=PERIOD_START, end=PERIOD_END, cache_dir="./data"
)

logger.info(f"Fetching {len(RISK_ON_UNIVERSE)} risk-on ETFs + benchmark...")
raw = YahooProvider().fetch(cfg)

if isinstance(raw.columns, pd.MultiIndex):
    close_all = raw.xs("close", axis=1, level=1)
else:
    close_all = raw[["close"]].rename(columns={"close": all_syms[0]})

close_all = close_all.sort_index().dropna(how="all")
close     = close_all[RISK_ON_UNIVERSE].copy()
bm_close  = close_all[BENCHMARK].copy()

print(f"\n[DATA]  {close.shape[0]:,} trading days  "
      f"{close.index[0].date()} to {close.index[-1].date()}")
print(f"        Universe: {', '.join(RISK_ON_UNIVERSE)}")

# ── 2. Signals → hysteresis → weights ────────────────────────────────────────

strategy = SectorMomentumStrategy(
    momentum_window=STRATEGY_PARAMS["momentum_window"],
    top_n=STRATEGY_PARAMS["top_n"],
)
raw_signals, ranks, mom_scores = strategy.generate_signals_full(close)
signals = apply_hysteresis(
    raw_signals, ranks, mom_scores,
    mode=HysteresisMode.ENTRY_MARGIN,
    top_n=STRATEGY_PARAMS["top_n"],
    entry_margin=STRATEGY_PARAMS["entry_margin"],
)
sf      = SignalFrame(signals=signals, strength=signals.copy(),
                      strategy_name=strategy.name)
weights = signal_frame_to_weights(sf)

# ── 3. Backtest ────────────────────────────────────────────────────────────────

vbt_cfg = VbtRunConfig(
    commission=STRATEGY_PARAMS["commission_bps"] / 10_000,
    rebalance_freq=STRATEGY_PARAMS["rebalance_freq"],
    risk_free_rate=0.05,
)
result = run_vbt_backtest(
    close=close, weights=weights, config=vbt_cfg,
    benchmark_close=bm_close, strategy_name=strategy.name,
)

# ── 4. Print results ───────────────────────────────────────────────────────────

excess_sign = "+" if (result.excess_return or 0) >= 0 else ""
print(f"\n[RESULTS]  {strategy.name}")
print(f"  Total Return    : {result.total_return:.2%}")
print(f"  CAGR            : {result.cagr:.2%}")
if result.annual_volatility:
    print(f"  Annual Vol      : {result.annual_volatility:.2%}")
print(f"  Sharpe Ratio    : {result.sharpe_ratio:.3f}")
if result.sortino_ratio:
    print(f"  Sortino Ratio   : {result.sortino_ratio:.3f}")
print(f"  Max Drawdown    : {result.max_drawdown:.2%}")
if result.annual_turnover:
    print(f"  Annual Turnover : {result.annual_turnover:.2%}")
print(f"  Trades          : {result.n_trades:,}")
if result.commission_paid:
    print(f"  Commission Paid : ${result.commission_paid:,.0f}")
if result.benchmark_return is not None:
    print(f"\n  SPY Total Return: {result.benchmark_return:.2%}")
    if result.excess_return is not None:
        print(f"  Excess vs SPY   : {excess_sign}{result.excess_return:.2%}")

# ── 5. Last rebalance ─────────────────────────────────────────────────────────

last_date = signals.dropna(how="all").index[-1]
last_row  = signals.loc[last_date]
top_etfs  = last_row[last_row == 1.0].index.tolist()
print(f"\n[LAST REBALANCE]  {last_date.date()}: {', '.join(top_etfs)}")

# ── 6. Save ExperimentRecord ──────────────────────────────────────────────────

pw = PortfolioWeights(
    weights={sym: 1.0 / STRATEGY_PARAMS["top_n"] for sym in top_etfs},
    method="equal_weight",
    rebalance_date=last_date.date(),
)
bm_str = f"{result.benchmark_return:.2%}" if result.benchmark_return is not None else "N/A"
ex_str = (
    f" Excess vs SPY: {excess_sign}{result.excess_return:.2%}."
    if result.excess_return is not None else ""
)
record = ExperimentRecord(
    description=(
        f"Sector ETF rotation: top-{STRATEGY_PARAMS['top_n']} by "
        f"{STRATEGY_PARAMS['momentum_window']}-day momentum, bi-monthly rebalancing (2ME), "
        f"entry_margin hysteresis (≥{STRATEGY_PARAMS['entry_margin']:.0%} ROC gap to displace). "
        f"Universe: {', '.join(RISK_ON_UNIVERSE)}."
    ),
    strategy_params=STRATEGY_PARAMS,
    symbols=RISK_ON_UNIVERSE,
    period_start=result.period_start,
    period_end=result.period_end,
    backtest_result=result,
    portfolio_weights=pw,
    tags=["sector-rotation", "momentum", "etf", "bimonthly",
          f"top{STRATEGY_PARAMS['top_n']}", "equal-weight", "entry-margin", "baseline"],
    notes=f"Benchmark SPY: {bm_str}.{ex_str}",
)

tracker = ExperimentTracker("./experiments")
exp_dir = tracker.save(record)

print(f"\n[SAVED]  {exp_dir}")
print(f"         record.json : {(exp_dir / 'record.json').stat().st_size:,} bytes")
print(f"         report.md   : {(exp_dir / 'report.md').stat().st_size:,} bytes")

# ── 7. Registry ───────────────────────────────────────────────────────────────

print("\n[REGISTRY] sector-rotation experiments (newest first, up to 5):")
for entry in tracker.list_experiments(tag="sector-rotation", limit=5):
    m  = entry["metrics"]
    print(
        f"  {entry['created_at'][:19]}  {entry['strategy_name']:<44}"
        f"  CAGR={m.get('cagr', float('nan')):.2%}"
        f"  Sharpe={m.get('sharpe_ratio', float('nan')):.3f}"
    )
