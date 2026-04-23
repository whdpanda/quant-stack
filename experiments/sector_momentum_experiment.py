"""Sector ETF Momentum Strategy - Research Experiment.

Strategy logic
--------------
Universe  : 10 sector / style ETFs  (VNQ QQQ XLE XLV XLF XLI XLB SPY XLP IEF)
Momentum  : 210-day price ROC (approx. 10 calendar months)
Selection : top 3 ETFs by cross-sectional momentum rank each month
Weighting : equal weight (1/3 each)
Rebalance : monthly (month-end), executed next business day
Benchmark : SPY buy-and-hold

Run
---
    python experiments/sector_momentum_experiment.py

Requires
--------
    pip install 'quant-stack[research]'
    pip install yfinance
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd
from loguru import logger

from quant_stack.core.schemas import DataConfig, ExperimentRecord
from quant_stack.data.providers.yahoo import YahooProvider
from quant_stack.research.strategies.sector_momentum import SectorMomentumStrategy
from quant_stack.research.vbt_adapter import (
    VbtRunConfig,
    run_vbt_backtest,
    signal_frame_to_weights,
)
from quant_stack.signals.base import SignalFrame
from quant_stack.tracking import ExperimentTracker, ReportGenerator

# ── Configuration ──────────────────────────────────────────────────────────────

UNIVERSE = ["VNQ", "QQQ", "XLE", "XLV", "XLF", "XLI", "XLB", "SPY", "XLP", "IEF"]
BENCHMARK = "SPY"

PERIOD_START = date(2010, 1, 1)
PERIOD_END   = date(2025, 12, 31)

STRATEGY_PARAMS = {
    "momentum_window": 210,   # ~10 calendar months
    "top_n": 3,               # hold top-3 ETFs
    "rebalance_freq": "ME",   # month-end
    "commission_bps": 10,     # 10 bps round-trip
}

# ── 1. Fetch data ──────────────────────────────────────────────────────────────

print("=" * 60)
print("  Sector ETF Momentum - Research Experiment")
print("=" * 60)

all_symbols = UNIVERSE + ([BENCHMARK] if BENCHMARK not in UNIVERSE else [])

cfg = DataConfig(
    symbols=all_symbols,
    start=PERIOD_START,
    end=PERIOD_END,
    cache_dir="./data",
)

logger.info(f"Fetching {len(all_symbols)} symbols: {all_symbols}")
provider = YahooProvider()
raw = provider.fetch(cfg)

# Extract adjusted close prices.
# YahooProvider / pd.concat returns MultiIndex columns: (symbol, field).
if isinstance(raw.columns, pd.MultiIndex):
    close_all = raw.xs("close", axis=1, level=1)
else:
    # Flat columns: single-symbol or pre-normalised frame — shouldn't happen here
    close_all = raw[["close"]].rename(columns={"close": all_symbols[0]})

close_all = close_all.sort_index().dropna(how="all")

# Split universe close and benchmark series
close = close_all[UNIVERSE].copy()
benchmark_close = close_all[BENCHMARK].copy()

print(f"\n[DATA] {close.shape[0]:,} trading days x {close.shape[1]} ETFs")
print(f"       {close.index[0].date()} to {close.index[-1].date()}")
missing = close.isna().sum()
if missing.any():
    print(f"       Missing values: {missing[missing > 0].to_dict()}")

# ── 2. Build signals ───────────────────────────────────────────────────────────

strategy = SectorMomentumStrategy(
    momentum_window=STRATEGY_PARAMS["momentum_window"],
    top_n=STRATEGY_PARAMS["top_n"],
)

print(f"\n[SIGNALS] Computing {STRATEGY_PARAMS['momentum_window']}-day momentum signals...")
daily_signals = strategy.generate_signals(close)

warmup_bars = daily_signals.isna().all(axis=1).sum()
active_bars = (~daily_signals.isna().all(axis=1)).sum()
print(f"          Warmup bars (all-NaN): {warmup_bars:,}")
print(f"          Active signal bars: {active_bars:,}")

# Confirm equal-weight: on any active day exactly top_n ETFs have signal=1.0
sample_active = daily_signals.dropna(how="all")
long_counts = (sample_active == 1.0).sum(axis=1)
assert (long_counts == STRATEGY_PARAMS["top_n"]).all(), (
    f"Expected exactly {STRATEGY_PARAMS['top_n']} longs per row; "
    f"got: {long_counts.value_counts().to_dict()}"
)
print(f"          Signal check OK: always exactly {STRATEGY_PARAMS['top_n']} longs active")

# ── 3. Convert signals to weights ─────────────────────────────────────────────

sf = SignalFrame(
    signals=daily_signals,
    strength=daily_signals.copy(),   # equal strength -> equal weight after normalisation
    strategy_name=strategy.name,
)
weights = signal_frame_to_weights(sf)

# ── 4. Run backtest ────────────────────────────────────────────────────────────

vbt_cfg = VbtRunConfig(
    commission=STRATEGY_PARAMS["commission_bps"] / 10_000,
    rebalance_freq=STRATEGY_PARAMS["rebalance_freq"],
    risk_free_rate=0.05,
)

print(f"\n[BACKTEST] Running vectorbt simulation (rebalance={vbt_cfg.rebalance_freq})...")
result = run_vbt_backtest(
    close=close,
    weights=weights,
    config=vbt_cfg,
    benchmark_close=benchmark_close,
    strategy_name=strategy.name,
)

# ── 5. Print performance summary ──────────────────────────────────────────────

excess_sign = "+" if (result.excess_return or 0) >= 0 else ""

print(f"\n[RESULTS] {strategy.name}")
print(f"  Period          : {result.period_start} to {result.period_end}")
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
        print(f"  Excess Return   : {excess_sign}{result.excess_return:.2%}")

# ── 6. Show last rebalance holdings ──────────────────────────────────────────

last_signal_date = daily_signals.dropna(how="all").index[-1]
last_holdings = daily_signals.loc[last_signal_date]
top_etfs = last_holdings[last_holdings == 1.0].index.tolist()
print(f"\n[LAST REBALANCE] Date: {last_signal_date.date()}")
print(f"  Holdings ({len(top_etfs)}): {', '.join(top_etfs)}")

# ── 7. Build ExperimentRecord ──────────────────────────────────────────────────

# Capture last-rebalance weights for the record
last_weights_dict = {sym: 1.0 / STRATEGY_PARAMS["top_n"] for sym in top_etfs}

from quant_stack.core.schemas import PortfolioWeights

portfolio_weights = PortfolioWeights(
    weights=last_weights_dict,
    method="equal_weight",
    rebalance_date=last_signal_date.date(),
)

record = ExperimentRecord(
    description=(
        f"Sector ETF rotation: top-{STRATEGY_PARAMS['top_n']} by "
        f"{STRATEGY_PARAMS['momentum_window']}-day momentum, monthly rebalancing. "
        f"Universe: {', '.join(UNIVERSE)}."
    ),
    strategy_params=STRATEGY_PARAMS,
    symbols=UNIVERSE,
    period_start=result.period_start,
    period_end=result.period_end,
    backtest_result=result,
    portfolio_weights=portfolio_weights,
    tags=["sector-rotation", "momentum", "etf", "monthly", "top3", "equal-weight"],
    notes=(
        f"Benchmark SPY total return: {result.benchmark_return:.2%}. "
        f"Excess return vs SPY: {excess_sign}{result.excess_return:.2%}. "
        f"Strategy holds top-{STRATEGY_PARAMS['top_n']} sector ETFs by 12-month "
        f"price momentum, rebalanced monthly. "
        f"210-bar warmup period holds cash."
    ) if result.benchmark_return is not None else "",
)

# ── 8. Save via ExperimentTracker ─────────────────────────────────────────────

tracker = ExperimentTracker("./experiments")
exp_dir = tracker.save(record)

print(f"\n[SAVED] {exp_dir}")
print(f"        record.json : {(exp_dir / 'record.json').stat().st_size:,} bytes")
print(f"        report.md   : {(exp_dir / 'report.md').stat().st_size:,} bytes")

# ── 9. Registry summary ───────────────────────────────────────────────────────

print("\n[REGISTRY] All experiments (newest first):")
for entry in tracker.list_experiments():
    m = entry["metrics"]
    sharpe = m.get("sharpe_ratio", float("nan"))
    total_r = m.get("total_return", float("nan"))
    cagr    = m.get("cagr", float("nan"))
    print(
        f"  {entry['created_at'][:19]}  {entry['strategy_name']:<38}"
        f"  CAGR={cagr:.2%}  Sharpe={sharpe:.3f}  TotalRet={total_r:.2%}"
        f"  tags={entry['tags']}"
    )

# ── 10. Print the generated report ────────────────────────────────────────────

report_path = exp_dir / "report.md"
print(f"\n{'=' * 60}")
print(f"  report.md")
print(f"{'=' * 60}\n")
try:
    print(report_path.read_text(encoding="utf-8"))
except UnicodeDecodeError:
    print(report_path.read_text(encoding="utf-8", errors="replace"))
