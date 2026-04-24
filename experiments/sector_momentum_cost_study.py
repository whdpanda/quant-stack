"""Sector ETF Momentum — Cost Sensitivity Analysis.

Purpose
-------
Validate live-trading feasibility of the formal strategy by testing how
performance degrades across four cost assumptions.  The question is whether
the strategy is "cost-fragile" or robust enough to warrant further work.

Cost model
----------
VbtRunConfig has two one-way cost parameters:
    commission  : brokerage fee per trade (fraction of trade value)
    slippage    : market-impact / bid-ask half-spread per trade

We treat  total_cost_bps = commission_bps + slippage_bps  as the single
variable and split evenly between the two:

    0 bps  → commission=0.0000,  slippage=0.0000
   10 bps  → commission=0.0005,  slippage=0.0005
   20 bps  → commission=0.0010,  slippage=0.0010  ← formal strategy baseline
   30 bps  → commission=0.0015,  slippage=0.0015

All other parameters are IDENTICAL to the formal strategy:
    universe=VNQ QQQ XLE XLV XLF XLI VTV GDX XLP (GDX consolidated 2026-04-25)
    momentum_window=210d, top_n=3, rebalance_freq=2ME,
    entry_margin=0.02, weighting=blend_70_30, vol_window=63d.

Run
---
    python experiments/sector_momentum_cost_study.py

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
    WeightingScheme,
    apply_hysteresis,
    compute_strength,
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

BASE_PARAMS = {
    "momentum_window":  210,
    "top_n":            3,
    "rebalance_freq":   "2ME",
    "hysteresis_mode":  "entry_margin",
    "entry_margin":     0.02,
    "weighting_scheme": "blend_70_30",
    "vol_window":       63,
}

# total_cost_bps → (commission_bps, slippage_bps)  — split evenly
COST_VARIANTS: list[int] = [0, 10, 20, 30]

FORMAL_COST_BPS = 20   # the formal strategy runs at 20bps total

# ── 1. Fetch data ──────────────────────────────────────────────────────────────

print("=" * 60)
print("  Sector ETF Momentum — Cost Sensitivity Analysis")
print("=" * 60)
print(f"\n  Cost model : total_cost = commission + slippage (split evenly)")
print(f"  Formal baseline : {FORMAL_COST_BPS} bps total per side")

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
print(f"        Universe: {', '.join(RISK_ON_UNIVERSE)}\n")

# ── 2. Build shared signals + weights (identical across all cost runs) ─────────

strategy = SectorMomentumStrategy(
    momentum_window=BASE_PARAMS["momentum_window"],
    top_n=BASE_PARAMS["top_n"],
)
raw_signals, ranks, mom_scores = strategy.generate_signals_full(close)
signals  = apply_hysteresis(
    raw_signals, ranks, mom_scores,
    mode=HysteresisMode.ENTRY_MARGIN,
    top_n=BASE_PARAMS["top_n"],
    entry_margin=BASE_PARAMS["entry_margin"],
)
strength = compute_strength(
    signals, close,
    scheme=WeightingScheme.BLEND_70_30,
    vol_window=BASE_PARAMS["vol_window"],
)
sf      = SignalFrame(signals=signals, strength=strength,
                      strategy_name=strategy.name)
weights = signal_frame_to_weights(sf)

# ── 3. Run across cost variants ────────────────────────────────────────────────

results = {}   # total_bps → BacktestResult

for total_bps in COST_VARIANTS:
    half = total_bps / 2 / 10_000          # each of commission / slippage
    label = f"{total_bps}bps"
    marker = "  ← formal baseline" if total_bps == FORMAL_COST_BPS else ""
    print(f"Running {label} (commission={total_bps//2}bps, slippage={total_bps//2}bps){marker}...",
          end=" ", flush=True)

    vbt_cfg = VbtRunConfig(
        commission=half,
        slippage=half,
        rebalance_freq=BASE_PARAMS["rebalance_freq"],
        risk_free_rate=0.05,
    )
    result = run_vbt_backtest(
        close=close,
        weights=weights,
        config=vbt_cfg,
        benchmark_close=bm_close,
        strategy_name=f"{strategy.name}_{label}",
    )
    results[total_bps] = result
    print(f"CAGR={result.cagr:.2%}  Sharpe={result.sharpe_ratio:.3f}  "
          f"MaxDD={result.max_drawdown:.2%}  Paid=${result.commission_paid:,.0f}")

# ── 4. Summary table ───────────────────────────────────────────────────────────

baseline = results[FORMAL_COST_BPS]

print("\n" + "=" * 102)
print(
    f"{'Cost':>6} {'TotalRet':>9} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
    f"{'MaxDD':>7} {'Turnover':>9} {'Trades':>7} {'CommPaid':>10} {'ExcessSPY':>10}"
)
print("-" * 102)

for total_bps in COST_VARIANTS:
    r           = results[total_bps]
    excess_str  = f"{r.excess_return:+.2%}" if r.excess_return is not None else "   N/A"
    sortino_str = f"{r.sortino_ratio:.3f}" if r.sortino_ratio else "   N/A"
    turnover    = f"{r.annual_turnover:.1%}" if r.annual_turnover else "  N/A"
    marker      = " ←" if total_bps == FORMAL_COST_BPS else "  "
    print(
        f"{total_bps:>4}bps{marker}"
        f" {r.total_return:>9.2%} {r.cagr:>7.2%} "
        f"{r.sharpe_ratio:>7.3f} {sortino_str:>8} "
        f"{r.max_drawdown:>7.2%} {turnover:>9} "
        f"{r.n_trades:>7,} ${r.commission_paid:>9,.0f} {excess_str:>10}"
    )

print("=" * 102)

# ── 5. Cost-impact analysis ────────────────────────────────────────────────────

r0  = results[0]
r10 = results[10]
r20 = results[20]
r30 = results[30]

cagr_decay_per_10bps = (r0.cagr - r30.cagr) / 3
sharpe_decay         = r0.sharpe_ratio - r30.sharpe_ratio
sharpe_at_30_vs_0    = sharpe_decay / r0.sharpe_ratio   # fractional drop

print("\n[COST IMPACT]")
print(f"  CAGR 0->30 bps    : {r0.cagr:.2%} -> {r30.cagr:.2%}"
      f"  (-{r0.cagr - r30.cagr:.2%} total, ~-{cagr_decay_per_10bps:.2%} per 10bps)")
print(f"  Sharpe 0->30 bps  : {r0.sharpe_ratio:.3f} -> {r30.sharpe_ratio:.3f}"
      f"  (-{sharpe_decay:.3f}, {sharpe_at_30_vs_0:.1%} drop)")
print(f"  MaxDD change      : {r0.max_drawdown:.2%} -> {r30.max_drawdown:.2%}"
      f"  (delta {r30.max_drawdown - r0.max_drawdown:+.2%})")

# ── 6. Robustness verdict ──────────────────────────────────────────────────────

print("\n[VERDICT]")

# Criterion 1: Sharpe at 30bps still reasonable (>0.5 is typical threshold)
sharpe_30_ok = r30.sharpe_ratio >= 0.5
# Criterion 2: CAGR at 30bps still positive
cagr_30_ok   = r30.cagr > 0.0
# Criterion 3: Sharpe decay per 10bps < 10% of baseline Sharpe
decay_mild   = (sharpe_decay / 3) / r0.sharpe_ratio < 0.10

still_attractive = sharpe_30_ok and cagr_30_ok
cost_fragile     = not decay_mild or sharpe_decay / r0.sharpe_ratio > 0.25

print(f"  At 20 bps (formal): Sharpe={r20.sharpe_ratio:.3f}  CAGR={r20.cagr:.2%}  "
      + ("→ attractive" if r20.sharpe_ratio >= 0.7 and r20.cagr > 0.05 else "→ borderline"))
print(f"  At 30 bps (stress): Sharpe={r30.sharpe_ratio:.3f}  CAGR={r30.cagr:.2%}  "
      + ("→ still viable" if sharpe_30_ok and cagr_30_ok else "→ marginal/unviable"))
print(f"  Cost-fragile?       {'Yes — Sharpe decays quickly with cost' if cost_fragile else 'No — gradual, stable degradation'}")
print(f"  Worth continuing?   {'Yes' if still_attractive else 'Uncertain — review at 30bps'}")

# ── 7. Save ExperimentRecords ──────────────────────────────────────────────────

tracker = ExperimentTracker("./experiments")

last_date = signals.dropna(how="all").index[-1]
last_row  = signals.loc[last_date]
top_etfs  = last_row[last_row == 1.0].index.tolist()

for total_bps in COST_VARIANTS:
    result = results[total_bps]
    label  = f"{total_bps}bps"
    half   = total_bps / 2

    pw = PortfolioWeights(
        weights={sym: round(1.0 / BASE_PARAMS["top_n"], 4) for sym in top_etfs},
        method="blend_70_30",
        rebalance_date=last_date.date(),
    )

    bm_str      = f"{result.benchmark_return:.2%}" if result.benchmark_return is not None else "N/A"
    excess_sign = "+" if (result.excess_return or 0) >= 0 else ""
    ex_str      = (
        f" Excess vs SPY: {excess_sign}{result.excess_return:.2%}."
        if result.excess_return is not None else ""
    )
    is_baseline = total_bps == FORMAL_COST_BPS

    record = ExperimentRecord(
        description=(
            f"Cost sensitivity [{label}]: sector ETF rotation "
            f"top-{BASE_PARAMS['top_n']} by {BASE_PARAMS['momentum_window']}-day momentum, "
            f"bi-monthly, entry_margin hysteresis, blend_70_30 weighting. "
            f"Total cost={total_bps}bps/side (commission={half:.0f}bps + slippage={half:.0f}bps). "
            f"Universe: {', '.join(RISK_ON_UNIVERSE)}."
        ),
        strategy_params={
            **BASE_PARAMS,
            "total_cost_bps":   total_bps,
            "commission_bps":   half,
            "slippage_bps":     half,
            "is_formal_cost":   is_baseline,
        },
        symbols=RISK_ON_UNIVERSE,
        period_start=result.period_start,
        period_end=result.period_end,
        backtest_result=result,
        portfolio_weights=pw,
        tags=(
            ["sector-rotation", "momentum", "etf", "bimonthly",
             "cost-study", label, f"top{BASE_PARAMS['top_n']}", "blend-70-30", "entry-margin"]
            + (["baseline"] if is_baseline else [])
        ),
        notes=(
            f"Benchmark SPY: {bm_str}.{ex_str} "
            f"Cost model: total_cost={total_bps}bps per side, split 50/50 commission/slippage."
            + (" [FORMAL BASELINE COST]" if is_baseline else "")
        ),
    )

    exp_dir = tracker.save(record)
    print(f"[SAVED]  {label:<8}  {exp_dir.name}")

# ── 8. Registry ────────────────────────────────────────────────────────────────

print("\n[REGISTRY] cost-study experiments (newest first):")
for entry in tracker.list_experiments(tag="cost-study", limit=10):
    m = entry["metrics"]
    print(
        f"  {entry['created_at'][:19]}  {entry['strategy_name']:<48}"
        f"  CAGR={m.get('cagr', float('nan')):.2%}"
        f"  Sharpe={m.get('sharpe_ratio', float('nan')):.3f}"
        f"  Paid=${m.get('commission_paid', 0):,.0f}"
    )
