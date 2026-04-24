"""Sector ETF Momentum - IEF Defensive Fallback Comparison.

Runs two strategies side-by-side and saves both ExperimentRecords:

  baseline   : Sector momentum top-N on risk-on universe, always fully
               invested (no absolute momentum filter, no IEF).

  ief_fallback: Same risk-on universe + absolute momentum filter.
               When fewer than N ETFs have positive momentum, remaining
               slots are filled by IEF.  100% IEF when all fail the filter.

Run
---
    python experiments/sector_momentum_ief_experiment.py

Requires
--------
    pip install 'quant-stack[research]'
    pip install yfinance
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger

from quant_stack.core.schemas import DataConfig, ExperimentRecord, PortfolioWeights
from quant_stack.data.providers.yahoo import YahooProvider
from quant_stack.research.strategies.sector_momentum import SectorMomentumStrategy
from quant_stack.research.strategies.sector_momentum_ief import (
    SectorMomentumIefFallbackStrategy,
)
from quant_stack.research.vbt_adapter import (
    VbtRunConfig,
    run_vbt_backtest,
    signal_frame_to_weights,
)
from quant_stack.signals.base import SignalFrame
from quant_stack.tracking import ExperimentTracker, ReportGenerator

# ── Shared config ──────────────────────────────────────────────────────────────

# IEF is the fallback asset — it participates in the IEF-fallback strategy
# but is NOT part of the risk-on ranking universe.
RISK_ON_UNIVERSE = ["VNQ", "QQQ", "XLE", "XLV", "XLF", "XLI", "XLB", "SPY", "XLP"]
FALLBACK_ASSET   = "IEF"
BENCHMARK        = "SPY"

PERIOD_START = date(2010, 1, 1)
PERIOD_END   = date(2025, 12, 31)

SHARED_PARAMS = {
    "momentum_window":        210,   # ~10 calendar months
    "top_n":                  3,
    "rebalance_freq":         "ME",
    "commission_bps":         10,
}
ABS_THRESHOLD = 0.0   # ETF must have momentum > 0 to qualify


# ── Utilities ──────────────────────────────────────────────────────────────────

def _fmt_pct(v) -> str:
    return f"{v:.2%}" if v is not None else "N/A"

def _fmt_f(v) -> str:
    return f"{v:.3f}" if v is not None else "N/A"

def _fmt_n(v) -> str:
    return f"{int(v):,}" if v is not None else "N/A"

def _fmt_usd(v) -> str:
    return f"${v:,.0f}" if v else "$0"

def _row(label, a, b, fmt=_fmt_pct, w=14) -> None:
    print(f"  {label:<26} {fmt(a):>{w}}   {fmt(b):>{w}}")


# ── 1. Fetch data ──────────────────────────────────────────────────────────────

print("=" * 66)
print("  Sector ETF Momentum — IEF Fallback Comparison")
print("=" * 66)

all_syms = list(dict.fromkeys(RISK_ON_UNIVERSE + [FALLBACK_ASSET, BENCHMARK]))
cfg = DataConfig(
    symbols=all_syms,
    start=PERIOD_START,
    end=PERIOD_END,
    cache_dir="./data",
)

logger.info(f"Fetching {len(all_syms)} symbols...")
raw = YahooProvider().fetch(cfg)

if isinstance(raw.columns, pd.MultiIndex):
    close_all = raw.xs("close", axis=1, level=1)
else:
    close_all = raw[["close"]].rename(columns={"close": all_syms[0]})

close_all = close_all.sort_index().dropna(how="all")

close_ro   = close_all[RISK_ON_UNIVERSE].copy()              # risk-on only
close_full = close_all[RISK_ON_UNIVERSE + [FALLBACK_ASSET]].copy()  # all assets
bm_close   = close_all[BENCHMARK].copy()

print(f"\n[DATA]  {close_ro.shape[0]:,} trading days")
print(f"        {close_ro.index[0].date()} to {close_ro.index[-1].date()}")
print(f"        Risk-on universe : {', '.join(RISK_ON_UNIVERSE)}")
print(f"        Fallback asset   : {FALLBACK_ASSET}")
print(f"        Benchmark        : {BENCHMARK}")

vbt_cfg = VbtRunConfig(
    commission=SHARED_PARAMS["commission_bps"] / 10_000,
    rebalance_freq=SHARED_PARAMS["rebalance_freq"],
    risk_free_rate=0.05,
)


# ── 2. Baseline ────────────────────────────────────────────────────────────────

print(f"\n[RUN 1/2] Baseline — no filter, always fully invested in risk-on assets...")
strat_base = SectorMomentumStrategy(
    momentum_window=SHARED_PARAMS["momentum_window"],
    top_n=SHARED_PARAMS["top_n"],
)
sigs_base = strat_base.generate_signals(close_ro)
sf_base   = SignalFrame(signals=sigs_base, strength=sigs_base.copy(),
                        strategy_name=strat_base.name)
w_base    = signal_frame_to_weights(sf_base)
result_base = run_vbt_backtest(
    close=close_ro, weights=w_base, config=vbt_cfg,
    benchmark_close=bm_close, strategy_name=strat_base.name,
)


# ── 3. IEF fallback ────────────────────────────────────────────────────────────

print(f"[RUN 2/2] IEF fallback — abs filter threshold={ABS_THRESHOLD}, "
      f"unused slots → {FALLBACK_ASSET}...")
strat_ief = SectorMomentumIefFallbackStrategy(
    momentum_window=SHARED_PARAMS["momentum_window"],
    top_n=SHARED_PARAMS["top_n"],
    abs_momentum_threshold=ABS_THRESHOLD,
    fallback_asset=FALLBACK_ASSET,
)
w_ief = strat_ief.compute_weights(close_full)
result_ief = run_vbt_backtest(
    close=close_full, weights=w_ief, config=vbt_cfg,
    benchmark_close=bm_close, strategy_name=strat_ief.name,
)


# ── 4. Comparison table ───────────────────────────────────────────────────────

W = 14
print(f"\n{'─' * 66}")
print(f"  {'Metric':<26} {'Baseline':>{W}}   {'IEF Fallback':>{W}}")
print(f"{'─' * 66}")
_row("Total Return",         result_base.total_return,      result_ief.total_return)
_row("CAGR",                 result_base.cagr,              result_ief.cagr)
_row("Annual Volatility",    result_base.annual_volatility, result_ief.annual_volatility)
_row("Sharpe Ratio",         result_base.sharpe_ratio,      result_ief.sharpe_ratio,    fmt=_fmt_f)
_row("Sortino Ratio",        result_base.sortino_ratio,     result_ief.sortino_ratio,   fmt=_fmt_f)
_row("Max Drawdown",         result_base.max_drawdown,      result_ief.max_drawdown)
_row("Annual Turnover",      result_base.annual_turnover,   result_ief.annual_turnover)
_row("Trades",               result_base.n_trades,          result_ief.n_trades,        fmt=_fmt_n)
_row("Commission Paid",      result_base.commission_paid,   result_ief.commission_paid, fmt=_fmt_usd)
print(f"{'─' * 66}")
_row("SPY Total Return",     result_base.benchmark_return,  result_ief.benchmark_return)
_row("Excess vs SPY",        result_base.excess_return,     result_ief.excess_return)
print(f"{'─' * 66}")


# ── 5. IEF fallback activation statistics ─────────────────────────────────────

active_ief = w_ief.dropna(how="all")
n_ro_held  = (active_ief[RISK_ON_UNIVERSE] > 0).sum(axis=1)
ief_held   = active_ief[FALLBACK_ASSET]
top_n      = SHARED_PARAMS["top_n"]

print(f"\n[IEF FALLBACK STATS]  (active trading days = {len(active_ief):,})")
print(f"  Abs-momentum threshold: {ABS_THRESHOLD}")
for k in range(top_n + 1):
    n_days = int((n_ro_held == k).sum())
    ief_frac = (top_n - k) / top_n
    bar = "#" * min(40, n_days * 40 // max(1, len(active_ief)))
    suffix = (
        f"(IEF = {ief_frac:.0%})" if k < top_n else "(IEF = 0%)"
    )
    label = (
        f"{k} risk-on ETF{'s' if k != 1 else '':1} selected {suffix}"
    )
    print(f"  {label:<42} {n_days:>5,}  {bar}")

pct_full = (n_ro_held == top_n).sum() / len(active_ief) * 100
pct_partial = ((n_ro_held > 0) & (n_ro_held < top_n)).sum() / len(active_ief) * 100
pct_all_ief = (n_ro_held == 0).sum() / len(active_ief) * 100
print(f"\n  Fully invested (no IEF)  : {pct_full:.1f}% of active days")
print(f"  Partial IEF fallback     : {pct_partial:.1f}% of active days")
print(f"  100% IEF (no risk-on)    : {pct_all_ief:.1f}% of active days")
print(f"  Avg IEF allocation       : {ief_held.mean():.1%}")
print(f"  Max IEF allocation       : {ief_held.max():.1%}")


# ── 6. Last rebalance holdings ─────────────────────────────────────────────────

def _last_holdings(weights: pd.DataFrame, risk_on_cols: list[str], fb: str):
    active = weights.dropna(how="all")
    last_date = active.index[-1]
    last_row = active.loc[last_date]
    ro_held = {c: last_row[c] for c in risk_on_cols if last_row.get(c, 0) > 0}
    ief_w = last_row.get(fb, 0.0)
    return last_date, ro_held, ief_w

def _last_sigs(sigs: pd.DataFrame):
    active = sigs.dropna(how="all")
    last_date = active.index[-1]
    held = active.loc[last_date]
    return last_date, held[held == 1.0].index.tolist()

date_base, etfs_base = _last_sigs(sigs_base)
date_ief, ro_ief, ief_w_last = _last_holdings(w_ief, RISK_ON_UNIVERSE, FALLBACK_ASSET)

print(f"\n[LAST REBALANCE]")
print(f"  Baseline  ({date_base.date()}): {', '.join(etfs_base) or 'none'}")
ief_str = f"  + {FALLBACK_ASSET} {ief_w_last:.0%}" if ief_w_last > 0 else ""
ro_str = ", ".join(f"{sym} {w:.0%}" for sym, w in sorted(ro_ief.items(), key=lambda x: -x[1]))
print(f"  IEF Fallback ({date_ief.date()}): {ro_str or 'none'}{ief_str}")


# ── 7. Build ExperimentRecords ────────────────────────────────────────────────

def _build_record(
    result,
    last_weights: dict[str, float],
    last_date,
    strategy_params: dict,
    tags: list[str],
    notes: str,
    symbols: list[str],
) -> ExperimentRecord:
    pw = PortfolioWeights(
        weights=last_weights,
        method="equal_weight",
        rebalance_date=last_date.date(),
    )
    bm_str = f"{result.benchmark_return:.2%}" if result.benchmark_return is not None else "N/A"
    ex_str = ""
    if result.excess_return is not None:
        sign = "+" if result.excess_return >= 0 else ""
        ex_str = f" Excess vs SPY: {sign}{result.excess_return:.2%}."
    return ExperimentRecord(
        description=(
            f"Sector ETF rotation: top-{strategy_params['top_n']} by "
            f"{strategy_params['momentum_window']}-day momentum, monthly rebalancing."
        ),
        strategy_params=strategy_params,
        symbols=symbols,
        period_start=result.period_start,
        period_end=result.period_end,
        backtest_result=result,
        portfolio_weights=pw,
        tags=["sector-rotation", "momentum", "etf", "monthly"] + tags,
        notes=f"Benchmark SPY: {bm_str}.{ex_str} " + notes,
    )


# Baseline record
params_base = {**SHARED_PARAMS, "abs_momentum_threshold": None, "fallback": None}
last_w_base = {sym: 1.0 / SHARED_PARAMS["top_n"] for sym in etfs_base}
record_base = _build_record(
    result_base, last_w_base, date_base,
    strategy_params=params_base,
    tags=["baseline", f"top{SHARED_PARAMS['top_n']}", "equal-weight"],
    notes="No absolute momentum filter. Always fully invested in risk-on universe.",
    symbols=RISK_ON_UNIVERSE,
)

# IEF fallback record
params_ief = {
    **SHARED_PARAMS,
    "abs_momentum_threshold": ABS_THRESHOLD,
    "fallback_asset": FALLBACK_ASSET,
}
record_ief = _build_record(
    result_ief, {**{sym: w for sym, w in ro_ief.items()}, FALLBACK_ASSET: ief_w_last}, date_ief,
    strategy_params=params_ief,
    tags=["ief-fallback", f"top{SHARED_PARAMS['top_n']}", "equal-weight"],
    notes=(
        f"Abs momentum filter threshold={ABS_THRESHOLD}. "
        f"Unused slots filled by {FALLBACK_ASSET}. "
        f"Fully invested {pct_full:.1f}% of days; "
        f"100% IEF {pct_all_ief:.1f}% of days."
    ),
    symbols=RISK_ON_UNIVERSE + [FALLBACK_ASSET],
)


# ── 8. Save records ───────────────────────────────────────────────────────────

tracker  = ExperimentTracker("./experiments")
dir_base = tracker.save(record_base)
dir_ief  = tracker.save(record_ief)

print(f"\n[SAVED]")
print(f"  baseline     : {dir_base}")
print(f"  ief_fallback : {dir_ief}")


# ── 9. Registry summary ───────────────────────────────────────────────────────

print(f"\n[REGISTRY] sector-rotation experiments (newest first, up to 10):")
for entry in tracker.list_experiments(tag="sector-rotation", limit=10):
    m    = entry["metrics"]
    cagr = m.get("cagr",         float("nan"))
    sh   = m.get("sharpe_ratio", float("nan"))
    dd   = m.get("max_drawdown", float("nan"))
    tags = [t for t in entry["tags"] if t not in ("sector-rotation", "etf", "monthly")]
    print(
        f"  {entry['created_at'][:19]}  {entry['strategy_name']:<46}"
        f"  CAGR={cagr:.2%}  Sharpe={sh:.3f}  MaxDD={dd:.2%}"
        f"  [{', '.join(tags)}]"
    )


# ── 10. Print both reports ────────────────────────────────────────────────────

for label, exp_dir in [("BASELINE", dir_base), ("IEF FALLBACK", dir_ief)]:
    rpt = exp_dir / "report.md"
    print(f"\n{'=' * 66}")
    print(f"  {label}: report.md")
    print(f"{'=' * 66}\n")
    try:
        print(rpt.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        print(rpt.read_text(encoding="utf-8", errors="replace"))
