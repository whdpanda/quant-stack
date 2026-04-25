"""Sector ETF Momentum — Portfolio-Level Validation.

Purpose
-------
Evaluate the formal strategy as a *component* inside a larger portfolio by
blending it with a SPY buy-and-hold allocation at four ratios.  The key
questions are:

1. Does adding the strategy improve Sharpe vs pure SPY?
2. Does it reduce Max Drawdown?
3. Is CAGR still acceptable at each blend ratio?
4. Is the strategy better as a "stable component" than as a standalone?

Blend ratios (SPY % / Strategy %)
----------------------------------
  100 /   0  :  pure SPY buy-and-hold (baseline)
   70 /  30  :  light allocation
   50 /  50  :  balanced
    0 / 100  :  pure strategy

Blend mechanics
---------------
Daily portfolio return = alpha * r_spy + (1 - alpha) * r_strategy
This is a daily-rebalanced synthetic blend — no friction modelled for
the SPY side; strategy returns already embed 20 bps total cost.

Run
---
    python experiments/portfolio_validation.py

Requires
--------
    pip install 'quant-stack[research]'   pip install yfinance
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from quant_stack.core.schemas import (
    BacktestResult,
    DataConfig,
    ExperimentRecord,
    PortfolioWeights,
)
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
    get_portfolio_daily_returns,
    signal_frame_to_weights,
)
from quant_stack.signals.base import SignalFrame
from quant_stack.tracking import ExperimentTracker

# ── Config ─────────────────────────────────────────────────────────────────────

BENCHMARK     = "SPY"
PERIOD_START  = date(2010, 1, 1)
PERIOD_END    = date(2025, 12, 31)
RF_ANNUAL     = 0.05
ANN           = 252

STRATEGY_PARAMS = {
    "momentum_window":  210,
    "top_n":            3,
    "rebalance_freq":   "2ME",
    "total_cost_bps":   20,
    "hysteresis_mode":  "entry_margin",
    "entry_margin":     0.02,
    "weighting_scheme": "blend_70_30",
    "vol_window":       63,
}

BLEND_CONFIGS: list[tuple[float, float, str]] = [
    (1.00, 0.00, "100pct_spy"),
    (0.70, 0.30, "70spy_30strat"),
    (0.50, 0.50, "50spy_50strat"),
    (0.00, 1.00, "100pct_strategy"),
]

# ── Metrics helper ─────────────────────────────────────────────────────────────

def _metrics(r: pd.Series, rf_annual: float = RF_ANNUAL, ann: int = ANN) -> dict:
    """Compute standard performance metrics from a daily return series."""
    r = r.dropna()
    if len(r) == 0:
        return {}

    total_return = float((1 + r).prod() - 1)
    n_years = len(r) / ann
    cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol = float(r.std() * math.sqrt(ann))
    rf_daily = (1 + rf_annual) ** (1 / ann) - 1
    excess = r - rf_daily
    sharpe = float(excess.mean() / excess.std() * math.sqrt(ann)) if excess.std() > 0 else float("nan")

    downside = r[r < rf_daily]
    sortino_denom = float(downside.std() * math.sqrt(ann)) if len(downside) > 0 else 0.0
    sortino = float(excess.mean() * ann / sortino_denom) if sortino_denom > 0 else float("nan")

    # Max drawdown
    wealth = (1 + r).cumprod()
    rolling_max = wealth.cummax()
    dd = (wealth - rolling_max) / rolling_max
    max_dd = float(dd.min())

    calmar = abs(cagr / max_dd) if max_dd != 0 else float("nan")

    # Worst calendar year
    annual_rets = r.resample("YE").apply(lambda x: float((1 + x).prod() - 1))
    worst_year = float(annual_rets.min()) if len(annual_rets) > 0 else float("nan")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "worst_year": worst_year,
        "n_years": n_years,
    }


# ── 1. Fetch data ──────────────────────────────────────────────────────────────

print("=" * 70)
print("  Sector ETF Momentum — Portfolio-Level Validation")
print("=" * 70)

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

# ── 2. Build strategy signals and weights ─────────────────────────────────────

strategy = SectorMomentumStrategy(
    momentum_window=STRATEGY_PARAMS["momentum_window"],
    top_n=STRATEGY_PARAMS["top_n"],
)
raw_signals, ranks, mom_scores = strategy.generate_signals_full(close)
signals  = apply_hysteresis(
    raw_signals, ranks, mom_scores,
    mode=HysteresisMode.ENTRY_MARGIN,
    top_n=STRATEGY_PARAMS["top_n"],
    entry_margin=STRATEGY_PARAMS["entry_margin"],
)
strength = compute_strength(
    signals, close,
    scheme=WeightingScheme.BLEND_70_30,
    vol_window=STRATEGY_PARAMS["vol_window"],
)
sf      = SignalFrame(signals=signals, strength=strength,
                      strategy_name=strategy.name)
weights = signal_frame_to_weights(sf)

# ── 3. Get daily return series for strategy and SPY ───────────────────────────

half_bps = STRATEGY_PARAMS["total_cost_bps"] / 2 / 10_000
vbt_cfg = VbtRunConfig(
    commission=half_bps,
    slippage=half_bps,
    rebalance_freq=STRATEGY_PARAMS["rebalance_freq"],
    risk_free_rate=RF_ANNUAL,
)

logger.info("Running strategy portfolio for daily returns...")
r_strategy = get_portfolio_daily_returns(close, weights, config=vbt_cfg)

# SPY daily returns (buy-and-hold, no costs)
r_spy = bm_close.sort_index().pct_change().rename("spy")

# Align to a common index (strategy warmup leaves leading NaN)
common_idx = r_strategy.dropna().index.intersection(r_spy.dropna().index)
r_strategy_aligned = r_strategy.loc[common_idx]
r_spy_aligned      = r_spy.loc[common_idx]

print(f"\n[RETURNS]  Aligned series: {len(common_idx):,} days  "
      f"{common_idx[0].date()} to {common_idx[-1].date()}")

# ── 4. Compute blended metrics ────────────────────────────────────────────────

blend_metrics: dict[str, dict] = {}
blend_returns: dict[str, pd.Series] = {}

for alpha_spy, alpha_strat, label in BLEND_CONFIGS:
    r_blend = alpha_spy * r_spy_aligned + alpha_strat * r_strategy_aligned
    m = _metrics(r_blend)
    blend_metrics[label] = m
    blend_returns[label] = r_blend

print(f"\n[BLEND METRICS]  (rf={RF_ANNUAL:.0%}, ann={ANN}d)")

# ── 5. Summary table ───────────────────────────────────────────────────────────

SPY_LABEL = "100pct_spy"
m_spy = blend_metrics[SPY_LABEL]

header = (
    f"{'Blend':>22}  {'TotalRet':>9} {'CAGR':>7} {'Vol':>6} "
    f"{'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} "
    f"{'WorstYr':>8}  {'dSharpe':>8} {'dMaxDD':>8}"
)
print("\n" + "=" * len(header))
print(header)
print("-" * len(header))

for alpha_spy, alpha_strat, label in BLEND_CONFIGS:
    m    = blend_metrics[label]
    d_sharpe = m["sharpe"] - m_spy["sharpe"]
    d_maxdd  = m["max_drawdown"] - m_spy["max_drawdown"]
    marker   = " <-- baseline" if label == SPY_LABEL else ""
    sortino_str = f"{m['sortino']:>8.3f}" if not math.isnan(m.get("sortino", float("nan"))) else "     N/A"
    calmar_str  = f"{m['calmar']:>7.3f}" if not math.isnan(m.get("calmar", float("nan"))) else "    N/A"
    blend_label = f"{int(alpha_spy*100)}% SPY + {int(alpha_strat*100)}% Strat"
    print(
        f"{blend_label:>22}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['annual_vol']:>6.2%} "
        f"{m['sharpe']:>7.3f} {sortino_str} {m['max_drawdown']:>7.2%} "
        f"{calmar_str} {m['worst_year']:>8.2%}  "
        f"{d_sharpe:>+8.3f} {d_maxdd:>+8.2%}"
        + marker
    )

print("=" * len(header))

# ── 6. Verdict ────────────────────────────────────────────────────────────────

print("\n[VERDICT]")

m_7030 = blend_metrics["70spy_30strat"]
m_5050 = blend_metrics["50spy_50strat"]
m_strat = blend_metrics["100pct_strategy"]

sharpe_improves  = m_7030["sharpe"] > m_spy["sharpe"] + 0.02
dd_reduces       = m_7030["max_drawdown"] > m_spy["max_drawdown"]   # max_dd is negative
cagr_acceptable  = m_7030["cagr"] > 0.06
strat_standalone = m_strat["sharpe"] > m_spy["sharpe"]

print(f"  Q1 — Does blending (70/30) improve Sharpe vs pure SPY?")
print(f"       SPY={m_spy['sharpe']:.3f}  70/30={m_7030['sharpe']:.3f}  "
      f"delta={m_7030['sharpe']-m_spy['sharpe']:+.3f}  "
      + ("-> YES" if sharpe_improves else "-> marginal / NO"))

print(f"  Q2 — Does blending (70/30) reduce Max Drawdown?")
print(f"       SPY={m_spy['max_drawdown']:.2%}  70/30={m_7030['max_drawdown']:.2%}  "
      f"delta={m_7030['max_drawdown']-m_spy['max_drawdown']:+.2%}  "
      + ("-> YES" if dd_reduces else "-> NO"))

print(f"  Q3 — Is CAGR still acceptable at 70/30?")
print(f"       70/30 CAGR={m_7030['cagr']:.2%}  "
      + ("-> YES (>6%)" if cagr_acceptable else "-> borderline"))

print(f"  Q4 — Is the strategy better as a component than standalone?")
print(f"       Best blend Sharpe={max(m['sharpe'] for m in blend_metrics.values()):.3f}  "
      f"Standalone={m_strat['sharpe']:.3f}  "
      + ("-> use as component" if not strat_standalone else "-> also strong standalone"))

# Best blend by Sharpe
best_label = max(
    [(lbl, blend_metrics[lbl]["sharpe"]) for _, _, lbl in BLEND_CONFIGS],
    key=lambda x: x[1]
)[0]
best_ratio = next((f"{int(a*100)}% SPY + {int(b*100)}% Strat" for a,b,l in BLEND_CONFIGS if l == best_label), best_label)
print(f"\n  Best blend by Sharpe: {best_ratio} (Sharpe={blend_metrics[best_label]['sharpe']:.3f})")

# ── 7. Save ExperimentRecords ──────────────────────────────────────────────────

tracker = ExperimentTracker("./experiments")

last_date = signals.dropna(how="all").index[-1]
last_row  = signals.loc[last_date]
top_etfs  = last_row[last_row == 1.0].index.tolist()

for alpha_spy, alpha_strat, label in BLEND_CONFIGS:
    m = blend_metrics[label]
    blend_label = f"{int(alpha_spy*100)}pct_spy_{int(alpha_strat*100)}pct_strat"

    # Reconstruct a BacktestResult-compatible object from blend metrics
    period_start = common_idx[0].date()
    period_end   = common_idx[-1].date()

    blend_result = BacktestResult(
        strategy_name=f"portfolio_blend_{label}",
        symbols=[BENCHMARK] + RISK_ON_UNIVERSE if alpha_strat > 0 else [BENCHMARK],
        period_start=period_start,
        period_end=period_end,
        total_return=m["total_return"],
        cagr=m["cagr"],
        sharpe_ratio=m["sharpe"],
        sortino_ratio=m.get("sortino") if not math.isnan(m.get("sortino", float("nan"))) else None,
        max_drawdown=m["max_drawdown"],
        annual_volatility=m["annual_vol"],
        annual_turnover=None,
        n_trades=0,
        commission_paid=0.0,
        benchmark_return=m_spy["total_return"],
        metadata={
            "alpha_spy": alpha_spy,
            "alpha_strategy": alpha_strat,
            "calmar": m.get("calmar"),
            "worst_year": m.get("worst_year"),
            "delta_sharpe_vs_spy": m["sharpe"] - m_spy["sharpe"],
            "delta_maxdd_vs_spy": m["max_drawdown"] - m_spy["max_drawdown"],
        },
    )

    pw = PortfolioWeights(
        weights={sym: round(alpha_strat / max(len(top_etfs), 1), 4) for sym in top_etfs}
               | {BENCHMARK: round(alpha_spy, 4)}
        if alpha_strat > 0 and alpha_spy > 0
        else (
            {sym: round(1.0 / max(len(top_etfs), 1), 4) for sym in top_etfs}
            if alpha_strat > 0
            else {BENCHMARK: 1.0}
        ),
        method=f"blend_{int(alpha_spy*100)}_{int(alpha_strat*100)}",
        rebalance_date=last_date.date(),
    )

    record = ExperimentRecord(
        description=(
            f"Portfolio blend {int(alpha_spy*100)}% SPY / {int(alpha_strat*100)}% Strategy: "
            f"daily-rebalanced synthetic blend of SPY B&H and formal sector-momentum strategy "
            f"(210d momentum, top-3, blend_70_30 weighting, entry_margin hysteresis, 20bps cost). "
            f"Universe: {', '.join(RISK_ON_UNIVERSE)}."
        ),
        strategy_params={
            **STRATEGY_PARAMS,
            "alpha_spy": alpha_spy,
            "alpha_strategy": alpha_strat,
            "blend_label": label,
            "is_spy_baseline": alpha_spy == 1.0,
        },
        symbols=([BENCHMARK] + RISK_ON_UNIVERSE) if alpha_strat > 0 else [BENCHMARK],
        period_start=period_start,
        period_end=period_end,
        backtest_result=blend_result,
        portfolio_weights=pw,
        tags=(
            ["portfolio-validation", "blend", f"spy{int(alpha_spy*100)}", f"strat{int(alpha_strat*100)}",
             "sector-rotation", "momentum", "etf"]
            + (["spy-baseline"] if alpha_spy == 1.0 else [])
            + (["pure-strategy"] if alpha_spy == 0.0 else [])
        ),
        notes=(
            f"Blend: {int(alpha_spy*100)}% SPY + {int(alpha_strat*100)}% strategy. "
            f"CAGR={m['cagr']:.2%}  Sharpe={m['sharpe']:.3f}  MaxDD={m['max_drawdown']:.2%}  "
            f"WorstYr={m['worst_year']:.2%}. "
            f"dSharpe vs SPY={m['sharpe']-m_spy['sharpe']:+.3f}  "
            f"dMaxDD vs SPY={m['max_drawdown']-m_spy['max_drawdown']:+.2%}."
        ),
    )

    exp_dir = tracker.save(record)
    print(f"\n[SAVED]  {label:<28}  {exp_dir.name}")

# ── 8. Registry ────────────────────────────────────────────────────────────────

print("\n[REGISTRY] portfolio-validation experiments (newest first):")
for entry in tracker.list_experiments(tag="portfolio-validation", limit=8):
    m = entry["metrics"]
    print(
        f"  {entry['created_at'][:19]}  {entry['strategy_name']:<44}"
        f"  CAGR={m.get('cagr', float('nan')):.2%}"
        f"  Sharpe={m.get('sharpe_ratio', float('nan')):.3f}"
        f"  MaxDD={m.get('max_drawdown', float('nan')):.2%}"
    )
