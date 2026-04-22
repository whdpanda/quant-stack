"""Demo: data → factors → signals → portfolio weights → vbt backtest.

Pipeline
--------
1. Generate synthetic price data (no network dependency)
2. Compute SMA-200 trend filter + 63-day momentum factor
3. Build SignalFrame: trend-eligible, top-2 momentum ranking
4. Convert to portfolio weights via signal_frame_to_weights
5. Run monthly-rebalancing backtest via run_vbt_backtest
6. Compare against SPY buy-and-hold benchmark
7. Print standardised BacktestResult

Run:
    python experiments/demo_vbt_adapter.py

Requires:
    pip install 'quant-stack[research]'
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_stack.core.schemas import SignalSource
from quant_stack.factors.momentum import momentum_63
from quant_stack.factors.trend import sma_200
from quant_stack.research.vbt_adapter import VbtRunConfig, run_vbt_backtest, signal_frame_to_weights
from quant_stack.signals.momentum import relative_momentum_ranking_signal
from quant_stack.signals.trend import as_eligibility_mask, trend_filter_signal

# ── 1. Synthetic universe ──────────────────────────────────────────────────────
# 4 equity-like assets + 1 bond proxy, 1500 business days (~6 years)

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "IEF"]
N_BARS = 1_500

rng = np.random.default_rng(42)
idx = pd.bdate_range("2018-01-02", periods=N_BARS)

# Simple correlated random-walk prices
# Equity assets trend up ~8%/yr; bond proxy trends up ~3%/yr
annual_returns = {"SPY": 0.08, "QQQ": 0.12, "IWM": 0.07, "EFA": 0.05, "IEF": 0.03}
annual_vols    = {"SPY": 0.16, "QQQ": 0.20, "IWM": 0.18, "EFA": 0.18, "IEF": 0.06}

prices = {}
for sym in SYMBOLS:
    mu  = annual_returns[sym] / 252
    sig = annual_vols[sym]    / np.sqrt(252)
    log_ret = rng.normal(mu - 0.5 * sig**2, sig, N_BARS)
    prices[sym] = 100.0 * np.exp(np.cumsum(log_ret))

close = pd.DataFrame(prices, index=idx)

print("=== Synthetic universe ===")
print(close.tail(3).to_string())

# ── 2. Factors ─────────────────────────────────────────────────────────────────

mom   = momentum_63(close)
trend = sma_200(close)

# ── 3. Signals ────────────────────────────────────────────────────────────────

trend_sf  = trend_filter_signal(close, trend)
eligible  = as_eligibility_mask(trend_sf)
signal_sf = relative_momentum_ranking_signal(mom, top_n=2, eligible=eligible)

# Inspect latest signals
print("\n=== Latest signals (last 3 dates) ===")
print(signal_sf.signals.tail(3).to_string())
print(f"Source: {signal_sf.source.value}  (must be RESEARCH — never direct-feed to execution)")

# ── 4. Convert to weights ──────────────────────────────────────────────────────

weights = signal_frame_to_weights(signal_sf)
print("\n=== Weight preview (last 3 dates) ===")
print(weights.tail(3).round(3).to_string())

# ── 5. Backtest ───────────────────────────────────────────────────────────────

cfg = VbtRunConfig(
    initial_cash   = 100_000.0,
    commission     = 0.001,        # 10 bps round-trip
    slippage       = 0.001,
    rebalance_freq = "ME",         # rebalance at month-end only
    risk_free_rate = 0.05,
)

benchmark = close["SPY"]          # compare vs. buy-and-hold SPY

result = run_vbt_backtest(
    close          = close,
    weights        = weights,
    config         = cfg,
    benchmark_close= benchmark,
    strategy_name  = "trend_momentum_top2",
)

# ── 6. Print results ──────────────────────────────────────────────────────────

print("\n" + "=" * 52)
print(f"  Strategy : {result.strategy_name}")
print(f"  Period   : {result.period_start} → {result.period_end}")
print(f"  Symbols  : {result.symbols}")
print("=" * 52)
print(f"  Total Return   : {result.total_return:>9.2%}")
print(f"  CAGR           : {result.cagr:>9.2%}")
print(f"  Volatility     : {(result.annual_volatility or 0):>9.2%}  (annual)")
print(f"  Sharpe Ratio   : {result.sharpe_ratio:>9.3f}")
print(f"  Sortino Ratio  : {(result.sortino_ratio or float('nan')):>9.3f}")
print(f"  Max Drawdown   : {result.max_drawdown:>9.2%}")
print(f"  Annual Turnover: {(result.annual_turnover or 0):>9.1%}")
print(f"  Trades         : {result.n_trades:>9d}")
print(f"  Commission Paid: ${result.commission_paid:>8.0f}")
print("-" * 52)
if result.benchmark_return is not None:
    print(f"  Benchmark (SPY): {result.benchmark_return:>9.2%}  total return")
    excess = result.excess_return
    print(f"  Excess Return  : {(excess or 0):>9.2%}")
print("=" * 52)

# ── 7. Demonstrate look-ahead prevention ─────────────────────────────────────

print("\n=== Anti look-ahead check ===")
import pandas.tseries.offsets as off

orders_sample = weights.dropna(how="all")
me_dates = close.resample("ME").last().index
print(f"  Weight rows (pre-shift)  : month-end dates, e.g. {me_dates[5].date()}")
# The adapter shifts by 1 BDay — confirm first execution date is not a ME date
from quant_stack.research.vbt_adapter import _prepare_orders
orders = _prepare_orders(close, weights, "ME")
first_exec = orders.dropna(how="all").index[0]
print(f"  First execution date     : {first_exec.date()}")
is_me = first_exec in me_dates
print(f"  Is month-end (would be look-ahead): {is_me}  <- must be False")
assert not is_me, "BUG: execution falls on decision date (look-ahead bias)"
print("  [OK] Execution correctly shifted to next business day")
