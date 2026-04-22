"""Trend filter signal generator.

The trend filter is a binary regime indicator:
    close > SMA  → 1.0 (risk-on for this symbol)
    close ≤ SMA  → 0.0 (risk-off / flat)

Primary use cases
-----------------
1. Standalone signal: only hold a symbol when it is in an uptrend.
2. Eligibility filter for other signals: pass the result as `eligible`
   to relative_momentum_ranking_signal() to only rank symbols that are
   currently above their long-term trend.

Typical parameterizations for low-frequency ETF strategies:
    SMA-200 filter (Faber 2007): 200-day SMA on monthly bar
    SMA-50 filter: shorter-term trend, higher turnover

References
----------
Faber, M. (2007). A Quantitative Approach to Tactical Asset Allocation.
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461
"""

from __future__ import annotations

import pandas as pd

from quant_stack.signals.base import SignalFrame


def trend_filter_signal(
    close: pd.DataFrame,
    sma: pd.DataFrame,
    strategy_name: str = "trend_filter",
) -> SignalFrame:
    """Long (risk-on) when close > SMA, Flat (risk-off) otherwise.

    Args:
        close: Adjusted close prices (DatetimeIndex × symbols).
               Must have the same shape and columns as sma.
        sma: Moving average values from sma_50() or sma_200()
             (DatetimeIndex × symbols).
        strategy_name: Identifier for this signal set.

    Returns:
        SignalFrame with binary signals (0.0 or 1.0).
        NaN where sma is NaN (insufficient history to compute the SMA).

    Example::

        from quant_stack.factors import sma_200
        from quant_stack.signals.trend import trend_filter_signal

        trend_sma = sma_200(close)
        trend = trend_filter_signal(close, trend_sma)

        # Use as eligibility mask for momentum ranking
        from quant_stack.signals.momentum import relative_momentum_ranking_signal
        from quant_stack.factors import momentum_63

        mom = momentum_63(close)
        sig = relative_momentum_ranking_signal(
            mom, top_n=3, eligible=trend.signals.astype(bool)
        )
    """
    if close.shape != sma.shape:
        raise ValueError(
            f"close shape {close.shape} must match sma shape {sma.shape}"
        )
    if not close.columns.equals(sma.columns):
        raise ValueError("close and sma must have identical columns (symbols)")

    # Binary signal: 1.0 = above trend, 0.0 = below trend
    above = (close > sma).astype(float)
    # Preserve NaN where SMA is NaN (not enough history)
    above[sma.isna()] = float("nan")

    return SignalFrame(
        signals=above,
        strength=above.copy(),
        strategy_name=strategy_name,
    )


def as_eligibility_mask(trend: SignalFrame) -> pd.DataFrame:
    """Convert a trend filter SignalFrame to a boolean eligibility DataFrame.

    Convenience wrapper for use with relative_momentum_ranking_signal().
    NaN cells are treated as ineligible (False).

    Args:
        trend: Output of trend_filter_signal().

    Returns:
        Boolean DataFrame (same shape as trend.signals).
    """
    return trend.signals.fillna(0.0).astype(bool)
