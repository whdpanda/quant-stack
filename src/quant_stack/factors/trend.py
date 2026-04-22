"""Trend / moving average factors.

SMA is used as a trend filter rather than a return predictor.
The canonical use case is: "is close > SMA_200?" as a binary regime filter
that gates momentum or other signals.

Reference: Faber (2007) "A Quantitative Approach to Tactical Asset Allocation"
uses the 10-month (≈ 200-day) SMA as a market timing filter for ETFs.
"""

from __future__ import annotations

import pandas as pd

from quant_stack.factors.base import validate_close


def sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average over *window* trading days.

    sma[T] = mean(close[T - window + 1 : T + 1])

    Uses min_periods=window so that values are NaN until the full window
    of history is available. This is the correct behaviour for a live
    system — never use a partial window as if it were a full window.

    Args:
        close: Wide-format adjusted close prices (DatetimeIndex × symbols).
        window: Lookback period in trading days.

    Returns:
        DataFrame of the same shape. NaN for the first *window - 1* rows.
    """
    validate_close(close)
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    return close.rolling(window=window, min_periods=window).mean()


def sma_50(close: pd.DataFrame) -> pd.DataFrame:
    """50-day simple moving average."""
    return sma(close, 50)


def sma_200(close: pd.DataFrame) -> pd.DataFrame:
    """200-day simple moving average."""
    return sma(close, 200)
