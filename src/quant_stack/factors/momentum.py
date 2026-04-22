"""Price momentum factors.

Momentum is defined as total price return over a lookback window:

    momentum[T] = close[T] / close[T - window] - 1

This is the standard cross-sectional momentum definition used in
Jegadeesh & Titman (1993) and subsequent ETF rotation literature.

Lookback windows used:
    21  ≈ 1 month
    63  ≈ 1 quarter
    126 ≈ 6 months

Transaction cost note: 1-month momentum (momentum_21) exhibits stronger
short-term reversal effects and higher turnover than 63/126-day versions.
For low-frequency ETF strategies, 63-day and 126-day tend to have better
risk-adjusted returns net of costs. Include transaction costs in any
backtest before comparing windows.

Walk-forward note: momentum signals should be validated with walk-forward
analysis (expanding or rolling window) to avoid overfitting to a specific
lookback. These functions produce raw factors; parameter selection belongs
to the research layer.
"""

from __future__ import annotations

import pandas as pd

from quant_stack.factors.base import validate_close


def momentum(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Price momentum over *window* trading days.

    momentum[T] = close[T] / close[T - window] - 1

    Args:
        close: Wide-format adjusted close prices (DatetimeIndex × symbols).
        window: Lookback period in trading days.

    Returns:
        DataFrame of the same shape. NaN for the first *window* rows.
    """
    validate_close(close)
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    return close.pct_change(periods=window)


def momentum_21(close: pd.DataFrame) -> pd.DataFrame:
    """21-day (≈ 1-month) price momentum."""
    return momentum(close, 21)


def momentum_63(close: pd.DataFrame) -> pd.DataFrame:
    """63-day (≈ 1-quarter) price momentum."""
    return momentum(close, 63)


def momentum_126(close: pd.DataFrame) -> pd.DataFrame:
    """126-day (≈ 6-month) price momentum."""
    return momentum(close, 126)
