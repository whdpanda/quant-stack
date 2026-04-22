"""Realized volatility factors.

Volatility is used for:
1. Risk-adjusted momentum: momentum / volatility (Sharpe-like signal)
2. Position sizing: inverse-volatility weighting
3. Regime detection: high vol = risk-off, low vol = risk-on

The standard low-frequency definition is annualized realized volatility
from daily log returns, computed with a rolling window.

Note on log vs simple returns for volatility:
    Log returns are used because their distribution is closer to normal,
    making the std() a more stable estimator. For daily returns the
    difference is negligible, but log returns are the convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_stack.factors.base import log_returns, validate_close

_TRADING_DAYS_PER_YEAR: int = 252


def realized_volatility(
    close: pd.DataFrame,
    window: int,
    annualize: bool = True,
) -> pd.DataFrame:
    """Annualized realized volatility from daily log returns.

    vol[T] = std(log_returns[T - window + 1 : T + 1]) * sqrt(252)

    Args:
        close: Wide-format adjusted close prices (DatetimeIndex × symbols).
        window: Rolling window in trading days.
        annualize: If True (default), multiply by sqrt(252).

    Returns:
        DataFrame of the same shape. NaN for the first *window* rows
        (window rows of returns require window + 1 rows of prices).
    """
    validate_close(close)
    if window <= 1:
        raise ValueError(f"window must be > 1 for meaningful volatility, got {window}")

    ret = log_returns(close)
    vol = ret.rolling(window=window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(_TRADING_DAYS_PER_YEAR)
    return vol


def volatility_20(close: pd.DataFrame) -> pd.DataFrame:
    """20-day annualized realized volatility (≈ 1-month)."""
    return realized_volatility(close, 20)
