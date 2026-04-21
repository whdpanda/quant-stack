"""Common data transformations applied before strategy / portfolio layers."""

from __future__ import annotations

import pandas as pd


def align_and_fill(
    df: pd.DataFrame,
    method: str = "ffill",
    limit: int = 5,
) -> pd.DataFrame:
    """Align all symbols to the same date index and forward-fill gaps.

    Args:
        df: MultiIndex-column DataFrame (field, symbol).
        method: Fill method passed to DataFrame.fillna.
        limit: Maximum consecutive NaN periods to fill.
    """
    return df.fillna(method=method, limit=limit)  # type: ignore[call-arg]


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a close-price DataFrame."""
    import numpy as np
    return np.log(close / close.shift(1)).dropna()


def simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple (arithmetic) returns."""
    return close.pct_change().dropna()


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Normalise a series to rolling z-scores."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=1)
    return (series - mean) / std
