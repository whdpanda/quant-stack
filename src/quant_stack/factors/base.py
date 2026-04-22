"""Shared utilities for factor computation.

All factor functions in this package follow the same contract:

Input
-----
close : pd.DataFrame
    Wide-format adjusted close prices.
    Index  : DatetimeIndex (daily, business-day frequency recommended)
    Columns: symbol strings (e.g. "SPY", "QQQ")
    Source : DataRepository.load_close()

Output
------
pd.DataFrame of the same shape as close.
    - NaN where insufficient history exists (enforced via min_periods)
    - Values are computed at time T using only data ≤ T  ← no look-ahead

Look-ahead bias contract
------------------------
Factor values are point-in-time: value[T] depends only on close[≤T].
When a factor drives a trading decision, the backtest/execution layer
is responsible for applying .shift(1) so that T's signal executes at T+1.
This layer never shifts signals — that would hide a design decision that
belongs to the consumer.

Survivorship bias note
----------------------
These functions compute whatever is in the input DataFrame. If the input
only contains currently-surviving symbols, the factor values inherit that
bias. Data selection is the data layer's responsibility, not ours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_close(close: pd.DataFrame) -> None:
    """Raise ValueError if close is not a valid input DataFrame."""
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(close).__name__}")
    if close.empty:
        raise ValueError("close DataFrame is empty")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(
            "close.index must be a DatetimeIndex. "
            "Use DataRepository.load_close() or convert with pd.to_datetime()."
        )
    if (close <= 0).any().any():
        import warnings
        warnings.warn(
            "close contains non-positive values. Verify data is correctly adjusted.",
            stacklevel=3,
        )


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: ln(close[T] / close[T-1]).

    Returns NaN for the first row (no prior close available).
    Used internally by volatility functions.
    """
    return np.log(close / close.shift(1))


def _apply_per_symbol(
    close: pd.DataFrame,
    func: "Callable[[pd.Series], pd.Series]",
) -> pd.DataFrame:
    """Apply func column-by-column (per symbol) and reassemble.

    Use this when a vectorized operation cannot be applied to the whole
    DataFrame at once (e.g. operations with symbol-specific parameters).
    For uniform-parameter operations (rolling, pct_change), apply directly
    to the DataFrame — pandas already operates per-column.
    """
    return pd.DataFrame(
        {col: func(close[col]) for col in close.columns},
        index=close.index,
    )
