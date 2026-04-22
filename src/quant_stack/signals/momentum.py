"""Momentum-based signal generators.

Two generators are provided:

absolute_momentum
    Long when a symbol's own momentum exceeds a threshold.
    This is a "time-series momentum" signal — each symbol is evaluated
    independently. No cross-sectional comparison.
    Reference: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum".

relative_momentum_ranking
    Long the top-N symbols by cross-sectional momentum rank.
    This is the standard "cross-sectional momentum" / ETF rotation signal.
    Reference: Antonacci (2012) "Risk Premia Harvesting Through Dual Momentum".

Transaction cost note
---------------------
Relative momentum ranking can produce high turnover when ranks are
unstable near the cutoff. Consider adding a buffer zone (e.g. exit only
when rank > top_n + buffer) to reduce unnecessary rebalancing.
This is not implemented here — it belongs in the backtest/execution layer
as a strategy-level decision.
"""

from __future__ import annotations

import pandas as pd

from quant_stack.signals.base import SignalFrame


def absolute_momentum_signal(
    momentum: pd.DataFrame,
    threshold: float = 0.0,
    strategy_name: str = "absolute_momentum",
) -> SignalFrame:
    """Long if momentum > threshold, Flat otherwise.

    This is a binary long/flat signal with no cross-sectional comparison.
    Each symbol is treated independently.

    Args:
        momentum: Factor values from momentum_21/63/126 (DatetimeIndex × symbols).
        threshold: Minimum momentum required to go long (default 0.0 = positive momentum).
                   Set > 0 to require a minimum return before going long
                   (e.g. 0.02 = must be up 2% over the window).
        strategy_name: Identifier for this signal set.

    Returns:
        SignalFrame with binary signals (0.0 or 1.0) and strength = signals.
        NaN rows are preserved where momentum is NaN (insufficient history).
    """
    # Binary signal: 1.0 = long, 0.0 = flat
    # NaN where momentum is NaN (not enough history) — intentional
    above = momentum > threshold        # bool, NaN where momentum is NaN
    signals = above.astype(float)       # True→1.0, False→0.0, NaN stays float NaN
    signals[momentum.isna()] = float("nan")  # restore NaNs masked by astype

    return SignalFrame(
        signals=signals,
        strength=signals.copy(),
        strategy_name=strategy_name,
    )


def relative_momentum_ranking_signal(
    momentum: pd.DataFrame,
    top_n: int = 3,
    eligible: pd.DataFrame | None = None,
    strategy_name: str = "relative_momentum_ranking",
) -> SignalFrame:
    """Long the top-N symbols by cross-sectional momentum rank.

    On each date, symbols are ranked by momentum (descending). The top_n
    symbols receive signal = 1.0; the rest receive signal = 0.0.

    Ties are broken by rank order (first occurrence wins). Symbols with
    NaN momentum are excluded from ranking and receive signal = NaN.

    Strength is normalized: rank-1 = 1.0, rank-top_n = ~1/top_n, others = 0.0.
    This allows downstream position sizing to over/underweight within the
    long basket based on relative conviction.

    Args:
        momentum: Factor values (DatetimeIndex × symbols).
        top_n: Number of symbols to hold long simultaneously.
        eligible: Optional boolean mask (same shape as momentum).
                  Only eligible symbols participate in ranking.
                  Useful for applying a trend filter before ranking.
        strategy_name: Identifier for this signal set.

    Returns:
        SignalFrame with signals, strength, and ranks fields populated.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")

    # Apply eligibility filter: mask out ineligible symbols
    masked_momentum = momentum.copy()
    if eligible is not None:
        if eligible.shape != momentum.shape:
            raise ValueError(
                f"eligible shape {eligible.shape} must match momentum shape {momentum.shape}"
            )
        masked_momentum[~eligible.astype(bool)] = float("nan")

    # Cross-sectional rank per day (ascending=False → rank 1 = highest momentum)
    ranks = masked_momentum.rank(axis=1, ascending=False, method="first", na_option="keep")

    # Long signal: rank ≤ top_n
    signals = (ranks <= top_n).astype(float)
    signals[ranks.isna()] = float("nan")  # preserve NaN where rank is NaN

    # Strength: linearly scaled within the long basket
    # rank 1 → strength = 1.0, rank top_n → strength = 1/top_n
    # Ensures sum-of-strengths = constant regardless of top_n
    strength = signals.copy()
    long_mask = ranks <= top_n
    strength[long_mask] = (1.0 - (ranks[long_mask] - 1.0) / top_n).clip(lower=0.0)
    strength[~long_mask & ~ranks.isna()] = 0.0
    strength[ranks.isna()] = float("nan")

    return SignalFrame(
        signals=signals,
        strength=strength,
        ranks=ranks,
        eligible=eligible,
        strategy_name=strategy_name,
    )
