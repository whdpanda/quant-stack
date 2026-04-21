"""Simple Moving Average crossover strategy — reference implementation."""

from __future__ import annotations

import pandas as pd

from quant_stack.research.base import Strategy


class SmaCrossStrategy(Strategy):
    """Go long when fast SMA crosses above slow SMA; exit when it crosses below.

    Params:
        fast_window: Lookback period for the fast MA (default: 20).
        slow_window: Lookback period for the slow MA (default: 50).
    """

    name = "sma_cross"

    def __init__(self, fast_window: int = 20, slow_window: int = 50) -> None:
        super().__init__(fast_window=fast_window, slow_window=slow_window)
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        fast = close.rolling(self.fast_window).mean()
        slow = close.rolling(self.slow_window).mean()
        return (fast > slow).astype(float)
