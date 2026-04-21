"""Abstract Strategy interface consumed by the backtest runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses must implement ``generate_signals``, which receives a
    close-price DataFrame and returns a same-shaped boolean/float signal
    DataFrame (True / 1.0 = long, False / 0.0 = flat).
    """

    name: str = "base_strategy"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        """Compute entry/exit signals.

        Args:
            close: DataFrame with DatetimeIndex, columns = symbol names,
                   values = adjusted close prices.

        Returns:
            DataFrame of the same shape. Non-zero / True means "hold long".
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
