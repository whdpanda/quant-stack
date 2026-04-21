"""Abstract data provider interface.

All providers must return a tidy DataFrame with MultiIndex columns
(field, symbol) or a dict[symbol, DataFrame]. The `fetch` method is
the single required entry point; caching is provider-specific.

Polars compatibility note: when Polars support is added, implement a
`fetch_polars` method returning pl.DataFrame. The abstract interface
here stays pandas-only until that migration is intentional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from quant_stack.core.schemas import DataConfig


class DataProvider(ABC):
    """Base class for all data providers."""

    @abstractmethod
    def fetch(self, config: DataConfig) -> pd.DataFrame:
        """Fetch OHLCV data for all symbols in config.

        Returns:
            DataFrame with DatetimeIndex and MultiIndex columns (field, symbol).
            Fields: open, high, low, close, volume.
        """

    def fetch_close(self, config: DataConfig) -> pd.DataFrame:
        """Convenience: return only adjusted close prices.

        fetch() returns MultiIndex columns with (symbol, field) ordering,
        so close prices live at level 1, not level 0.
        """
        df = self.fetch(config)
        if isinstance(df.columns, pd.MultiIndex):
            return df.xs("close", axis=1, level=1)
        return df[["close"]] if "close" in df.columns else df

    @staticmethod
    def _validate_date_range(start: date, end: date) -> None:
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
