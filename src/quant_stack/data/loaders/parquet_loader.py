"""Parquet DataLoader for daily OHLCV data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_stack.data.loaders.base import DataLoader


class ParquetDataLoader(DataLoader):
    """Load daily OHLCV data from a Parquet file.

    The file must contain at least the required OHLCV columns (see
    ``loaders.base.REQUIRED_COLUMNS``). Column names are normalised to
    lowercase after loading.

    Polars note
    -----------
    Parquet is the natural migration target for Polars. When Polars support
    is added, this class will gain a ``load_polars()`` override. Until then,
    ``supports_polars = False`` and ``load_polars()`` raises NotImplementedError.

    Args:
        read_parquet_kwargs: Extra keyword arguments forwarded to ``pd.read_parquet``.
    """

    def __init__(self, **read_parquet_kwargs: Any) -> None:
        self.read_parquet_kwargs = read_parquet_kwargs

    def load(self, path: Path | str) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path.resolve()}")

        df = pd.read_parquet(path, **self.read_parquet_kwargs)
        df = self._normalise_columns(df)

        # Parquet files written by DataProvider (yahoo) use DatetimeIndex;
        # flatten it into a regular 'date' column if needed.
        if df.index.name in ("date", "Date") or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = self._normalise_columns(df)

        if "symbol" not in df.columns:
            df["symbol"] = self._infer_symbol(path)

        return df
