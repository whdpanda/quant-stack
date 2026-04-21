"""CSV DataLoader for daily OHLCV data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_stack.data.loaders.base import DataLoader


class CsvDataLoader(DataLoader):
    """Load daily OHLCV data from a CSV file.

    Supported layouts
    -----------------
    Long (tidy) — preferred, one row per (date, symbol):
        date,symbol,open,high,low,close,volume[,adj_close]
        2024-01-02,SPY,470.0,472.5,469.0,471.0,1000000

    No-symbol — one row per date, symbol inferred from filename:
        date,open,high,low,close,volume
        2024-01-02,470.0,472.5,469.0,471.0,1000000

    Column names are case-insensitive and leading/trailing whitespace is ignored.

    Args:
        date_col: Name of the date column in the source file (default "date").
        read_csv_kwargs: Extra keyword arguments forwarded to ``pd.read_csv``.
    """

    def __init__(
        self,
        date_col: str = "date",
        **read_csv_kwargs: Any,
    ) -> None:
        self.date_col = date_col
        self.read_csv_kwargs = read_csv_kwargs

    def load(self, path: Path | str) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

        df = pd.read_csv(path, **self.read_csv_kwargs)
        df = self._normalise_columns(df)

        # If no 'symbol' column, inject from filename
        if "symbol" not in df.columns:
            df["symbol"] = self._infer_symbol(path)

        # Rename non-standard date column if needed
        if self.date_col.lower() != "date" and self.date_col.lower() in df.columns:
            df = df.rename(columns={self.date_col.lower(): "date"})

        return df
