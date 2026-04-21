"""DataRepository — unified service for loading local OHLCV files.

This is the primary interface for the research and portfolio layers.
It does NOT connect to any online data source; for live/cached data
use DataProvider (data/providers/yahoo.py).

File layout conventions (applied in priority order)
----------------------------------------------------
1. Per-symbol parquet  : {data_dir}/{SYMBOL}.parquet
2. Per-symbol CSV      : {data_dir}/{SYMBOL}.csv
3. Combined parquet    : {data_dir}/combined.parquet  (must have 'symbol' column)
4. Combined CSV        : {data_dir}/combined.csv      (must have 'symbol' column)

Output formats
--------------
load()        → tidy long format — (date, symbol, open, high, low, close, volume, adj_close)
load_close()  → wide format     — DatetimeIndex × symbol columns (adj_close or close)
                                   compatible with vectorbt's Portfolio.from_signals()

Survivorship-bias note
----------------------
DataRepository loads exactly what is on disk. It does not filter out
delisted symbols, backfill gaps beyond fill_limit, or add synthetic bars.
If your dataset suffers from survivorship bias (only surviving tickers),
that is a data-preparation concern, not something to silently "fix" here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from quant_stack.core.exceptions import DataProviderError
from quant_stack.data.loaders.base import CANONICAL_COLUMNS, DataLoader
from quant_stack.data.loaders.csv_loader import CsvDataLoader
from quant_stack.data.loaders.parquet_loader import ParquetDataLoader
from quant_stack.data.validation import DataValidator, ValidationConfig


class DataRepository:
    """Load, validate, and serve local OHLCV data.

    Args:
        data_dir: Directory containing OHLCV files.
        loader: DataLoader instance to use. If None, auto-selected from
                the first matching file found in data_dir.
        validator: DataValidator instance. If None, defaults are applied.
    """

    def __init__(
        self,
        data_dir: str | Path,
        loader: DataLoader | None = None,
        validator: DataValidator | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self._loader = loader
        self._validator = validator or DataValidator()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(
        self,
        symbols: list[str],
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Load tidy long-format OHLCV data for the given symbols.

        Args:
            symbols: List of ticker symbols (case-insensitive).
            start: Inclusive start date. None = no lower bound.
            end: Inclusive end date. None = no upper bound.

        Returns:
            Tidy DataFrame with CANONICAL_COLUMNS, sorted by (symbol, date).

        Raises:
            DataProviderError: If no file is found for a requested symbol.
        """
        symbols_upper = [s.strip().upper() for s in symbols]
        frames: list[pd.DataFrame] = []

        for symbol in symbols_upper:
            df = self._load_symbol(symbol)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined = self._validator.validate(combined)
        combined = self._filter_date_range(combined, start, end)
        return combined[CANONICAL_COLUMNS].reset_index(drop=True)

    def load_close(
        self,
        symbols: list[str],
        start: date | str | None = None,
        end: date | str | None = None,
        price: Literal["adj_close", "close"] = "adj_close",
    ) -> pd.DataFrame:
        """Return close prices in wide format for vectorbt / PyPortfolioOpt.

        Args:
            symbols: Ticker symbols to include.
            start: Inclusive start date.
            end: Inclusive end date.
            price: Which price column to use — "adj_close" (default) or "close".

        Returns:
            DataFrame with DatetimeIndex and one column per symbol.
            Column name = symbol string.
        """
        tidy = self.load(symbols, start, end)
        wide = tidy.pivot_table(index="date", columns="symbol", values=price)
        wide.index = pd.to_datetime(wide.index)
        wide.index.name = "date"
        wide.columns.name = None
        return wide

    def available_symbols(self) -> list[str]:
        """Return symbols discoverable from per-symbol files in data_dir.

        Does not inspect combined files; only scans {SYMBOL}.csv and
        {SYMBOL}.parquet patterns.
        """
        symbols: list[str] = []
        for path in sorted(self.data_dir.iterdir()):
            if path.suffix in (".csv", ".parquet") and path.stem.lower() not in (
                "combined", "all"
            ):
                symbols.append(path.stem.upper())
        return symbols

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load raw data for one symbol, trying file formats in priority order."""
        # Priority: per-symbol parquet > per-symbol csv > combined parquet > combined csv
        candidates = [
            (self.data_dir / f"{symbol}.parquet", ParquetDataLoader()),
            (self.data_dir / f"{symbol.lower()}.parquet", ParquetDataLoader()),
            (self.data_dir / f"{symbol}.csv", CsvDataLoader()),
            (self.data_dir / f"{symbol.lower()}.csv", CsvDataLoader()),
        ]

        for path, default_loader in candidates:
            if path.exists():
                loader = self._loader or default_loader
                logger.debug(f"Loading {symbol} from {path}")
                return loader.load(path)

        # Fall back to combined files
        for combined_name in ("combined.parquet", "combined.csv"):
            combined_path = self.data_dir / combined_name
            if combined_path.exists():
                logger.debug(f"Loading {symbol} from combined file {combined_path}")
                loader = self._loader or self._loader_for(combined_path)
                df = loader.load(combined_path)
                df.columns = [c.strip().lower() for c in df.columns]
                if "symbol" in df.columns:
                    mask = df["symbol"].str.upper() == symbol
                    subset = df.loc[mask]
                    if not subset.empty:
                        return subset.copy()

        raise DataProviderError(
            f"No data file found for symbol '{symbol}' in {self.data_dir}. "
            f"Expected: {symbol}.parquet, {symbol}.csv, or a combined file "
            "with a 'symbol' column."
        )

    @staticmethod
    def _loader_for(path: Path) -> DataLoader:
        if path.suffix == ".parquet":
            return ParquetDataLoader()
        return CsvDataLoader()

    @staticmethod
    def _filter_date_range(
        df: pd.DataFrame,
        start: date | str | None,
        end: date | str | None,
    ) -> pd.DataFrame:
        if start is None and end is None:
            return df
        dates = pd.to_datetime(df["date"])
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= dates >= pd.Timestamp(start)
        if end is not None:
            mask &= dates <= pd.Timestamp(end)
        return df.loc[mask]
