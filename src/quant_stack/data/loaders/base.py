"""DataLoader abstract interface for local OHLCV files.

Output contract
---------------
All concrete loaders must return a DataFrame conforming to CANONICAL_COLUMNS.
Validation and cleaning are NOT the loader's responsibility — that belongs to
DataValidator. Loaders only read bytes from disk and return a raw DataFrame.

Polars compatibility
--------------------
`load_polars()` is declared here so the interface exists before the migration.
Subclasses that support Polars override it and set `supports_polars = True`.
Callers can check `loader.supports_polars` before calling `load_polars()`.

Column naming conventions
-------------------------
Required (must be present after loading):
    date    — trade date, any parseable date string or datetime
    symbol  — ticker symbol (string)
    open, high, low, close — adjusted or unadjusted OHLC prices (float)
    volume  — share/unit volume (float; may be 0 for illiquid days)

Optional:
    adj_close — split/dividend-adjusted close (float)
                If absent, DataValidator will set adj_close = close.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

# ── Schema constants ──────────────────────────────────────────────────────────

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"date", "symbol", "open", "high", "low", "close", "volume"}
)
OPTIONAL_COLUMNS: frozenset[str] = frozenset({"adj_close"})
CANONICAL_COLUMNS: list[str] = [
    "date", "symbol", "open", "high", "low", "close", "volume", "adj_close"
]
PRICE_COLUMNS: list[str] = ["open", "high", "low", "close", "adj_close"]
NUMERIC_COLUMNS: list[str] = ["open", "high", "low", "close", "volume", "adj_close"]


# ── Abstract loader ───────────────────────────────────────────────────────────

class DataLoader(ABC):
    """Load daily OHLCV data from a single local file.

    Returns a raw DataFrame — callers (typically DataRepository) are
    responsible for applying DataValidator before use.

    If the source file has no 'symbol' column, the loader infers the symbol
    from the file stem (e.g. ``SPY.csv`` → symbol = ``"SPY"``).
    """

    supports_polars: bool = False  # class-level flag, overridden by Polars subclasses

    @abstractmethod
    def load(self, path: Path | str) -> pd.DataFrame:
        """Read the file at *path* and return a raw DataFrame.

        The returned DataFrame must have at least the columns in
        REQUIRED_COLUMNS (case-insensitive, strip whitespace). Column
        normalisation (lowercase, strip) is the loader's responsibility;
        all other validation is handled by DataValidator.

        Args:
            path: Absolute or relative path to the data file.

        Returns:
            Raw DataFrame. Index is arbitrary (RangeIndex is fine).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed at all.
        """

    def load_polars(self, path: Path | str) -> Any:
        """Load into a Polars DataFrame — not yet implemented.

        Reserved for a future ``PolarsParquetLoader`` subclass.
        Check ``loader.supports_polars`` before calling this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support Polars loading. "
            "Install polars and use a Polars-compatible loader."
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase column names and strip whitespace."""
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    @staticmethod
    def _infer_symbol(path: Path) -> str:
        """Derive symbol from filename stem: 'spy.csv' → 'SPY'."""
        return path.stem.upper()
