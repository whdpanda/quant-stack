"""Local CSV / Parquet data provider.

Expected file layout:  {cache_dir}/{SYMBOL}.csv  or  {cache_dir}/{SYMBOL}.parquet
Required columns (case-insensitive): date, open, high, low, close, volume
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from quant_stack.core.exceptions import DataProviderError
from quant_stack.core.schemas import DataConfig
from quant_stack.data.base import DataProvider


class CSVProvider(DataProvider):
    """Load OHLCV data from local CSV or Parquet files."""

    def fetch(self, config: DataConfig) -> pd.DataFrame:
        self._validate_date_range(config.start, config.end)
        frames: dict[str, pd.DataFrame] = {}

        for symbol in config.symbols:
            df = self._load_symbol(symbol, config)
            mask = (df.index.date >= config.start) & (df.index.date <= config.end)
            frames[symbol] = df.loc[mask]
            logger.debug(f"Loaded {symbol} from local file ({len(frames[symbol])} rows)")

        return pd.concat(frames, axis=1).sort_index()

    # ------------------------------------------------------------------

    def _load_symbol(self, symbol: str, config: DataConfig) -> pd.DataFrame:
        base = Path(config.cache_dir)
        for ext in (".parquet", ".csv"):
            path = base / f"{symbol}{ext}"
            if path.exists():
                return self._read_file(path)
        raise DataProviderError(
            f"No local data file found for '{symbol}' in {config.cache_dir}"
        )

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df[["open", "high", "low", "close", "volume"]]
