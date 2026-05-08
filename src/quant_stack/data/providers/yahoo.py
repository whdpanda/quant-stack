"""yfinance-backed data provider with local parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from quant_stack.core.exceptions import DataProviderError
from quant_stack.core.schemas import DataConfig
from quant_stack.data.base import DataProvider

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
CACHE_VALIDATION_WINDOW = 5


class YahooProvider(DataProvider):
    """Fetch OHLCV data from Yahoo Finance via yfinance.

    Data is cached to ``{cache_dir}/{symbol}.parquet`` to avoid redundant
    network calls. Delete the cache files to force a fresh download.
    """

    def fetch(self, config: DataConfig) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise DataProviderError("yfinance is not installed: pip install yfinance") from e

        self._validate_date_range(config.start, config.end)
        frames: dict[str, pd.DataFrame] = {}

        for symbol in config.symbols:
            cache_path = Path(config.cache_dir) / f"{symbol}.parquet"
            df = self._load_cache(cache_path, config)

            if df is None:
                logger.info(f"Downloading {symbol} from Yahoo Finance…")
                raw = yf.download(
                    symbol,
                    start=config.start.isoformat(),
                    end=config.end.isoformat(),
                    auto_adjust=True,
                    progress=False,
                )
                if raw.empty:
                    raise DataProviderError(f"No data returned for symbol '{symbol}'")
                df = self._normalise(raw)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
                logger.debug(f"Cached {symbol} → {cache_path}")
            else:
                logger.debug(f"Loaded {symbol} from cache {cache_path}")

            frames[symbol] = df

        return pd.concat(frames, axis=1).sort_index()

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
        """Lowercase columns and ensure DatetimeIndex.

        Newer yfinance (>=0.2) returns MultiIndex columns like ('Close', 'SPY')
        even for single-ticker downloads; take only the first level.
        """
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df[OHLCV_COLUMNS]

    @staticmethod
    def _load_cache(path: Path, config: DataConfig) -> pd.DataFrame | None:
        """Return cached yfinance OHLCV DataFrame if it is valid and complete."""
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df = YahooProvider._validate_cache_frame(df, path)
        if df is None:
            return None
        if df.index.min().date() <= config.start and df.index.max().date() >= config.end:
            mask = (df.index.date >= config.start) & (df.index.date <= config.end)
            return df.loc[mask]
        return None  # cache is stale or incomplete

    @staticmethod
    def _validate_cache_frame(df: pd.DataFrame, path: Path) -> pd.DataFrame | None:
        """Reject cache files that are not credible yfinance OHLCV data."""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Ignoring invalid Yahoo cache {path}: index is not DatetimeIndex")
            return None

        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            logger.warning(f"Ignoring invalid Yahoo cache {path}: columns are MultiIndex")
            return None

        df.columns = [str(col).strip().lower() for col in df.columns]
        missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
        if missing:
            logger.warning(f"Ignoring invalid Yahoo cache {path}: missing OHLCV columns {missing}")
            return None

        df = df[OHLCV_COLUMNS].sort_index()
        if df.empty or df.index.hasnans:
            logger.warning(f"Ignoring invalid Yahoo cache {path}: empty or invalid date index")
            return None

        numeric = df.apply(pd.to_numeric, errors="coerce")
        if numeric[OHLCV_COLUMNS].isna().any().any():
            logger.warning(f"Ignoring invalid Yahoo cache {path}: non-numeric OHLCV values")
            return None

        latest = numeric.tail(CACHE_VALIDATION_WINDOW)
        same_ohlc = (
            latest["open"].eq(latest["high"])
            & latest["open"].eq(latest["low"])
            & latest["open"].eq(latest["close"])
        )
        zero_volume = latest["volume"].eq(0)
        if not latest.empty and (same_ohlc & zero_volume).all():
            logger.warning(
                f"Ignoring suspected synthetic Yahoo cache {path}: "
                "latest rows have open=high=low=close and volume=0"
            )
            return None

        return numeric
