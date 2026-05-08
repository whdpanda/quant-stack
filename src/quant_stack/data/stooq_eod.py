"""Minimal Stooq EOD close downloader.

Contract: this module returns close-only price frames for Stooq-backed
monitoring flows. It must not write local cache files, synthesize OHLCV bars
from closes, or create yfinance-style ``data/{ticker}.parquet`` files.
"""

from __future__ import annotations

import os
from datetime import date
from io import StringIO
from time import sleep
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from loguru import logger

from quant_stack.core.exceptions import DataProviderError

DEFAULT_MAX_MISSING_RATIO = 0.01
DEFAULT_MAX_MISSING_DAYS = 5


def fetch_stooq_close(
    symbols: list[str],
    start: date,
    end: date,
    *,
    api_key: str | None = None,
    timeout_sec: int = 60,
    max_retries: int = 3,
    max_missing_ratio: float = DEFAULT_MAX_MISSING_RATIO,
    max_missing_days: int = DEFAULT_MAX_MISSING_DAYS,
    min_rows: int | None = None,
) -> pd.DataFrame:
    """Return daily close prices from Stooq with original ticker columns.

    The returned frame is close-only: columns are exactly the requested ticker
    symbols, not ``open/high/low/close/volume``. This helper intentionally has
    no parquet/cache side effects.
    """
    if not symbols:
        raise DataProviderError("Stooq close download requires at least one ticker")

    frames: list[pd.Series] = []
    for symbol in symbols:
        frames.append(
            _download_one(
                symbol,
                start,
                end,
                api_key=api_key,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
            )
        )

    close = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    missing = [symbol for symbol in symbols if symbol not in close.columns]
    if close.empty:
        raise DataProviderError("Stooq returned empty close data")
    if missing:
        raise DataProviderError(f"Stooq data missing required tickers: {missing}")
    if close.index.has_duplicates:
        raise DataProviderError("Stooq close data contains duplicate dates")
    if close[symbols].isna().any().any():
        close = _align_common_dates(
            close[symbols],
            max_missing_ratio=max_missing_ratio,
            max_missing_days=max_missing_days,
        )
    if min_rows is not None and len(close) < min_rows:
        raise DataProviderError(
            f"Stooq close data has only {len(close)} common rows; required at least {min_rows}"
        )
    if (close[symbols] <= 0).any().any():
        raise DataProviderError("Stooq close data contains non-positive prices")
    return close[symbols]


def _align_common_dates(
    close: pd.DataFrame,
    *,
    max_missing_ratio: float,
    max_missing_days: int,
) -> pd.DataFrame:
    missing_counts = close.isna().sum()
    missing = missing_counts[missing_counts > 0]
    allowed_missing = max(max_missing_days, int(len(close) * max_missing_ratio))
    too_many_missing = missing[missing > allowed_missing]
    if not too_many_missing.empty:
        details = "; ".join(_ticker_gap_summary(close, ticker) for ticker in too_many_missing.index)
        raise DataProviderError(
            "Stooq close data has too many missing dates after ticker alignment: "
            f"{details}"
        )

    aligned = close.dropna(how="any")
    dropped = len(close) - len(aligned)
    if aligned.empty:
        details = "; ".join(_ticker_gap_summary(close, ticker) for ticker in missing.index)
        raise DataProviderError(
            "Stooq close data has no common dates across required tickers: "
            f"{details}"
        )
    if dropped:
        details = "; ".join(_ticker_gap_summary(close, ticker) for ticker in missing.index)
        logger.warning(
            f"Dropped {dropped} Stooq rows with incomplete ticker coverage: {details}"
        )
    return aligned


def _ticker_gap_summary(close: pd.DataFrame, ticker: str) -> str:
    series = close[ticker]
    non_na = series.dropna()
    first = non_na.index.min().date().isoformat() if not non_na.empty else "none"
    last = non_na.index.max().date().isoformat() if not non_na.empty else "none"
    return f"{ticker} missing={int(series.isna().sum())} first={first} last={last}"


def _to_stooq_symbol(symbol: str) -> str:
    return f"{symbol.lower()}.us"


def _download_one(
    symbol: str,
    start: date,
    end: date,
    *,
    api_key: str | None,
    timeout_sec: int,
    max_retries: int,
) -> pd.Series:
    stooq_symbol = _to_stooq_symbol(symbol)
    params = {
        "s": stooq_symbol,
        "d1": start.strftime("%Y%m%d"),
        "d2": end.strftime("%Y%m%d"),
        "i": "d",
    }
    api_key = api_key or os.environ.get("STOOQ_APIKEY")
    if api_key:
        params["apikey"] = api_key
    url = f"https://stooq.com/q/d/l/?{urlencode(params)}"

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(url, timeout=timeout_sec) as response:
                text = response.read().decode("utf-8")
            return _parse_stooq_csv(symbol, stooq_symbol, text)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                sleep(1)

    raise DataProviderError(
        f"Stooq download failed for {symbol} ({stooq_symbol}): {last_error}"
    )


def _parse_stooq_csv(symbol: str, stooq_symbol: str, text: str) -> pd.Series:
    if not text.lstrip().startswith("Date,"):
        preview = text.strip().splitlines()[0] if text.strip() else "empty response"
        raise DataProviderError(
            f"Stooq returned an unexpected response for {symbol} ({stooq_symbol}): {preview}"
        )

    df = pd.read_csv(StringIO(text))
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        raise DataProviderError(
            f"Stooq returned no usable close data for {symbol} ({stooq_symbol})"
        )

    df = df.assign(Date=pd.to_datetime(df["Date"]))
    if df["Date"].duplicated().any():
        raise DataProviderError(
            f"Stooq close data contains duplicate dates for {symbol} ({stooq_symbol})"
        )

    series = pd.to_numeric(df.set_index("Date")["Close"], errors="coerce").rename(symbol)
    if series.empty:
        raise DataProviderError(
            f"Stooq close data is empty after parsing for {symbol} ({stooq_symbol})"
        )
    if series.isna().any():
        raise DataProviderError(
            f"Stooq close data contains NaN values for {symbol} ({stooq_symbol})"
        )
    if (series <= 0).any():
        raise DataProviderError(
            f"Stooq close data contains non-positive prices for {symbol} ({stooq_symbol})"
        )
    return series.astype(float).sort_index()
