from __future__ import annotations

from datetime import date

import pandas as pd

from quant_stack.core.schemas import DataConfig
from quant_stack.data.providers.yahoo import YahooProvider


def _config(tmp_path) -> DataConfig:
    return DataConfig(
        symbols=["SPY"],
        start=date(2024, 1, 2),
        end=date(2024, 1, 5),
        cache_dir=str(tmp_path),
    )


def test_load_cache_accepts_valid_yfinance_ohlcv(tmp_path) -> None:
    path = tmp_path / "SPY.parquet"
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000],
        },
        index=pd.bdate_range("2024-01-02", periods=4),
    )
    df.index.name = "date"
    df.to_parquet(path)

    cached = YahooProvider._load_cache(path, _config(tmp_path))

    assert cached is not None
    assert cached.columns.tolist() == ["open", "high", "low", "close", "volume"]
    assert cached.index.min().date() == date(2024, 1, 2)
    assert cached.index.max().date() == date(2024, 1, 5)


def test_load_cache_rejects_close_only_stooq_wide_frame(tmp_path) -> None:
    path = tmp_path / "SPY.parquet"
    close_only = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 103.0]},
        index=pd.bdate_range("2024-01-02", periods=4),
    )
    close_only.index.name = "date"
    close_only.to_parquet(path)

    assert YahooProvider._load_cache(path, _config(tmp_path)) is None


def test_load_cache_rejects_synthetic_zero_volume_ohlcv(tmp_path) -> None:
    path = tmp_path / "SPY.parquet"
    idx = pd.bdate_range("2024-01-02", periods=6)
    df = pd.DataFrame(
        {
            "open": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "low": [98.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [99.5, 100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1_000_000, 0, 0, 0, 0, 0],
        },
        index=idx,
    )
    df.index.name = "date"
    df.to_parquet(path)

    assert YahooProvider._load_cache(path, _config(tmp_path)) is None


def test_load_cache_rejects_non_datetime_index(tmp_path) -> None:
    path = tmp_path / "SPY.parquet"
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000],
        },
        index=["2024-01-02"],
    )
    df.to_parquet(path)

    assert YahooProvider._load_cache(path, _config(tmp_path)) is None
