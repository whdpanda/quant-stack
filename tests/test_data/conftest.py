"""Shared fixtures for data layer tests.

All test data is generated in-process using pytest's tmp_path fixture.
No network calls, no external files required.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest


# ── Raw DataFrame builders ────────────────────────────────────────────────────

def make_ohlcv(
    symbol: str,
    start: str = "2024-01-02",
    periods: int = 20,
    base_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic tidy OHLCV DataFrame for one symbol."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="B")
    closes = base_price * np.cumprod(1 + rng.normal(0.0003, 0.01, periods))
    opens = closes * (1 + rng.uniform(-0.005, 0.005, periods))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.01, periods))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.01, periods))
    volumes = rng.integers(500_000, 5_000_000, periods).astype(float)

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "symbol": symbol,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volumes,
    })


# ── File fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def spy_csv(tmp_path) -> tuple[str, pd.DataFrame]:
    """Single-symbol CSV without adj_close column."""
    df = make_ohlcv("SPY", base_price=470.0, seed=1)
    path = tmp_path / "SPY.csv"
    df.to_csv(path, index=False)
    return str(path), df


@pytest.fixture()
def qqq_csv(tmp_path) -> tuple[str, pd.DataFrame]:
    """Single-symbol CSV without a symbol column (symbol inferred from filename)."""
    df = make_ohlcv("QQQ", base_price=380.0, seed=2)
    df_no_symbol = df.drop(columns=["symbol"])
    path = tmp_path / "QQQ.csv"
    df_no_symbol.to_csv(path, index=False)
    return str(path), df


@pytest.fixture()
def spy_parquet(tmp_path) -> tuple[str, pd.DataFrame]:
    """Single-symbol Parquet with DatetimeIndex (yfinance cache format)."""
    df = make_ohlcv("SPY", base_price=470.0, seed=1)
    df_indexed = df.copy()
    df_indexed.index = pd.to_datetime(df_indexed["date"])
    df_indexed.index.name = "date"
    df_indexed = df_indexed.drop(columns=["date"])
    path = tmp_path / "SPY.parquet"
    df_indexed.to_parquet(path)
    return str(path), df


@pytest.fixture()
def combined_csv(tmp_path) -> tuple[str, pd.DataFrame]:
    """Combined CSV with two symbols."""
    spy = make_ohlcv("SPY", base_price=470.0, seed=1)
    qqq = make_ohlcv("QQQ", base_price=380.0, seed=2)
    combined = pd.concat([spy, qqq], ignore_index=True)
    path = tmp_path / "combined.csv"
    combined.to_csv(path, index=False)
    return str(path), combined


@pytest.fixture()
def data_dir_with_files(tmp_path) -> str:
    """Directory containing SPY.csv and QQQ.csv."""
    for symbol, base, seed in [("SPY", 470.0, 1), ("QQQ", 380.0, 2)]:
        df = make_ohlcv(symbol, base_price=base, seed=seed)
        df.to_csv(tmp_path / f"{symbol}.csv", index=False)
    return str(tmp_path)


@pytest.fixture()
def data_dir_with_parquet(tmp_path) -> str:
    """Directory containing SPY.parquet and QQQ.parquet."""
    for symbol, base, seed in [("SPY", 470.0, 1), ("QQQ", 380.0, 2)]:
        df = make_ohlcv(symbol, base_price=base, seed=seed)
        df.to_parquet(tmp_path / f"{symbol}.parquet")
    return str(tmp_path)
