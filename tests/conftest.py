"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_stack.core.schemas import BacktestConfig, DataConfig


@pytest.fixture()
def sample_close() -> pd.DataFrame:
    """Synthetic close price DataFrame with two symbols, 252 rows."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", periods=252, freq="B")
    data = {
        "SPY": 400.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, 252)),
        "QQQ": 320.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, 252)),
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture()
def base_data_config() -> DataConfig:
    return DataConfig(
        symbols=["SPY", "QQQ"],
        start=date(2022, 1, 1),
        end=date(2022, 12, 31),
    )


@pytest.fixture()
def base_backtest_config(base_data_config: DataConfig) -> BacktestConfig:
    return BacktestConfig(
        data=base_data_config,
        strategy_name="sma_cross",
        initial_cash=100_000,
    )
