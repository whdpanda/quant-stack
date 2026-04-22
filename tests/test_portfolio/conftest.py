"""Shared fixtures for portfolio allocator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def returns_3() -> pd.DataFrame:
    """250 days of synthetic daily returns for 3 assets."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=250, freq="B")
    data = rng.normal(loc=0.0005, scale=0.01, size=(250, 3))
    return pd.DataFrame(data, index=idx, columns=["SPY", "QQQ", "IEF"])


@pytest.fixture
def returns_1() -> pd.DataFrame:
    """250 days of returns for a single asset."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=250, freq="B")
    data = rng.normal(loc=0.0003, scale=0.008, size=(250, 1))
    return pd.DataFrame(data, index=idx, columns=["SPY"])


@pytest.fixture
def flat_returns() -> pd.DataFrame:
    """Zero-volatility returns (edge case)."""
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    return pd.DataFrame(0.0, index=idx, columns=["A", "B"])
