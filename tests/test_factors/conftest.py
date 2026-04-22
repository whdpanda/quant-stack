"""Shared fixtures for factor tests.

Fixtures use deterministic synthetic prices so tests are reproducible
and verify exact expected values, not just shape/type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def flat_close() -> pd.DataFrame:
    """All prices constant at 100 — momentum = 0, vol = 0, SMA = 100."""
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    return pd.DataFrame({"SPY": 100.0, "QQQ": 200.0}, index=idx)


@pytest.fixture()
def trending_close() -> pd.DataFrame:
    """Prices increase linearly by 0.1/day for SPY, flat for QQQ."""
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    spy = 100.0 + 0.1 * np.arange(300)
    qqq = np.full(300, 200.0)
    return pd.DataFrame({"SPY": spy, "QQQ": qqq}, index=idx)


@pytest.fixture()
def random_close() -> pd.DataFrame:
    """Realistic synthetic OHLC with two symbols, 400 days."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-03", periods=400, freq="B")
    spy = 400.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, 400))
    qqq = 300.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, 400))
    return pd.DataFrame({"SPY": spy, "QQQ": qqq}, index=idx)


@pytest.fixture()
def single_symbol_close() -> pd.DataFrame:
    """Single symbol for edge-case tests."""
    rng = np.random.default_rng(99)
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300))
    return pd.DataFrame({"SPY": prices}, index=idx)
