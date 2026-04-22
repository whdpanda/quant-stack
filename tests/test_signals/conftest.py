"""Fixtures for signal tests — pre-computed factor DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_stack.factors import momentum_63, sma_200


def _make_close(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({
        "SPY": 400.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)),
        "QQQ": 300.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "IEF": 100.0 * np.cumprod(1 + rng.normal(0.0001, 0.005, n)),
    }, index=idx)


@pytest.fixture()
def sample_close() -> pd.DataFrame:
    return _make_close()


@pytest.fixture()
def sample_momentum(sample_close) -> pd.DataFrame:
    return momentum_63(sample_close)


@pytest.fixture()
def sample_sma(sample_close) -> pd.DataFrame:
    return sma_200(sample_close)
