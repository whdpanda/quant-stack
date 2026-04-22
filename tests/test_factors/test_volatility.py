"""Tests for realized volatility factor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_stack.factors.volatility import realized_volatility, volatility_20


class TestRealizedVolatility:
    def test_shape_preserved(self, random_close) -> None:
        result = realized_volatility(random_close, 20)
        assert result.shape == random_close.shape

    def test_first_window_rows_nan(self, random_close) -> None:
        """Need window+1 prices to get window returns, so first window rows are NaN."""
        result = volatility_20(random_close)
        assert result.iloc[:20].isna().all().all()
        assert result.iloc[20].notna().all()

    def test_flat_prices_zero_volatility(self, flat_close) -> None:
        result = volatility_20(flat_close)
        non_nan = result.dropna()
        # Flat prices → 0 returns → 0 volatility
        assert (non_nan.abs() < 1e-10).all().all()

    def test_annualized_value_reasonable(self, random_close) -> None:
        """Annualized vol from daily noise=1% → ~sqrt(252)*0.01 ≈ 15.9%."""
        result = volatility_20(random_close)
        median_vol = result.dropna().median()
        # Generated with std=0.01/day, so ~15.9% annualized; allow wide tolerance
        assert (median_vol > 0.05).all()
        assert (median_vol < 0.50).all()

    def test_non_annualized_smaller(self, random_close) -> None:
        annualized = realized_volatility(random_close, 20, annualize=True)
        raw = realized_volatility(random_close, 20, annualize=False)
        factor = annualized.dropna() / raw.dropna()
        expected = np.sqrt(252)
        assert (abs(factor - expected) < 0.001).all().all()

    def test_higher_noise_higher_vol(self) -> None:
        """Verify that higher daily noise produces higher realized volatility."""
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        low_noise = pd.DataFrame(
            {"A": 100 * np.cumprod(1 + rng.normal(0, 0.005, 100))}, index=idx
        )
        high_noise = pd.DataFrame(
            {"A": 100 * np.cumprod(1 + rng.normal(0, 0.02, 100))}, index=idx
        )
        vol_low = volatility_20(low_noise).dropna().mean()
        vol_high = volatility_20(high_noise).dropna().mean()
        assert (vol_high > vol_low).all()

    def test_window_1_raises(self, random_close) -> None:
        with pytest.raises(ValueError, match="window must be > 1"):
            realized_volatility(random_close, 1)
