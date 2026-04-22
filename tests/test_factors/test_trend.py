"""Tests for SMA trend factor functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_stack.factors.trend import sma, sma_50, sma_200


class TestSmaBase:
    def test_shape_preserved(self, random_close) -> None:
        assert sma(random_close, 50).shape == random_close.shape

    def test_first_window_minus_one_rows_nan(self, random_close) -> None:
        """First (window - 1) rows must be NaN; row[window-1] is the first valid."""
        result = sma(random_close, 50)
        assert result.iloc[:49].isna().all().all()
        assert result.iloc[49].notna().all()

    def test_flat_prices_sma_equals_price(self, flat_close) -> None:
        result = sma(flat_close, 50)
        non_nan = result.dropna()
        pd.testing.assert_frame_equal(
            non_nan,
            flat_close.loc[non_nan.index],
            check_names=False,
        )

    def test_value_correct_for_known_prices(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        close = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=idx)
        result = sma(close, 5)
        # First valid: index 4 = mean(1..5) = 3.0
        assert abs(result.iloc[4, 0] - 3.0) < 1e-10
        # Index 9 = mean(6..10) = 8.0
        assert abs(result.iloc[9, 0] - 8.0) < 1e-10

    def test_invalid_window_raises(self, random_close) -> None:
        with pytest.raises(ValueError):
            sma(random_close, 0)


class TestSma50:
    def test_first_valid_index(self, random_close) -> None:
        result = sma_50(random_close)
        # Row 49 (0-indexed) is the first non-NaN
        assert result.iloc[:49].isna().all().all()
        assert result.iloc[49].notna().all()

    def test_trending_sma_below_current_price(self, trending_close) -> None:
        """For monotonically rising prices, SMA50 < current close."""
        result = sma_50(trending_close)
        spy_sma = result["SPY"].dropna()
        spy_close = trending_close["SPY"].loc[spy_sma.index]
        assert (spy_close > spy_sma).all()


class TestSma200:
    def test_requires_200_rows(self, random_close) -> None:
        result = sma_200(random_close)
        assert result.iloc[:199].isna().all().all()
        assert result.iloc[199].notna().all()

    def test_sma50_leads_sma200(self, trending_close) -> None:
        """For rising prices, SMA50 > SMA200 (faster MA is closer to current price)."""
        s50 = sma_50(trending_close)
        s200 = sma_200(trending_close)
        common = s50.dropna().index.intersection(s200.dropna().index)
        assert (s50.loc[common]["SPY"] > s200.loc[common]["SPY"]).all()
