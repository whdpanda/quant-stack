"""Tests for momentum factor functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_stack.factors.momentum import momentum, momentum_21, momentum_63, momentum_126


class TestMomentumBase:
    def test_shape_preserved(self, random_close) -> None:
        result = momentum(random_close, 21)
        assert result.shape == random_close.shape

    def test_columns_preserved(self, random_close) -> None:
        result = momentum(random_close, 21)
        assert list(result.columns) == list(random_close.columns)

    def test_index_preserved(self, random_close) -> None:
        result = momentum(random_close, 21)
        assert result.index.equals(random_close.index)

    def test_first_window_rows_are_nan(self, random_close) -> None:
        window = 21
        result = momentum(random_close, window)
        assert result.iloc[:window].isna().all().all(), (
            "First `window` rows must be NaN (no look-ahead / partial window)"
        )

    def test_value_correct_for_known_prices(self) -> None:
        """Verify exact computation: close[T]/close[T-N] - 1."""
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        close = pd.DataFrame({"A": np.full(30, 100.0)}, index=idx)
        close.iloc[0] = 80.0  # earlier price
        # momentum[T] = close[T] / close[T-21] - 1
        # At row 21: close[21] / close[0] - 1 = 100/80 - 1 = 0.25
        result = momentum(close, 21)
        assert abs(result.iloc[21, 0] - 0.25) < 1e-10

    def test_no_lookahead_flat_prices(self, flat_close) -> None:
        """Flat prices → momentum always 0 (after warmup)."""
        result = momentum(flat_close, 21)
        non_nan = result.dropna()
        assert (non_nan.abs() < 1e-10).all().all()

    def test_invalid_window_raises(self, random_close) -> None:
        with pytest.raises(ValueError, match="window must be > 0"):
            momentum(random_close, 0)

    def test_invalid_input_type_raises(self) -> None:
        with pytest.raises(TypeError):
            momentum("not_a_dataframe", 21)  # type: ignore[arg-type]

    def test_empty_dataframe_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            momentum(pd.DataFrame(), 21)


class TestMomentum21:
    def test_nan_count(self, random_close) -> None:
        result = momentum_21(random_close)
        assert result.iloc[:21].isna().all().all()
        assert result.iloc[21:].notna().all().all()

    def test_trending_positive_for_rising_prices(self, trending_close) -> None:
        result = momentum_21(trending_close)
        # SPY rises linearly, so momentum > 0 after warmup
        spy_mom = result["SPY"].dropna()
        assert (spy_mom > 0).all()

    def test_flat_prices_zero_momentum(self, flat_close) -> None:
        result = momentum_21(flat_close)
        assert result.dropna().abs().max().max() < 1e-10


class TestMomentum63:
    def test_nan_count(self, random_close) -> None:
        result = momentum_63(random_close)
        assert result.iloc[:63].isna().all().all()
        assert result.iloc[63:].notna().all().all()


class TestMomentum126:
    def test_nan_count(self, random_close) -> None:
        result = momentum_126(random_close)
        assert result.iloc[:126].isna().all().all()
        assert result.iloc[126:].notna().all().all()

    def test_126_day_requires_more_warmup_than_21(self, random_close) -> None:
        """126-day requires 126 NaN rows; 21-day only 21."""
        m21 = momentum_21(random_close)
        m126 = momentum_126(random_close)
        assert m21.iloc[:21].isna().all().all()
        assert m21.iloc[21:].notna().all().all()
        assert m126.iloc[:126].isna().all().all()
        assert m126.iloc[126:].notna().all().all()
