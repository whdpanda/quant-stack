"""Tests for trend filter signal generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_stack.core.schemas import SignalSource
from quant_stack.factors.trend import sma_200
from quant_stack.signals.trend import as_eligibility_mask, trend_filter_signal


class TestTrendFilterSignal:
    def test_shape_preserved(self, sample_close, sample_sma) -> None:
        sf = trend_filter_signal(sample_close, sample_sma)
        assert sf.signals.shape == sample_close.shape

    def test_source_is_research(self, sample_close, sample_sma) -> None:
        sf = trend_filter_signal(sample_close, sample_sma)
        assert sf.source == SignalSource.RESEARCH

    def test_binary_values(self, sample_close, sample_sma) -> None:
        sf = trend_filter_signal(sample_close, sample_sma)
        non_nan = sf.signals.dropna()
        assert set(non_nan.values.ravel().tolist()).issubset({0.0, 1.0})

    def test_nan_where_sma_nan(self, sample_close, sample_sma) -> None:
        sf = trend_filter_signal(sample_close, sample_sma)
        sma_nan = sample_sma.isna()
        signal_nan = sf.signals.isna()
        assert (signal_nan[sma_nan]).all().all()

    def test_long_when_above_sma(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        close = pd.DataFrame({"A": [110.0, 110.0, 110.0, 110.0, 110.0]}, index=idx)
        sma_val = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0, 100.0]}, index=idx)
        sf = trend_filter_signal(close, sma_val)
        assert (sf.signals == 1.0).all().all()

    def test_flat_when_below_sma(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        close = pd.DataFrame({"A": [90.0, 90.0, 90.0, 90.0, 90.0]}, index=idx)
        sma_val = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0, 100.0]}, index=idx)
        sf = trend_filter_signal(close, sma_val)
        assert (sf.signals == 0.0).all().all()

    def test_shape_mismatch_raises(self, sample_close, sample_sma) -> None:
        with pytest.raises(ValueError, match="shape"):
            trend_filter_signal(sample_close.iloc[:, :1], sample_sma)

    def test_column_mismatch_raises(self, sample_close, sample_sma) -> None:
        renamed_sma = sample_sma.rename(columns={"SPY": "SPYY"})
        with pytest.raises(ValueError, match="identical columns"):
            trend_filter_signal(sample_close, renamed_sma)


class TestAsEligibilityMask:
    def test_returns_bool_dataframe(self, sample_close, sample_sma) -> None:
        sf = trend_filter_signal(sample_close, sample_sma)
        mask = as_eligibility_mask(sf)
        assert mask.dtypes.eq(bool).all()

    def test_nan_treated_as_false(self, sample_close, sample_sma) -> None:
        sf = trend_filter_signal(sample_close, sample_sma)
        mask = as_eligibility_mask(sf)
        # Where SMA is NaN (first 199 rows), signal is NaN, mask should be False
        sma_nan_rows = sample_sma.isna().any(axis=1)
        assert not mask[sma_nan_rows].any().any()

    def test_valid_rows_boolean(self, sample_close, sample_sma) -> None:
        sf = trend_filter_signal(sample_close, sample_sma)
        mask = as_eligibility_mask(sf)
        valid = mask[~sample_sma.isna().any(axis=1)]
        assert valid.dtypes.eq(bool).all()


class TestTrendPlusMomentumIntegration:
    """Integration test: trend filter as eligibility for momentum ranking."""

    def test_trend_filters_momentum_universe(self, sample_close) -> None:
        from quant_stack.factors import momentum_63, sma_200
        from quant_stack.signals.momentum import relative_momentum_ranking_signal

        mom = momentum_63(sample_close)
        sma = sma_200(sample_close)
        trend = trend_filter_signal(sample_close, sma)
        eligible = as_eligibility_mask(trend)

        sf = relative_momentum_ranking_signal(mom, top_n=2, eligible=eligible)

        # No symbol should be long when it was not eligible
        long_mask = sf.signals == 1.0
        # On rows where eligibility data exists
        common_idx = eligible.dropna(how="any").index.intersection(
            sf.signals.dropna(how="any").index
        )
        for sym in sample_close.columns:
            not_eligible_dates = common_idx[~eligible.loc[common_idx, sym]]
            if len(not_eligible_dates) > 0:
                # Symbol should not be long on dates when it's ineligible
                assert (sf.signals.loc[not_eligible_dates, sym] == 0.0).all(), (
                    f"{sym} was long on ineligible dates"
                )
