"""Tests for momentum signal generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_stack.core.schemas import SignalSource
from quant_stack.signals.base import SignalFrame
from quant_stack.signals.momentum import (
    absolute_momentum_signal,
    relative_momentum_ranking_signal,
)


class TestSignalFrameBase:
    """Tests for the SignalFrame container itself."""

    def test_repr(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum)
        assert "absolute_momentum" in repr(sf)
        assert "SPY" in repr(sf)

    def test_source_is_research(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum)
        assert sf.source == SignalSource.RESEARCH

    def test_shape_mismatch_raises(self, sample_momentum) -> None:
        with pytest.raises(ValueError, match="same shape"):
            SignalFrame(
                signals=sample_momentum,
                strength=sample_momentum.iloc[:, :1],  # different shape
                strategy_name="test",
            )

    def test_to_long_df_schema(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum)
        long_df = sf.to_long_df()
        assert "date" in long_df.columns
        assert "symbol" in long_df.columns
        assert "direction" in long_df.columns
        assert "strength" in long_df.columns
        assert "source" in long_df.columns

    def test_to_long_df_no_nan_rows(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum)
        long_df = sf.to_long_df()
        assert not long_df["strength"].isna().any()

    def test_latest_returns_signal_objects(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum)
        signals = sf.latest()
        assert len(signals) > 0
        from quant_stack.core.schemas import Signal
        assert all(isinstance(s, Signal) for s in signals)

    def test_latest_source_is_research(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum)
        for sig in sf.latest():
            assert sig.source == SignalSource.RESEARCH


class TestAbsoluteMomentumSignal:
    def test_shape_preserved(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum)
        assert sf.signals.shape == sample_momentum.shape

    def test_binary_values_only(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum)
        non_nan = sf.signals.dropna()
        assert set(non_nan.values.ravel().astype(float).tolist()).issubset({0.0, 1.0})

    def test_nan_rows_preserved_from_factor(self, sample_momentum) -> None:
        """Signal must be NaN where factor is NaN — no default fill."""
        sf = absolute_momentum_signal(sample_momentum)
        factor_nan = sample_momentum.isna()
        signal_nan = sf.signals.isna()
        # Wherever factor is NaN, signal must also be NaN
        assert (signal_nan[factor_nan]).all().all()

    def test_positive_threshold(self, sample_momentum) -> None:
        sf = absolute_momentum_signal(sample_momentum, threshold=0.05)
        # Where signal=1, momentum must be > 0.05 (checked per non-NaN cell)
        long_cells = sf.signals.stack()  # drops NaN, gives (date, symbol) Series
        long_dates_syms = long_cells[long_cells == 1.0].index
        for date, sym in long_dates_syms:
            assert sample_momentum.loc[date, sym] > 0.05

    def test_no_signal_when_all_below_threshold(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        # Prices declining → all momentum negative
        close_vals = 100.0 - 0.1 * np.arange(100)
        mom = pd.DataFrame({"A": close_vals}, index=idx).pct_change(21)
        sf = absolute_momentum_signal(mom, threshold=0.0)
        non_nan = sf.signals.dropna()
        assert (non_nan == 0.0).all().all()


class TestRelativeMomentumRankingSignal:
    def test_shape_preserved(self, sample_momentum) -> None:
        sf = relative_momentum_ranking_signal(sample_momentum, top_n=2)
        assert sf.signals.shape == sample_momentum.shape

    def test_ranks_populated(self, sample_momentum) -> None:
        sf = relative_momentum_ranking_signal(sample_momentum, top_n=2)
        assert sf.ranks is not None
        assert sf.ranks.shape == sample_momentum.shape

    def test_exactly_top_n_long_per_date(self, sample_momentum) -> None:
        top_n = 2
        sf = relative_momentum_ranking_signal(sample_momentum, top_n=top_n)
        # For each row with sufficient data, exactly top_n symbols are long
        valid_rows = sf.signals.dropna(how="any")
        long_count = valid_rows.sum(axis=1)
        assert (long_count == top_n).all(), (
            f"Expected {top_n} longs per date, got:\n{long_count.value_counts()}"
        )

    def test_strength_positive_for_long(self, sample_momentum) -> None:
        sf = relative_momentum_ranking_signal(sample_momentum, top_n=2)
        # stack() drops NaN cells — only non-NaN (date, symbol) pairs remain
        long_cells = sf.signals.stack()
        strength_cells = sf.strength.stack()
        long_strength = strength_cells[long_cells == 1.0]
        assert (long_strength > 0).all()

    def test_strength_zero_for_flat(self, sample_momentum) -> None:
        sf = relative_momentum_ranking_signal(sample_momentum, top_n=2)
        flat_cells = sf.signals.stack()
        strength_cells = sf.strength.stack()
        flat_strength = strength_cells[flat_cells == 0.0]
        assert (flat_strength == 0.0).all()

    def test_rank_1_has_highest_strength(self, sample_momentum) -> None:
        sf = relative_momentum_ranking_signal(sample_momentum, top_n=2)
        valid = sf.ranks.dropna(how="any")
        for date in valid.index[:10]:  # spot-check 10 dates
            rank_row = sf.ranks.loc[date]
            strength_row = sf.strength.loc[date]
            rank1_sym = rank_row.idxmin()
            rank2_sym = rank_row.where(rank_row > 1).idxmin()
            assert strength_row[rank1_sym] >= strength_row[rank2_sym]

    def test_eligibility_filter_applied(self, sample_momentum) -> None:
        # Mark only SPY and QQQ as eligible (not IEF)
        eligible = pd.DataFrame(
            True,
            index=sample_momentum.index,
            columns=sample_momentum.columns,
        )
        eligible["IEF"] = False
        sf = relative_momentum_ranking_signal(sample_momentum, top_n=2, eligible=eligible)
        # IEF should never be long
        assert (sf.signals["IEF"].dropna() == 0.0).all()

    def test_top_n_invalid_raises(self, sample_momentum) -> None:
        with pytest.raises(ValueError, match="top_n must be > 0"):
            relative_momentum_ranking_signal(sample_momentum, top_n=0)

    def test_to_long_df_has_rank_column(self, sample_momentum) -> None:
        sf = relative_momentum_ranking_signal(sample_momentum, top_n=2)
        long_df = sf.to_long_df()
        assert "rank" in long_df.columns
