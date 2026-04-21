"""Tests for DataRepository."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_stack.core.exceptions import DataProviderError
from quant_stack.data.loaders import CANONICAL_COLUMNS
from quant_stack.data.repository import DataRepository


class TestLoad:
    def test_load_single_symbol_csv(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        df = repo.load(["SPY"])
        assert list(df.columns) == CANONICAL_COLUMNS
        assert (df["symbol"] == "SPY").all()

    def test_load_multiple_symbols(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        df = repo.load(["SPY", "QQQ"])
        assert set(df["symbol"].unique()) == {"SPY", "QQQ"}

    def test_load_parquet(self, data_dir_with_parquet) -> None:
        repo = DataRepository(data_dir_with_parquet)
        df = repo.load(["SPY"])
        assert (df["symbol"] == "SPY").all()
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_date_filtering_start(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        df_full = repo.load(["SPY"])
        mid_date = df_full["date"].sort_values().iloc[len(df_full) // 2]
        df_filtered = repo.load(["SPY"], start=mid_date)
        assert (df_filtered["date"] >= mid_date).all()
        assert len(df_filtered) < len(df_full)

    def test_date_filtering_end(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        df_full = repo.load(["SPY"])
        mid_date = df_full["date"].sort_values().iloc[len(df_full) // 2]
        df_filtered = repo.load(["SPY"], end=mid_date)
        assert (df_filtered["date"] <= mid_date).all()

    def test_symbol_not_found_raises(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        with pytest.raises(DataProviderError, match="MISSING"):
            repo.load(["MISSING"])

    def test_symbols_case_insensitive(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        df = repo.load(["spy"])  # lowercase input
        assert (df["symbol"] == "SPY").all()

    def test_output_sorted(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        df = repo.load(["SPY", "QQQ"])
        for sym, group in df.groupby("symbol"):
            dates = group["date"].reset_index(drop=True)
            assert (dates.diff().dropna() >= pd.Timedelta(0)).all(), (
                f"Dates not sorted for {sym}"
            )

    def test_load_from_combined_csv(self, combined_csv) -> None:
        import os
        path, _ = combined_csv
        data_dir = os.path.dirname(path)
        repo = DataRepository(data_dir)
        df = repo.load(["SPY"])
        assert (df["symbol"] == "SPY").all()


class TestLoadClose:
    def test_returns_wide_format(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        wide = repo.load_close(["SPY", "QQQ"])
        assert isinstance(wide, pd.DataFrame)
        assert set(wide.columns) == {"SPY", "QQQ"}
        assert isinstance(wide.index, pd.DatetimeIndex)

    def test_no_nans_in_wide_format(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        wide = repo.load_close(["SPY", "QQQ"])
        # Both symbols have the same date range in fixtures, so no NaNs expected
        assert wide.notna().all().all()

    def test_price_column_adj_close_default(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        tidy = repo.load(["SPY"])
        wide = repo.load_close(["SPY"])
        # adj_close should be used; since source has no adj_close, it equals close
        pd.testing.assert_series_equal(
            wide["SPY"].reset_index(drop=True),
            tidy["adj_close"].reset_index(drop=True),
            check_names=False,
        )

    def test_date_range_filtering(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        wide_full = repo.load_close(["SPY"])
        start_date = wide_full.index[5]
        wide_filtered = repo.load_close(["SPY"], start=start_date)
        assert wide_filtered.index[0] >= start_date


class TestAvailableSymbols:
    def test_lists_csv_symbols(self, data_dir_with_files) -> None:
        repo = DataRepository(data_dir_with_files)
        symbols = repo.available_symbols()
        assert "SPY" in symbols
        assert "QQQ" in symbols

    def test_lists_parquet_symbols(self, data_dir_with_parquet) -> None:
        repo = DataRepository(data_dir_with_parquet)
        symbols = repo.available_symbols()
        assert "SPY" in symbols
        assert "QQQ" in symbols

    def test_empty_dir_returns_empty(self, tmp_path) -> None:
        repo = DataRepository(tmp_path)
        assert repo.available_symbols() == []
