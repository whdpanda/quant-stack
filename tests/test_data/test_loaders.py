"""Tests for CsvDataLoader and ParquetDataLoader."""

from __future__ import annotations

import pytest
import pandas as pd

from quant_stack.data.loaders import CsvDataLoader, ParquetDataLoader
from quant_stack.data.loaders.base import REQUIRED_COLUMNS


class TestCsvDataLoader:
    def test_load_with_symbol_column(self, spy_csv) -> None:
        path, expected = spy_csv
        loader = CsvDataLoader()
        df = loader.load(path)
        assert "symbol" in df.columns
        assert set(df["symbol"].unique()) == {"SPY"}
        assert len(df) == len(expected)

    def test_load_infers_symbol_from_filename(self, qqq_csv) -> None:
        path, _ = qqq_csv
        loader = CsvDataLoader()
        df = loader.load(path)
        assert "symbol" in df.columns
        # Filename is QQQ.csv → symbol should be inferred as "QQQ"
        assert df["symbol"].iloc[0] == "QQQ"

    def test_columns_lowercased(self, spy_csv) -> None:
        path, _ = spy_csv
        df = CsvDataLoader().load(path)
        assert all(c == c.lower() for c in df.columns)

    def test_required_columns_present(self, spy_csv) -> None:
        path, _ = spy_csv
        df = CsvDataLoader().load(path)
        assert REQUIRED_COLUMNS.issubset(set(df.columns))

    def test_file_not_found_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            CsvDataLoader().load(tmp_path / "MISSING.csv")

    def test_polars_not_implemented(self, spy_csv) -> None:
        path, _ = spy_csv
        with pytest.raises(NotImplementedError):
            CsvDataLoader().load_polars(path)

    def test_supports_polars_false(self) -> None:
        assert CsvDataLoader().supports_polars is False


class TestParquetDataLoader:
    def test_load_with_datetimeindex(self, spy_parquet) -> None:
        path, expected = spy_parquet
        loader = ParquetDataLoader()
        df = loader.load(path)
        assert "date" in df.columns
        assert "symbol" in df.columns
        assert len(df) == len(expected)

    def test_infers_symbol_from_filename(self, spy_parquet) -> None:
        path, _ = spy_parquet
        df = ParquetDataLoader().load(path)
        assert df["symbol"].iloc[0] == "SPY"

    def test_file_not_found_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            ParquetDataLoader().load(tmp_path / "MISSING.parquet")

    def test_columns_lowercased(self, spy_parquet) -> None:
        path, _ = spy_parquet
        df = ParquetDataLoader().load(path)
        assert all(c == c.lower() for c in df.columns)
