"""Tests for DataValidator — one test class per validation step."""

from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
import pytest

from quant_stack.data.loaders import CsvDataLoader
from quant_stack.data.validation import DataValidationError, DataValidator, ValidationConfig
from tests.test_data.conftest import make_ohlcv


def raw_df(symbol: str = "SPY", periods: int = 10) -> pd.DataFrame:
    """Helper: raw tidy DataFrame (string dates, no adj_close)."""
    return make_ohlcv(symbol, periods=periods)


class TestCheckRequiredColumns:
    def test_all_present_passes(self) -> None:
        df = DataValidator.check_required_columns(raw_df())
        assert df is not None

    def test_missing_column_raises(self) -> None:
        df = raw_df().drop(columns=["volume"])
        with pytest.raises(DataValidationError, match="volume"):
            DataValidator.check_required_columns(df)

    def test_error_lists_missing_columns(self) -> None:
        df = raw_df().drop(columns=["open", "close"])
        with pytest.raises(DataValidationError) as exc_info:
            DataValidator.check_required_columns(df)
        assert "open" in str(exc_info.value) or "close" in str(exc_info.value)


class TestCoerceTypes:
    def test_date_becomes_datetime(self) -> None:
        df = DataValidator.coerce_types(raw_df())
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_symbol_uppercased(self) -> None:
        df = raw_df()
        df["symbol"] = "spy"
        result = DataValidator.coerce_types(df)
        assert (result["symbol"] == "SPY").all()

    def test_numeric_cols_float64(self) -> None:
        df = DataValidator.coerce_types(raw_df())
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == "float64", f"{col} should be float64"

    def test_non_numeric_coerced_to_nan(self) -> None:
        df = raw_df()
        df.loc[0, "close"] = "bad_value"
        result = DataValidator.coerce_types(df)
        assert pd.isna(result.loc[0, "close"])


class TestAddAdjClose:
    def test_adds_adj_close_equal_to_close(self) -> None:
        df = DataValidator.coerce_types(raw_df())
        result = DataValidator.add_adj_close(df)
        assert "adj_close" in result.columns
        pd.testing.assert_series_equal(
            result["adj_close"].reset_index(drop=True),
            result["close"].reset_index(drop=True),
            check_names=False,
        )

    def test_existing_adj_close_preserved(self) -> None:
        df = DataValidator.coerce_types(raw_df())
        df["adj_close"] = df["close"] * 0.95
        original = df["adj_close"].copy()
        result = DataValidator.add_adj_close(df)
        pd.testing.assert_series_equal(result["adj_close"], original)


class TestSort:
    def test_sorted_by_symbol_then_date(self) -> None:
        spy = make_ohlcv("SPY", periods=5)
        qqq = make_ohlcv("QQQ", periods=5)
        df = pd.concat([qqq, spy], ignore_index=True)  # QQQ first
        result = DataValidator.sort(df)
        symbols = result["symbol"].tolist()
        assert symbols[:5] == ["QQQ"] * 5
        assert symbols[5:] == ["SPY"] * 5

    def test_dates_ascending_within_symbol(self) -> None:
        df = DataValidator.sort(make_ohlcv("SPY", periods=10))
        dates = pd.to_datetime(df["date"])
        assert (dates.diff().dropna() >= pd.Timedelta(0)).all()


class TestHandleDuplicates:
    def test_exact_duplicates_dropped(self) -> None:
        df = make_ohlcv("SPY", periods=5)
        df_with_dups = pd.concat([df, df.iloc[:2]], ignore_index=True)
        result = DataValidator.handle_duplicates(df_with_dups)
        assert len(result) == 5

    def test_key_conflict_raises(self) -> None:
        df = make_ohlcv("SPY", periods=5)
        conflict = df.iloc[:1].copy()
        conflict["close"] = 999.0  # same (date, symbol), different price
        df_conflict = pd.concat([df, conflict], ignore_index=True)
        with pytest.raises(DataValidationError, match="Conflicting rows"):
            DataValidator.handle_duplicates(df_conflict)


class TestFillMissing:
    def test_ffill_fills_gap(self) -> None:
        df = DataValidator.coerce_types(make_ohlcv("SPY", periods=10))
        df.loc[3, "close"] = float("nan")
        result = DataValidator.fill_missing(df, method="ffill", limit=5)
        assert not result["close"].isna().any()

    def test_fill_none_leaves_nans(self) -> None:
        df = DataValidator.coerce_types(make_ohlcv("SPY", periods=5))
        df.loc[2, "close"] = float("nan")
        result = DataValidator.fill_missing(df, method="none")
        assert result["close"].isna().sum() == 1

    def test_fill_respects_limit(self) -> None:
        df = DataValidator.coerce_types(make_ohlcv("SPY", periods=15))
        df.loc[5:11, "close"] = float("nan")  # 7 consecutive NaNs
        result = DataValidator.fill_missing(df, method="ffill", limit=3)
        # Rows 5-7 should be filled; rows 8-11 should remain NaN
        assert result.loc[5, "close"] != float("nan")  # filled
        assert pd.isna(result.loc[9, "close"])  # beyond limit


class TestOHLCVConsistency:
    def test_valid_data_no_warning(self) -> None:
        df = DataValidator.coerce_types(make_ohlcv("SPY", periods=20))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DataValidator.check_ohlcv_consistency(df, action="warn")

    def test_high_lt_low_warns(self) -> None:
        df = DataValidator.coerce_types(make_ohlcv("SPY", periods=5))
        df.loc[0, "high"] = df.loc[0, "low"] - 1.0
        with pytest.warns(UserWarning, match="high < low"):
            DataValidator.check_ohlcv_consistency(df, action="warn")

    def test_high_lt_low_raises(self) -> None:
        df = DataValidator.coerce_types(make_ohlcv("SPY", periods=5))
        df.loc[0, "high"] = df.loc[0, "low"] - 1.0
        with pytest.raises(DataValidationError, match="high < low"):
            DataValidator.check_ohlcv_consistency(df, action="raise")

    def test_negative_volume_warns(self) -> None:
        df = DataValidator.coerce_types(make_ohlcv("SPY", periods=5))
        df.loc[0, "volume"] = -1.0
        with pytest.warns(UserWarning, match="volume < 0"):
            DataValidator.check_ohlcv_consistency(df, action="warn")


class TestFullPipeline:
    def test_full_pipeline_returns_canonical_columns(self) -> None:
        from quant_stack.data.loaders.base import CANONICAL_COLUMNS
        df = DataValidator().validate(raw_df())
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_full_pipeline_dtypes(self) -> None:
        df = DataValidator().validate(raw_df())
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["symbol"].dtype == object
        assert df["close"].dtype == "float64"

    def test_full_pipeline_sorted(self) -> None:
        spy = make_ohlcv("SPY", periods=5)
        qqq = make_ohlcv("QQQ", periods=5)
        mixed = pd.concat([qqq, spy], ignore_index=True)
        result = DataValidator().validate(mixed)
        assert result["symbol"].tolist()[:5] == ["QQQ"] * 5
