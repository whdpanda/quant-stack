from __future__ import annotations

from datetime import date
from io import BytesIO

import pandas as pd
import pytest

from quant_stack.core.exceptions import DataProviderError
from quant_stack.data import stooq_eod
from quant_stack.research.strategies.sector_momentum import (
    HysteresisMode,
    RISK_ON_UNIVERSE,
    SectorMomentumStrategy,
    WeightingScheme,
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._body = BytesIO(text.encode("utf-8"))

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.read()


def test_stooq_symbol_mapping_uses_us_suffix() -> None:
    assert stooq_eod._to_stooq_symbol("XLF") == "xlf.us"
    assert stooq_eod._to_stooq_symbol("XLK") == "xlk.us"
    assert stooq_eod._to_stooq_symbol("QQQ") == "qqq.us"


def test_fetch_stooq_close_preserves_original_ticker_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "xlf.us": "Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,10.5,100\n",
        "qqq.us": "Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,20.5,100\n",
    }

    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        del timeout
        symbol = url.split("s=", 1)[1].split("&", 1)[0]
        return _FakeResponse(responses[symbol])

    monkeypatch.setattr(stooq_eod, "urlopen", fake_urlopen)

    close = stooq_eod.fetch_stooq_close(
        ["XLF", "QQQ"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        max_retries=0,
    )

    assert close.columns.tolist() == ["XLF", "QQQ"]
    assert close.iloc[0].to_dict() == {"XLF": 10.5, "QQQ": 20.5}


def test_fetch_stooq_close_drops_small_alignment_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "xlf.us": (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,1,1,1,10.5,100\n"
            "2024-01-03,1,1,1,10.6,100\n"
            "2024-01-04,1,1,1,10.7,100\n"
        ),
        "qqq.us": (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,1,1,1,20.5,100\n"
            "2024-01-03,1,1,1,20.6,100\n"
        ),
    }

    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        del timeout
        symbol = url.split("s=", 1)[1].split("&", 1)[0]
        return _FakeResponse(responses[symbol])

    monkeypatch.setattr(stooq_eod, "urlopen", fake_urlopen)

    close = stooq_eod.fetch_stooq_close(
        ["XLF", "QQQ"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        max_retries=0,
    )

    assert close.columns.tolist() == ["XLF", "QQQ"]
    assert close.index.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert not close.isna().any().any()


def test_fetch_stooq_close_fails_on_large_alignment_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "xlf.us": (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,1,1,1,10.5,100\n"
            "2024-01-03,1,1,1,10.6,100\n"
            "2024-01-04,1,1,1,10.7,100\n"
            "2024-01-05,1,1,1,10.8,100\n"
        ),
        "qqq.us": "Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,20.5,100\n",
    }

    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        del timeout
        symbol = url.split("s=", 1)[1].split("&", 1)[0]
        return _FakeResponse(responses[symbol])

    monkeypatch.setattr(stooq_eod, "urlopen", fake_urlopen)

    with pytest.raises(DataProviderError, match="QQQ missing=3"):
        stooq_eod.fetch_stooq_close(
            ["XLF", "QQQ"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            max_retries=0,
            max_missing_ratio=0.01,
            max_missing_days=1,
        )


def test_fetch_stooq_close_fails_when_common_rows_are_too_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        del url, timeout
        return _FakeResponse("Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,10.5,100\n")

    monkeypatch.setattr(stooq_eod, "urlopen", fake_urlopen)

    with pytest.raises(DataProviderError, match="required at least 2"):
        stooq_eod.fetch_stooq_close(
            ["XLF"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            max_retries=0,
            min_rows=2,
        )


def test_fetch_stooq_close_does_not_write_yahoo_ohlcv_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fail_to_parquet(self: pd.DataFrame, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise AssertionError("Stooq close helper must not write parquet cache")

    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        del url, timeout
        return _FakeResponse(
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,1,2,0.5,10.5,100\n"
            "2024-01-03,2,3,1.5,10.6,200\n"
        )

    monkeypatch.setattr(stooq_eod, "urlopen", fake_urlopen)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)
    monkeypatch.chdir(tmp_path)

    close = stooq_eod.fetch_stooq_close(
        ["XLF"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        max_retries=0,
    )

    assert close.columns.tolist() == ["XLF"]
    assert close.index.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert close["XLF"].tolist() == [10.5, 10.6]
    assert not {"open", "high", "low", "close", "volume"}.intersection(close.columns)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "data" / "XLF.parquet").exists()


def test_fetch_stooq_close_fails_when_any_ticker_download_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        del timeout
        if "qqq.us" in url:
            raise TimeoutError("timed out")
        return _FakeResponse("Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,10.5,100\n")

    monkeypatch.setattr(stooq_eod, "urlopen", fake_urlopen)
    monkeypatch.setattr(stooq_eod, "sleep", lambda seconds: None)

    with pytest.raises(DataProviderError, match="QQQ"):
        stooq_eod.fetch_stooq_close(
            ["XLF", "QQQ"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            max_retries=0,
        )


@pytest.mark.parametrize(
    "body, message",
    [
        ("Date,Open,High,Low,Close,Volume\n", "no usable close data"),
        ("Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,,100\n", "NaN"),
        ("Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,0,100\n", "non-positive"),
        (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,1,1,1,10.5,100\n"
            "2024-01-02,1,1,1,10.6,100\n",
            "duplicate dates",
        ),
    ],
)
def test_fetch_stooq_close_validates_bad_close_data(
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    message: str,
) -> None:
    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        del url, timeout
        return _FakeResponse(body)

    monkeypatch.setattr(stooq_eod, "urlopen", fake_urlopen)

    with pytest.raises(DataProviderError, match=message):
        stooq_eod.fetch_stooq_close(
            ["XLF"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            max_retries=0,
        )


def test_sector_momentum_strategy_parameters_are_unchanged() -> None:
    assert RISK_ON_UNIVERSE == ["IBB", "QQQ", "XLE", "XLV", "XLF", "XLI", "VTV", "GDX", "XLP"]
    strategy = SectorMomentumStrategy(momentum_window=210, top_n=3)
    assert strategy.momentum_window == 210
    assert strategy.top_n == 3
    assert HysteresisMode.ENTRY_MARGIN.value == "entry_margin"
    assert WeightingScheme.BLEND_70_30.value == "blend_70_30"
