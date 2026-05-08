from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

import experiments.shadow_run as shadow_run


def test_shadow_run_downloads_from_formal_period_start(monkeypatch):
    captured = {}
    index = pd.bdate_range("2010-01-04", periods=shadow_run.MOMENTUM_WINDOW + 1)

    def fake_fetch_stooq_close(symbols, *, start, end, min_rows):
        captured["symbols"] = symbols
        captured["start"] = start
        captured["end"] = end
        captured["min_rows"] = min_rows
        return pd.DataFrame(
            {symbol: range(1, len(index) + 1) for symbol in symbols},
            index=index,
        )

    monkeypatch.setattr(shadow_run, "fetch_stooq_close", fake_fetch_stooq_close)

    close = shadow_run._download_fresh_prices(["IBB", "QQQ"])

    assert captured["start"] == shadow_run.PERIOD_START == date(2010, 1, 1)
    assert captured["min_rows"] == shadow_run.MOMENTUM_WINDOW + 1
    assert close.columns.tolist() == ["IBB", "QQQ"]


def test_shadow_run_history_context_is_not_short_window(monkeypatch):
    captured = {}
    index = pd.bdate_range("2010-01-04", periods=shadow_run.MOMENTUM_WINDOW + 20)

    def fake_fetch_stooq_close(symbols, *, start, end, min_rows):
        del end, min_rows
        captured["calendar_days"] = (date.today() - start).days
        return pd.DataFrame(
            {symbol: range(1, len(index) + 1) for symbol in symbols},
            index=index,
        )

    monkeypatch.setattr(shadow_run, "fetch_stooq_close", fake_fetch_stooq_close)

    shadow_run._download_fresh_prices(["IBB"])

    assert captured["calendar_days"] > 5_000


def test_non_rebalance_day_is_not_execution_ready():
    result = SimpleNamespace(
        needs_rebalance=True,
        is_scheduled_rebalance_day=False,
        executable_count=3,
    )

    assert shadow_run._is_execution_ready_rebalance(result) is False


def test_scheduled_rebalance_day_with_executable_orders_is_execution_ready():
    result = SimpleNamespace(
        needs_rebalance=True,
        is_scheduled_rebalance_day=True,
        executable_count=3,
    )

    assert shadow_run._is_execution_ready_rebalance(result) is True


def test_shadow_run_strategy_parameters_match_formal_strategy():
    assert shadow_run.MOMENTUM_WINDOW == 210
    assert shadow_run.TOP_N == 3
    assert shadow_run.ENTRY_MARGIN == 0.02
    assert shadow_run.VOL_WINDOW == 63
    assert shadow_run.REBALANCE_FREQ == "2ME"
    assert shadow_run.PERIOD_START == date(2010, 1, 1)
