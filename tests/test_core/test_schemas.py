"""Unit tests for core schemas."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from quant_stack.core.schemas import (
    BarData,
    BarFreq,
    BacktestConfig,
    BacktestResult,
    DataConfig,
    ExperimentRecord,
    PortfolioWeights,
    Signal,
    SignalDirection,
    SignalSource,
)


class TestDataConfig:
    def test_valid(self) -> None:
        cfg = DataConfig(symbols=["SPY"], start=date(2020, 1, 1), end=date(2023, 1, 1))
        assert cfg.symbols == ["SPY"]

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValidationError, match="end date must be after"):
            DataConfig(symbols=["SPY"], start=date(2023, 1, 1), end=date(2020, 1, 1))

    def test_empty_symbols_raises(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(symbols=[], start=date(2020, 1, 1), end=date(2023, 1, 1))


class TestBarData:
    _valid = dict(
        symbol="SPY",
        timestamp=datetime(2024, 1, 2, 16, 0),
        open=470.0,
        high=472.5,
        low=469.0,
        close=471.0,
        volume=1_000_000.0,
    )

    def test_valid(self) -> None:
        bar = BarData(**self._valid)
        assert bar.symbol == "SPY"
        assert bar.effective_close == 471.0

    def test_adj_close_effective(self) -> None:
        bar = BarData(**{**self._valid, "adj_close": 468.0})
        assert bar.effective_close == 468.0

    def test_high_lt_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= low"):
            BarData(**{**self._valid, "high": 465.0})

    def test_high_lt_close_raises(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= close"):
            BarData(**{**self._valid, "high": 470.5, "close": 471.0})

    def test_frozen(self) -> None:
        bar = BarData(**self._valid)
        with pytest.raises(ValidationError):
            bar.close = 999.0  # type: ignore[misc]

    def test_default_freq(self) -> None:
        bar = BarData(**self._valid)
        assert bar.freq == BarFreq.DAILY


class TestSignal:
    _valid = dict(
        symbol="SPY",
        timestamp=datetime(2024, 1, 2),
        direction=SignalDirection.LONG,
        strategy_name="sma_cross",
        source=SignalSource.RESEARCH,
    )

    def test_valid(self) -> None:
        sig = Signal(**self._valid)
        assert sig.strength == 1.0
        assert sig.source == SignalSource.RESEARCH

    def test_strength_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Signal(**{**self._valid, "strength": 1.5})
        with pytest.raises(ValidationError):
            Signal(**{**self._valid, "strength": -0.1})

    def test_frozen(self) -> None:
        sig = Signal(**self._valid)
        with pytest.raises(ValidationError):
            sig.strength = 0.5  # type: ignore[misc]


class TestPortfolioWeights:
    def test_valid(self) -> None:
        w = PortfolioWeights(weights={"SPY": 0.6, "QQQ": 0.4})
        assert abs(sum(w.weights.values()) - 1.0) < 0.01

    def test_weights_not_summing_raises(self) -> None:
        with pytest.raises(ValidationError, match="sum to"):
            PortfolioWeights(weights={"SPY": 0.3, "QQQ": 0.3})

    def test_negative_weights_raises(self) -> None:
        with pytest.raises(ValidationError, match="negative weights"):
            PortfolioWeights(weights={"SPY": 1.1, "QQQ": -0.1})


class TestBacktestResult:
    def test_excess_return(self) -> None:
        r = BacktestResult(
            strategy_name="test",
            total_return=0.15,
            cagr=0.10,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            n_trades=50,
            benchmark_return=0.12,
        )
        assert abs(r.excess_return - 0.03) < 1e-9

    def test_excess_return_none_without_benchmark(self) -> None:
        r = BacktestResult(
            strategy_name="test",
            total_return=0.15,
            cagr=0.10,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            n_trades=50,
        )
        assert r.excess_return is None


class TestExperimentRecord:
    def _make_record(self) -> ExperimentRecord:
        return ExperimentRecord(
            description="SMA Cross SPY/QQQ",
            symbols=["SPY", "QQQ"],
            period_start=date(2020, 1, 1),
            period_end=date(2023, 12, 31),
            tags=["momentum", "equities"],
        )

    def test_auto_id_and_timestamp(self) -> None:
        r = self._make_record()
        assert len(r.experiment_id) == 36  # UUID4 format
        assert r.created_at is not None

    def test_save_and_load(self, tmp_path) -> None:
        r = self._make_record()
        path = tmp_path / "exp.json"
        r.save(path)
        loaded = ExperimentRecord.load(path)
        assert loaded.experiment_id == r.experiment_id
        assert loaded.symbols == ["SPY", "QQQ"]
        assert loaded.period_start == date(2020, 1, 1)

    def test_config_snapshot_roundtrip(self, tmp_path) -> None:
        from quant_stack.core.config import load_config
        cfg = load_config("config/settings.yaml")
        r = ExperimentRecord(
            symbols=["SPY"],
            config_snapshot=cfg.model_dump(mode="json"),
        )
        path = tmp_path / "exp_cfg.json"
        r.save(path)
        loaded = ExperimentRecord.load(path)
        assert loaded.config_snapshot["backtest"]["initial_cash"] == 100_000
