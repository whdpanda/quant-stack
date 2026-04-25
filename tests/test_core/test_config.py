"""Tests for the unified config layer."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_stack.core.config import (
    AppConfig,
    BacktestLayerConfig,
    ExecutionLayerConfig,
    PortfolioLayerConfig,
    RiskConfig,
    load_config,
)


class TestRiskConfig:
    def test_defaults_valid(self) -> None:
        r = RiskConfig()
        assert r.max_position_size == 0.40
        assert r.max_drawdown_halt == 0.15
        assert r.daily_loss_limit == 0.03

    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be in"):
            RiskConfig(max_position_size=0.0)

    def test_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be in"):
            RiskConfig(daily_loss_limit=1.0)

    def test_drawdown_must_exceed_daily(self) -> None:
        with pytest.raises(ValidationError, match="max_drawdown_halt"):
            RiskConfig(max_drawdown_halt=0.02, daily_loss_limit=0.03)


class TestPortfolioLayerConfig:
    def test_weight_bounds_invalid(self) -> None:
        with pytest.raises(ValidationError, match="weight_bounds"):
            PortfolioLayerConfig(weight_bounds=(0.5, 0.3))

    def test_efficient_risk_needs_target(self) -> None:
        with pytest.raises(ValidationError, match="target_volatility"):
            PortfolioLayerConfig(method="efficient_risk", target_volatility=None)

    def test_efficient_risk_with_target(self) -> None:
        cfg = PortfolioLayerConfig(method="efficient_risk", target_volatility=0.12)
        assert cfg.target_volatility == 0.12


class TestAppConfig:
    def test_default_construction(self) -> None:
        cfg = AppConfig()
        assert cfg.execution.risk.max_position_size == 0.40
        assert cfg.backtest.initial_cash == 100_000.0

    def test_frozen(self) -> None:
        cfg = AppConfig()
        with pytest.raises(ValidationError):
            cfg.backtest = BacktestLayerConfig(initial_cash=50_000)  # type: ignore[misc]

    def test_to_data_config(self) -> None:
        from datetime import date
        cfg = AppConfig()
        dc = cfg.to_data_config(["SPY", "QQQ"])
        assert dc.symbols == ["SPY", "QQQ"]
        assert dc.start == cfg.data.default_start
        assert dc.end == cfg.data.default_end

    def test_to_backtest_config(self) -> None:
        cfg = AppConfig()
        bc = cfg.to_backtest_config(["SPY"], "sma_cross")
        assert bc.strategy_name == "sma_cross"
        assert bc.commission == cfg.backtest.commission


class TestLoadConfig:
    def test_load_settings_yaml(self) -> None:
        cfg = load_config("config/settings.yaml")
        assert cfg.strategy.lookback_days == 210
        assert cfg.execution.mode == "paper"
        assert cfg.execution.risk.max_drawdown_halt == 0.15

    def test_partial_yaml(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            backtest:
              initial_cash: 200000
        """)
        p = tmp_path / "partial.yaml"
        p.write_text(yaml_content)
        cfg = load_config(p)
        assert cfg.backtest.initial_cash == 200_000
        # Unspecified sections use defaults
        assert cfg.data.fill_limit == 5

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("config/nonexistent.yaml")
