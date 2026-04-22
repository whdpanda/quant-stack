"""Shared fixtures for tracking tests."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from quant_stack.core.schemas import (
    BacktestResult,
    ExperimentRecord,
    PortfolioWeights,
)


@pytest.fixture
def minimal_result() -> BacktestResult:
    return BacktestResult(
        strategy_name="test_strategy",
        symbols=["SPY", "QQQ"],
        period_start=date(2020, 1, 2),
        period_end=date(2023, 12, 29),
        total_return=0.42,
        cagr=0.10,
        sharpe_ratio=1.25,
        max_drawdown=0.15,
        n_trades=48,
        commission_paid=1200.0,
        benchmark_return=0.12,
        sortino_ratio=1.80,
        annual_volatility=0.12,
        annual_turnover=2.40,
    )


@pytest.fixture
def minimal_weights() -> PortfolioWeights:
    return PortfolioWeights(
        weights={"SPY": 0.60, "QQQ": 0.40},
        method="equal_weight",
        rebalance_date=date(2023, 12, 29),
    )


@pytest.fixture
def full_record(minimal_result, minimal_weights) -> ExperimentRecord:
    return ExperimentRecord(
        description="Unit test experiment",
        strategy_params={"top_n": 2, "window": 63},
        symbols=["SPY", "QQQ"],
        period_start=date(2020, 1, 2),
        period_end=date(2023, 12, 29),
        backtest_result=minimal_result,
        portfolio_weights=minimal_weights,
        tags=["test", "momentum"],
        notes="This is a test note.",
        artifact_paths={"weights_csv": "artifacts/weights.csv"},
    )
