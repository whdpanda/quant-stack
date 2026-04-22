"""Shared fixtures for test_agent."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from quant_stack.core.schemas import BacktestResult, ExperimentRecord, PortfolioWeights


@pytest.fixture()
def minimal_result() -> BacktestResult:
    return BacktestResult(
        strategy_name="test_strategy",
        total_return=0.20,
        cagr=0.10,
        sharpe_ratio=1.00,
        max_drawdown=0.08,
        n_trades=5,
    )


@pytest.fixture()
def full_record(minimal_result) -> ExperimentRecord:
    return ExperimentRecord(
        description="unit-test experiment",
        strategy_params={"top_n": 2, "window": 63},
        symbols=["SPY", "QQQ"],
        period_start=date(2020, 1, 1),
        period_end=date(2023, 1, 1),
        backtest_result=minimal_result,
        portfolio_weights=PortfolioWeights(
            weights={"SPY": 0.60, "QQQ": 0.40},
            expected_return=0.10,
            expected_volatility=0.15,
            sharpe_ratio=1.0,
        ),
        tags=["test", "unit"],
        notes="synthetic data only",
    )
