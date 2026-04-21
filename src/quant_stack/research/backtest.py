"""Thin vectorbt wrapper for running backtests from BacktestConfig.

vectorbt is an optional dependency; import errors surface a clear message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from quant_stack.core.exceptions import BacktestError
from quant_stack.core.schemas import BacktestConfig, BacktestResult
from quant_stack.research.base import Strategy

if TYPE_CHECKING:
    pass


def run_backtest(
    strategy: Strategy,
    close: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a vectorbt portfolio simulation.

    Args:
        strategy: Instantiated Strategy that produces entry signals.
        close: Adjusted close prices (DatetimeIndex, columns = symbols).
        config: BacktestConfig with cash and frequency settings.

    Returns:
        BacktestResult with key performance metrics.
    """
    try:
        import vectorbt as vbt  # noqa: F401
    except ImportError as e:
        raise BacktestError(
            "vectorbt is not installed: pip install 'quant-stack[research]'"
        ) from e

    logger.info(f"Running backtest: {strategy!r} on {list(close.columns)}")

    signals = strategy.generate_signals(close)
    entries = signals.astype(bool)
    exits = ~entries

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=config.initial_cash,
        fees=config.commission,
        slippage=config.slippage,
        freq=config.freq,
    )

    stats = portfolio.stats()
    logger.info(f"Backtest complete. Sharpe: {stats.get('Sharpe Ratio', float('nan')):.3f}")

    period_start = config.data.start if config.data else None
    period_end = config.data.end if config.data else None
    symbols = list(close.columns) if hasattr(close, "columns") else []

    return BacktestResult(
        strategy_name=strategy.name,
        symbols=symbols,
        period_start=period_start,
        period_end=period_end,
        total_return=float(stats.get("Total Return [%]", 0.0)) / 100,
        sharpe_ratio=float(stats.get("Sharpe Ratio", float("nan"))),
        max_drawdown=float(stats.get("Max Drawdown [%]", 0.0)) / 100,
        cagr=float(stats.get("Annualized Return [%]", 0.0)) / 100,
        n_trades=int(stats.get("Total Trades", 0)),
        metadata={"vbt_stats": stats.to_dict()},
    )
