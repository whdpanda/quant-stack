"""Thin vectorbt wrapper for running backtests from BacktestConfig.

vectorbt is an optional dependency; import errors surface a clear message.

Multi-asset behaviour
---------------------
``run_backtest`` converts binary strategy signals to equal-weight target
positions so that all simultaneously-active assets share the portfolio cash
equally.  Internally it uses ``Portfolio.from_orders`` with
``size_type="targetpercent"``, ``group_by=True``, and ``cash_sharing=True``
— the same approach used by ``vbt_adapter.run_vbt_backtest``.

Single-asset strategies are unaffected by this change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from quant_stack.core.exceptions import BacktestError
from quant_stack.core.schemas import BacktestConfig, BacktestResult
from quant_stack.research.base import Strategy


def run_backtest(
    strategy: Strategy,
    close: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a vectorbt portfolio simulation.

    Args:
        strategy: Instantiated Strategy that produces entry signals
                  (float 0.0 / 1.0, shape matches ``close``).
        close: Adjusted close prices (DatetimeIndex, columns = symbols).
        config: BacktestConfig with cash and frequency settings.

    Returns:
        BacktestResult with key performance metrics.
    """
    try:
        import vectorbt as vbt
    except ImportError as e:
        raise BacktestError(
            "vectorbt is not installed: pip install 'quant-stack[research]'"
        ) from e

    logger.info(f"Running backtest: {strategy!r} on {list(close.columns)}")

    signals = strategy.generate_signals(close)

    # Convert binary signals to equal-weight target fractions.
    # Each date, active assets share the invested budget equally.
    # Rows where no asset is active produce a 0-weight row (hold cash).
    active_count = signals.sum(axis=1).replace(0.0, np.nan)
    target_weights = signals.div(active_count, axis=0).fillna(0.0)

    # Run as a grouped portfolio with shared cash — avoids the silent
    # per-column mean-aggregation that from_signals produces without groupby.
    portfolio = vbt.Portfolio.from_orders(
        close=close,
        size=target_weights,
        size_type="targetpercent",
        fees=config.commission,
        slippage=config.slippage,
        init_cash=config.initial_cash,
        cash_sharing=True,
        group_by=True,
        freq=config.freq,
    )

    stats = portfolio.stats()
    logger.info(f"Backtest complete. Sharpe: {stats.get('Sharpe Ratio', float('nan')):.3f}")

    period_start = config.data.start if config.data else None
    period_end = config.data.end if config.data else None
    symbols = list(close.columns) if hasattr(close, "columns") else []

    total_return = float(stats.get("Total Return [%]", 0.0)) / 100

    # "Annualized Return [%]" does not exist in vectorbt stats().
    # Compute CAGR from total_return and the actual data period instead.
    if period_start and period_end:
        n_years = (period_end - period_start).days / 365.25
        cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    else:
        # Fallback when no date range is available
        try:
            ann = portfolio.annualized_return()
            cagr = float(ann) if np.isscalar(ann) else float(ann.iloc[0])
        except Exception:
            cagr = 0.0

    return BacktestResult(
        strategy_name=strategy.name,
        symbols=symbols,
        period_start=period_start,
        period_end=period_end,
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=float(stats.get("Sharpe Ratio", float("nan"))),
        max_drawdown=float(stats.get("Max Drawdown [%]", 0.0)) / 100,
        n_trades=int(stats.get("Total Trades", 0)),
        commission_paid=float(stats.get("Total Fees Paid", 0.0)),
        metadata={"vbt_stats": {k: _safe(v) for k, v in stats.items()}},
    )


def _safe(v: object) -> object:
    """Convert non-JSON-serialisable stats values."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (pd.Timestamp, pd.Timedelta)):
        return str(v)
    return v
