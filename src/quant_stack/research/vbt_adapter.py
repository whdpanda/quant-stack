"""vectorbt adapter for low-frequency portfolio research.

Design contracts
----------------
Look-ahead prevention
    Weights at date T (computed from close data ≤ T) are executed at T+1.
    Implemented by shifting the order index forward by 1 business day.
    The caller must ensure weights are computed without future knowledge.

Monthly rebalancing
    config.rebalance_freq = "ME" resamples the weight series to month-end,
    then shifts the execution index by 1 BDay. On all other bars vectorbt
    receives NaN → it holds the last position without trading.

CASH column
    Silently excluded before passing size to vectorbt. An allocator that
    reserves a cash_buffer (e.g., weights["CASH"] = 0.1) will naturally
    leave that fraction uninvested in the vbt portfolio.

Inputs
    close   : daily adjusted-close prices, DatetimeIndex × symbol columns.
    weights : target-weight DataFrame, same symbol columns (CASH excluded
              automatically). Dense (daily) or sparse (rebalance dates only).
              NaN rows during factor warm-up period are preserved → cash held.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from quant_stack.core.schemas import BacktestResult
from quant_stack.signals.base import SignalFrame


# ── Config ────────────────────────────────────────────────────────────────────

class VbtRunConfig(BaseModel):
    """Runtime parameters for the vectorbt adapter."""

    model_config = ConfigDict(frozen=True)

    initial_cash: float = Field(default=100_000.0, gt=0)
    commission: float = Field(default=0.001, ge=0.0, le=0.05)
    slippage: float = Field(default=0.001, ge=0.0, le=0.05)
    # pandas offset alias for rebalancing; None = act on every non-NaN row
    rebalance_freq: str | None = "ME"
    risk_free_rate: float = Field(default=0.05, ge=0.0)
    ann_factor: int = Field(default=252, gt=0)
    freq: str = "1D"   # vectorbt internal freq hint


# ── Public helpers ────────────────────────────────────────────────────────────

def signal_frame_to_weights(sf: SignalFrame) -> pd.DataFrame:
    """Convert SignalFrame.strength to row-normalised portfolio weights.

    Behaviour by row type:
    - Warmup rows (all-NaN signals) → all NaN  (no order → hold initial cash)
    - Post-warmup, no longs → all 0.0  (explicit liquidation to cash)
    - Post-warmup, some longs → row-normalised strength values
    """
    warmup_mask = sf.signals.isna().all(axis=1)

    # Keep strength only where signal is long; zero out flat positions
    w = sf.strength.where(sf.signals == 1.0, other=0.0)

    # Row-normalise (rows with no longs become all-NaN temporarily)
    row_sum = w.sum(axis=1).replace(0.0, np.nan)
    w = w.div(row_sum, axis=0)

    # Post-warmup rows with no longs: set to 0 (explicit cash), not NaN (hold)
    no_long_post_warmup = w.isna().all(axis=1) & ~warmup_mask
    w.loc[no_long_post_warmup] = 0.0

    # Restore warmup NaN
    w.loc[warmup_mask] = np.nan

    return w


# ── Main entry point ──────────────────────────────────────────────────────────

def run_vbt_backtest(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    config: VbtRunConfig | None = None,
    benchmark_close: pd.Series | None = None,
    strategy_name: str = "vbt_portfolio",
) -> BacktestResult:
    """Run a weight-based portfolio backtest using vectorbt.

    Args:
        close: Daily adjusted-close prices.  DatetimeIndex × symbol columns.
        weights: Target weights.  Same columns as close (plus optional CASH).
                 Dense (daily from signal layer) or sparse (rebalance dates).
        config: Runtime parameters.  Defaults to VbtRunConfig().
        benchmark_close: Optional single-asset price series for comparison.
                         Used as a buy-and-hold benchmark.
        strategy_name: Label stored in the returned BacktestResult.

    Returns:
        BacktestResult with standardised performance metrics.
    """
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise ImportError(
            "vectorbt is not installed: pip install 'quant-stack[research]'"
        ) from exc

    if config is None:
        config = VbtRunConfig()

    # Align close and weights to a common index
    close = close.sort_index()
    weights = weights.sort_index()

    asset_cols = [c for c in weights.columns if c != "CASH"]
    missing = [c for c in asset_cols if c not in close.columns]
    if missing:
        raise ValueError(f"Weight columns not found in close prices: {missing}")

    close_assets = close[asset_cols]

    # Prepare sparse order targets (shift for look-ahead prevention)
    orders = _prepare_orders(close_assets, weights[asset_cols], config.rebalance_freq)

    logger.info(
        f"vbt backtest | strategy={strategy_name} | assets={asset_cols} "
        f"| bars={len(close_assets)} | rebalance_freq={config.rebalance_freq}"
    )

    portfolio = vbt.Portfolio.from_orders(
        close=close_assets,
        size=orders,
        size_type="targetpercent",
        fees=config.commission,
        slippage=config.slippage,
        init_cash=config.initial_cash,
        cash_sharing=True,
        group_by=True,
        freq=config.freq,
    )

    benchmark_return = (
        _run_benchmark(benchmark_close, config, vbt) if benchmark_close is not None else None
    )

    return _build_result(
        portfolio=portfolio,
        config=config,
        strategy_name=strategy_name,
        close=close_assets,
        orders=orders,
        benchmark_return=benchmark_return,
        symbols=asset_cols,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _prepare_orders(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    rebalance_freq: str | None,
) -> pd.DataFrame:
    """Return a sparse order DataFrame aligned to the close index.

    Rebalance dates contain target weights; all other rows are NaN (hold).
    The index is shifted by 1 business day to prevent look-ahead.
    """
    if rebalance_freq is not None:
        # Sample at period-end, take last known weight in each period
        w_rebal = weights.resample(rebalance_freq).last().dropna(how="all")
    else:
        w_rebal = weights.dropna(how="all")

    # Decision at date T → execution at T+1 BDay
    w_rebal = w_rebal.copy()
    w_rebal.index = w_rebal.index + pd.tseries.offsets.BDay(1)

    # Reindex to full price history: NaN on non-execution days
    return w_rebal.reindex(close.index)


def _compute_annual_turnover(orders: pd.DataFrame, ann_factor: int) -> float:
    """One-way annual portfolio turnover from the order DataFrame."""
    # Forward-fill to recover the held weight series, then diff to get changes
    w_held = orders.ffill().fillna(0.0)
    daily_one_way = w_held.diff().abs().sum(axis=1) / 2.0
    return float(daily_one_way.mean() * ann_factor)


def _run_benchmark(
    benchmark_close: pd.Series,
    config: VbtRunConfig,
    vbt: object,
) -> float:
    """Buy-and-hold benchmark total return."""
    bm = benchmark_close.sort_index().dropna()
    bm_pf = vbt.Portfolio.from_holding(  # type: ignore[attr-defined]
        close=bm,
        init_cash=config.initial_cash,
        fees=config.commission,
        freq=config.freq,
    )
    total = bm_pf.total_return()
    return float(total) if np.isscalar(total) else float(total.iloc[0])


def _build_result(
    portfolio: object,
    config: VbtRunConfig,
    strategy_name: str,
    close: pd.DataFrame,
    orders: pd.DataFrame,
    benchmark_return: float | None,
    symbols: list[str],
) -> BacktestResult:
    """Extract metrics from a vbt Portfolio and return a BacktestResult."""
    stats = portfolio.stats()  # type: ignore[attr-defined]

    def _pct(key: str, default: float = 0.0) -> float:
        return float(stats.get(key, default * 100)) / 100.0

    def _val(key: str, default: float = 0.0) -> float:
        return float(stats.get(key, default))

    total_return = _pct("Total Return [%]")
    max_dd = _pct("Max Drawdown [%]")
    sharpe = _val("Sharpe Ratio", float("nan"))
    sortino = _val("Sortino Ratio", float("nan"))
    n_trades = int(_val("Total Trades"))

    # Compute annualised volatility from daily portfolio returns
    daily_rets = portfolio.returns()  # type: ignore[attr-defined]
    if hasattr(daily_rets, "iloc"):   # Series or DataFrame
        ann_vol = float(daily_rets.std() * np.sqrt(config.ann_factor))
    else:
        ann_vol = 0.0

    # CAGR from total return and period length
    period_start = close.index[0].date()
    period_end = close.index[-1].date()
    n_years = (close.index[-1] - close.index[0]).days / 365.25
    cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # Commission paid (vectorbt gives in absolute terms)
    commission_paid = float(stats.get("Total Fees Paid", 0.0))

    # Turnover
    turnover = _compute_annual_turnover(orders, config.ann_factor)

    logger.info(
        f"vbt result | total_return={total_return:.2%} cagr={cagr:.2%} "
        f"sharpe={sharpe:.3f} sortino={sortino:.3f} max_dd={max_dd:.2%} "
        f"turnover={turnover:.1%}/yr"
    )

    return BacktestResult(
        strategy_name=strategy_name,
        symbols=symbols,
        period_start=period_start,
        period_end=period_end,
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino if not np.isnan(sortino) else None,
        max_drawdown=max_dd,
        annual_volatility=ann_vol if ann_vol > 0 else None,
        annual_turnover=turnover,
        n_trades=n_trades,
        commission_paid=commission_paid,
        benchmark_return=benchmark_return,
        metadata={"vbt_stats": {k: _safe(v) for k, v in stats.items()}},
    )


def _safe(v: object) -> object:
    """Recursively convert vbt stat values to JSON-serialisable Python types."""
    import math

    if v is None:
        return None

    # pandas NaT / NA sentinels (must come before Timestamp/Timedelta checks)
    try:
        if v is pd.NaT or v is pd.NA:
            return None
    except AttributeError:
        pass

    # numpy / pandas scalar with .item() — covers int64, float64, bool_, etc.
    # Guard: str/bytes also have .item() in some environments, skip them.
    if hasattr(v, "item") and not isinstance(v, (str, bytes, bool)):
        try:
            inner = v.item()
            return _safe(inner)          # recurse; item() may still be numpy
        except (ValueError, TypeError):
            pass

    # float: replace nan/inf with None so JSON doesn't choke
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    # bool before int — bool is a subclass of int in Python
    if isinstance(v, bool):
        return v

    if isinstance(v, int):
        return v

    if isinstance(v, str):
        return v

    # pandas Timestamp / datetime-like with isoformat()
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)

    # pandas Timedelta
    if isinstance(v, pd.Timedelta):
        return str(v)

    # dict / list: recurse
    if isinstance(v, dict):
        return {str(k): _safe(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(vv) for vv in v]

    # Fallback: stringify anything else
    try:
        return str(v)
    except Exception:
        return None
