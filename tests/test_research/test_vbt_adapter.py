"""Tests for the vectorbt research adapter.

Tests are split into two groups:
  - No-vbt: config validation, signal_frame_to_weights, _prepare_orders.
    These run without the research extra installed.
  - vbt-required: full backtest execution.
    Skipped automatically if vectorbt is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_stack.research.vbt_adapter import (
    VbtRunConfig,
    _compute_annual_turnover,
    _prepare_orders,
    signal_frame_to_weights,
)
from quant_stack.signals.base import SignalFrame
from quant_stack.core.schemas import SignalSource


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def daily_close() -> pd.DataFrame:
    """300 business-day synthetic prices for 3 assets."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=300)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (300, 3)), axis=0))
    return pd.DataFrame(prices, index=idx, columns=["SPY", "QQQ", "IEF"])


@pytest.fixture
def monthly_weights(daily_close) -> pd.DataFrame:
    """Month-end weight snapshots (equal-weight, 3 assets)."""
    me_idx = daily_close.resample("ME").last().index
    w = pd.DataFrame(
        {"SPY": 1 / 3, "QQQ": 1 / 3, "IEF": 1 / 3},
        index=me_idx,
    )
    return w


@pytest.fixture
def signal_frame_fixture(daily_close) -> SignalFrame:
    """Simple SignalFrame: all 3 assets long (strength 1.0) after first 20 bars."""
    n = len(daily_close)
    signals = pd.DataFrame(np.nan, index=daily_close.index, columns=daily_close.columns)
    strength = pd.DataFrame(np.nan, index=daily_close.index, columns=daily_close.columns)
    signals.iloc[20:] = 1.0
    strength.iloc[20:] = 1.0
    return SignalFrame(
        signals=signals,
        strength=strength,
        strategy_name="test_sf",
        source=SignalSource.RESEARCH,
    )


# ── VbtRunConfig ──────────────────────────────────────────────────────────────

class TestVbtRunConfig:
    def test_defaults(self) -> None:
        c = VbtRunConfig()
        assert c.initial_cash == 100_000.0
        assert c.commission == 0.001
        assert c.slippage == 0.001
        assert c.rebalance_freq == "ME"
        assert c.risk_free_rate == 0.05

    def test_no_rebalance_freq(self) -> None:
        c = VbtRunConfig(rebalance_freq=None)
        assert c.rebalance_freq is None

    def test_commission_bounds(self) -> None:
        with pytest.raises(Exception):
            VbtRunConfig(commission=-0.001)
        with pytest.raises(Exception):
            VbtRunConfig(commission=0.10)  # > 0.05

    def test_frozen(self) -> None:
        c = VbtRunConfig()
        with pytest.raises(Exception):
            c.commission = 0.002  # type: ignore[misc]


# ── signal_frame_to_weights ───────────────────────────────────────────────────

class TestSignalFrameToWeights:
    def test_warmup_rows_are_nan(self, signal_frame_fixture) -> None:
        w = signal_frame_to_weights(signal_frame_fixture)
        warmup = signal_frame_fixture.signals.isna().all(axis=1)
        assert w.loc[warmup].isna().all().all()

    def test_post_warmup_sums_to_one(self, signal_frame_fixture) -> None:
        w = signal_frame_to_weights(signal_frame_fixture)
        valid = w.loc[~signal_frame_fixture.signals.isna().all(axis=1)]
        row_sums = valid.sum(axis=1)
        assert (row_sums.abs() - 1.0).abs().max() < 1e-9

    def test_equal_strength_gives_equal_weights(self, signal_frame_fixture) -> None:
        w = signal_frame_to_weights(signal_frame_fixture)
        valid = w.dropna(how="any")
        assert (valid.sub(valid.mean(axis=1), axis=0).abs() < 1e-9).all().all()

    def test_no_long_row_gives_zero_weights(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=5)
        # Row 2: no long signals (all flat = 0.0)
        sigs = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
        sigs.iloc[2] = 0.0
        strengths = sigs.copy()
        sf = SignalFrame(signals=sigs, strength=strengths, strategy_name="t", source=SignalSource.RESEARCH)
        w = signal_frame_to_weights(sf)
        assert (w.iloc[2] == 0.0).all()

    def test_partial_longs_normalized(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=3)
        sigs = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [1.0, 0.0, 0.0]}, index=idx)
        strengths = pd.DataFrame({"A": [0.6, 0.6, 0.0], "B": [0.4, 0.0, 0.0]}, index=idx)
        sf = SignalFrame(signals=sigs, strength=strengths, strategy_name="t", source=SignalSource.RESEARCH)
        w = signal_frame_to_weights(sf)
        # Row 0: A=0.6, B=0.4 → normalised A=0.6, B=0.4 (already sum to 1)
        assert abs(w.iloc[0]["A"] - 0.6) < 1e-9
        # Row 1: only A long → A=1.0, B=0.0
        assert abs(w.iloc[1]["A"] - 1.0) < 1e-9
        assert abs(w.iloc[1]["B"]) < 1e-9
        # Row 2: no longs → 0
        assert (w.iloc[2] == 0.0).all()


# ── _prepare_orders ───────────────────────────────────────────────────────────

class TestPrepareOrders:
    def test_reindexed_to_close(self, daily_close, monthly_weights) -> None:
        orders = _prepare_orders(daily_close, monthly_weights, "ME")
        assert orders.index.equals(daily_close.index)

    def test_mostly_nan_between_rebalances(self, daily_close, monthly_weights) -> None:
        orders = _prepare_orders(daily_close, monthly_weights, "ME")
        non_nan_count = orders.dropna(how="all").shape[0]
        # ~12 month-end rebalance dates in 300 bars (≈ 14 months of data)
        assert non_nan_count <= 20

    def test_shifted_by_one_bday(self, daily_close, monthly_weights) -> None:
        # Rebalance dates should not fall on exact month-ends (shifted +1 BDay)
        orders = _prepare_orders(daily_close, monthly_weights, "ME")
        rebal_dates = orders.dropna(how="all").index
        me_dates = monthly_weights.index
        for d in rebal_dates:
            # none of the execution dates should be a month-end date from weights
            assert d not in me_dates

    def test_no_rebalance_freq_uses_all_rows(self, daily_close, monthly_weights) -> None:
        orders = _prepare_orders(daily_close, monthly_weights, rebalance_freq=None)
        # All weight rows (shifted by 1 bday) should appear in orders
        non_nan = orders.dropna(how="all")
        # monthly_weights has ~12-14 rows; they should all land in close index
        assert len(non_nan) >= len(monthly_weights) - 2  # allow for BDay boundary


# ── _compute_annual_turnover ──────────────────────────────────────────────────

class TestComputeAnnualTurnover:
    def test_zero_turnover_static_weights(self) -> None:
        idx = pd.bdate_range("2023-01-01", periods=252)
        # Static 33%/33%/34% allocation, no changes
        orders = pd.DataFrame(
            {"A": [1 / 3] + [np.nan] * 251,
             "B": [1 / 3] + [np.nan] * 251,
             "C": [1 / 3] + [np.nan] * 251},
            index=idx,
        )
        t = _compute_annual_turnover(orders, 252)
        # One initial trade, then nothing → near-zero annualized after 252 bars
        assert t < 0.05

    def test_full_turnover_monthly_rotation(self) -> None:
        # Every month, rotate entirely to a different asset
        idx = pd.bdate_range("2023-01-01", periods=252)
        me_idx = pd.Series(index=idx).resample("ME").last().index
        orders = pd.DataFrame(np.nan, index=idx, columns=["A", "B"])
        toggle = True
        for d in me_idx:
            if d in idx:
                orders.loc[d, "A"] = 1.0 if toggle else 0.0
                orders.loc[d, "B"] = 0.0 if toggle else 1.0
                toggle = not toggle
        t = _compute_annual_turnover(orders, 252)
        # Each full rotation = 200% one-way, ~12 times/year
        assert t > 0.5  # at minimum 50% p.a. turnover


# ── End-to-end (requires vectorbt) ───────────────────────────────────────────

vbt = pytest.importorskip("vectorbt", reason="vectorbt not installed")


class TestRunVbtBacktest:
    """Integration tests — only run when vectorbt is available."""

    def _make_close(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-02", periods=500)
        prices = 100 * np.exp(np.cumsum(rng.normal(5e-4, 0.01, (500, 3)), axis=0))
        return pd.DataFrame(prices, index=idx, columns=["SPY", "QQQ", "IEF"])

    def _make_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        me_idx = close.resample("ME").last().index
        return pd.DataFrame(
            {"SPY": 0.4, "QQQ": 0.4, "IEF": 0.2},
            index=me_idx,
        )

    def test_returns_backtest_result(self) -> None:
        from quant_stack.core.schemas import BacktestResult
        from quant_stack.research.vbt_adapter import run_vbt_backtest

        close = self._make_close()
        weights = self._make_weights(close)
        result = run_vbt_backtest(close, weights, strategy_name="test_strategy")
        assert isinstance(result, BacktestResult)

    def test_result_fields_populated(self) -> None:
        from quant_stack.research.vbt_adapter import run_vbt_backtest

        close = self._make_close()
        weights = self._make_weights(close)
        r = run_vbt_backtest(close, weights, strategy_name="test_strategy")

        assert r.strategy_name == "test_strategy"
        assert set(r.symbols) == {"SPY", "QQQ", "IEF"}
        assert r.total_return > -1.0  # not total loss
        assert r.cagr > -1.0
        assert not np.isnan(r.sharpe_ratio)
        assert r.max_drawdown >= 0.0
        assert r.n_trades >= 0
        assert r.annual_turnover is not None and r.annual_turnover >= 0

    def test_period_dates_set(self) -> None:
        from quant_stack.research.vbt_adapter import run_vbt_backtest

        close = self._make_close()
        weights = self._make_weights(close)
        r = run_vbt_backtest(close, weights, strategy_name="test_strategy")
        assert r.period_start is not None
        assert r.period_end is not None
        assert r.period_end > r.period_start

    def test_benchmark_return_populated(self) -> None:
        from quant_stack.research.vbt_adapter import run_vbt_backtest

        close = self._make_close()
        weights = self._make_weights(close)
        bm = close["SPY"]
        r = run_vbt_backtest(close, weights, benchmark_close=bm, strategy_name="s")
        assert r.benchmark_return is not None

    def test_cash_column_excluded(self) -> None:
        from quant_stack.research.vbt_adapter import run_vbt_backtest

        close = self._make_close()
        me_idx = close.resample("ME").last().index
        # weights include CASH column
        weights = pd.DataFrame(
            {"SPY": 0.3, "QQQ": 0.3, "IEF": 0.2, "CASH": 0.2},
            index=me_idx,
        )
        r = run_vbt_backtest(close, weights, strategy_name="with_cash")
        assert "CASH" not in r.symbols
        assert "CASH" not in r.metadata.get("vbt_stats", {})

    def test_missing_weight_column_raises(self) -> None:
        from quant_stack.research.vbt_adapter import run_vbt_backtest

        close = self._make_close()
        me_idx = close.resample("ME").last().index
        weights = pd.DataFrame({"TSLA": 1.0}, index=me_idx)
        with pytest.raises(ValueError, match="not found in close prices"):
            run_vbt_backtest(close, weights, strategy_name="bad")

    def test_no_rebalance_freq(self) -> None:
        from quant_stack.research.vbt_adapter import run_vbt_backtest

        close = self._make_close()
        me_idx = close.resample("ME").last().index
        weights = pd.DataFrame(
            {"SPY": 1 / 3, "QQQ": 1 / 3, "IEF": 1 / 3},
            index=me_idx,
        )
        cfg = VbtRunConfig(rebalance_freq=None)
        r = run_vbt_backtest(close, weights, config=cfg, strategy_name="no_resamp")
        assert isinstance(r.total_return, float)

    def test_signal_frame_roundtrip(self) -> None:
        """signal_frame_to_weights → run_vbt_backtest full pipeline."""
        from quant_stack.research.vbt_adapter import run_vbt_backtest

        close = self._make_close()
        n = len(close)
        sigs = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        sigs.iloc[:63] = np.nan   # simulate factor warmup
        sf = SignalFrame(
            signals=sigs,
            strength=sigs.copy(),
            strategy_name="all_long",
            source=SignalSource.RESEARCH,
        )
        weights = signal_frame_to_weights(sf)
        r = run_vbt_backtest(close, weights, strategy_name="sf_roundtrip")
        assert isinstance(r.total_return, float)
