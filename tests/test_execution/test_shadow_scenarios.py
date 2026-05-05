"""Scenario-based tests for ShadowExecutionService — schedule awareness and execution flags.

Covers:
  1. All-cash first deployment → BUY orders, needs_rebalance=True, executable_count > 0
  2. Portfolio already at target → 0 orders, needs_rebalance=False
  3. Large drift (full rotation) → SELL + BUY orders, needs_rebalance=True, high turnover
  4. Whole-share constraint (small NAV, small delta) → needs_rebalance=True, executable_count=0
  5. Schedule gate — last bday of rebalance month → is_scheduled_rebalance_day=True
  6. Schedule gate — mid-month or non-rebalance month → is_scheduled_rebalance_day=False
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from quant_stack.core.config import RiskConfig
from quant_stack.execution.adapters import DryRunExecutionAdapter
from quant_stack.execution.domain import PositionSnapshot, TargetWeights
from quant_stack.execution.service import RebalanceService
from quant_stack.execution.shadow import ShadowExecutionService


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def risk() -> RiskConfig:
    return RiskConfig(max_position_size=0.60, max_drawdown_halt=0.20, daily_loss_limit=0.05)


@pytest.fixture
def dry_service(risk, tmp_path) -> RebalanceService:
    return RebalanceService(
        adapter=DryRunExecutionAdapter(),
        risk=risk,
        dry_run=True,
        artifacts_dir=tmp_path / "artifacts",
    )


@pytest.fixture
def shadow_service(dry_service, tmp_path) -> ShadowExecutionService:
    return ShadowExecutionService(service=dry_service, shadow_dir=tmp_path / "shadow")


# ── Scenario 1: All-cash first deployment ─────────────────────────────────────

class TestAllCashFirstDeployment:
    """
    First rebalance: no existing positions, 100% cash, large NAV.
    Expected: needs_rebalance=True, all BUY orders, executable_count == number of targets.
    """

    NAV = 100_000.0
    TARGET = {"GDX": 0.333, "IBB": 0.333, "XLE": 0.334}
    PRICES = {"GDX": 86.0, "IBB": 170.0, "XLE": 60.0}

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 5, 5, 16, 0),
            nav=self.NAV,
            positions={},
            cash_fraction=1.0,
            source="manual",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 5, 5),
            weights=dict(self.TARGET),
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot, latest_prices=self.PRICES)

    def test_needs_rebalance(self, shadow_result):
        assert shadow_result.needs_rebalance is True

    def test_all_orders_are_buys(self, shadow_result):
        orders = shadow_result.plan.orders
        assert len(orders) == len(self.TARGET)
        assert all(str(o.side) == "buy" for o in orders)

    def test_executable_count_equals_order_count(self, shadow_result):
        # NAV=$100k, ~$33k notional per symbol; prices $60–$170 → floor >> 0 for all
        assert shadow_result.executable_count == len(self.TARGET)

    def test_not_scheduled_rebalance_day(self, shadow_result):
        # May 5 is not the last business day of May
        assert shadow_result.is_scheduled_rebalance_day is False


# ── Scenario 2: Portfolio already at target ────────────────────────────────────

class TestPortfolioAtTarget:
    """
    Current positions exactly match target weights.
    Expected: 0 orders, needs_rebalance=False, executable_count=0.
    """

    NAV = 100_000.0
    WEIGHTS = {"GDX": 0.333, "IBB": 0.333, "XLE": 0.334}

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 5, 5, 16, 0),
            nav=self.NAV,
            positions=dict(self.WEIGHTS),
            cash_fraction=0.0,
            source="manual",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 5, 5),
            weights=dict(self.WEIGHTS),
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot)

    def test_no_orders(self, shadow_result):
        assert len(shadow_result.plan.orders) == 0

    def test_needs_rebalance_false(self, shadow_result):
        assert shadow_result.needs_rebalance is False

    def test_executable_count_zero(self, shadow_result):
        assert shadow_result.executable_count == 0


# ── Scenario 3: Large drift — partial rotation (SELL + BUY) ──────────────────

class TestLargeDriftPartialRotation:
    """
    Current portfolio holds 2 symbols not in target, plus 1 overlap.
    QQQ and XLV are sold; IBB and XLE are bought.  GDX delta is ~0 and filtered.

    Deltas: QQQ -33.3%, XLV -33.3%, IBB +33.3%, XLE +33.4% → turnover ≈ 133% < 150% cap.
    Expected: SELL orders (QQQ, XLV), BUY orders (IBB, XLE), needs_rebalance=True.
    """

    NAV = 100_000.0
    CURRENT = {"QQQ": 0.333, "XLV": 0.333, "GDX": 0.334}   # sum=1.0
    CASH_FRAC = 0.0
    TARGET = {"GDX": 0.333, "IBB": 0.333, "XLE": 0.334}

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 5, 5, 16, 0),
            nav=self.NAV,
            positions=dict(self.CURRENT),
            cash_fraction=self.CASH_FRAC,
            source="manual",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 5, 5),
            weights=dict(self.TARGET),
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot)

    def test_needs_rebalance(self, shadow_result):
        assert shadow_result.needs_rebalance is True

    def test_result_success(self, shadow_result):
        # Turnover ≈ 133%; sell proceeds fund buys; all risk checks should pass
        assert shadow_result.result.success is True

    def test_has_sell_orders(self, shadow_result):
        sells = [o for o in shadow_result.plan.orders if str(o.side) == "sell"]
        assert len(sells) > 0

    def test_has_buy_orders(self, shadow_result):
        buys = [o for o in shadow_result.plan.orders if str(o.side) == "buy"]
        assert len(buys) > 0

    def test_turnover_within_cap(self, shadow_result):
        # ~133% turnover — confirms orders exist but stays under the 150% risk cap
        assert 0.50 < shadow_result.plan.total_turnover < 1.50


# ── Scenario 4: Whole-share constraint ────────────────────────────────────────

class TestWholeShareConstraint:
    """
    Small NAV with small deltas — floor(delta_notional / price) == 0 for all buy orders.
    Expected: needs_rebalance=True (drift exists), executable_count=0 (unexecutable).

    Prices: GDX $86, IBB $170
    Delta per symbol: 1% × $5,000 = $50
      floor($50 / $86)  = 0  → GDX not broker-executable
      floor($50 / $170) = 0  → IBB not broker-executable
    """

    NAV = 5_000.0
    CURRENT = {"GDX": 0.27, "IBB": 0.34, "XLE": 0.37}
    CASH_FRAC = 0.02                            # 0.27+0.34+0.37+0.02 = 1.00
    TARGET = {"GDX": 0.28, "IBB": 0.35, "XLE": 0.37}   # XLE unchanged → 2 buy orders
    PRICES = {"GDX": 86.0, "IBB": 170.0, "XLE": 59.0}

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 5, 5, 16, 0),
            nav=self.NAV,
            positions=dict(self.CURRENT),
            cash_fraction=self.CASH_FRAC,
            source="manual",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 5, 5),
            weights=dict(self.TARGET),
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot, latest_prices=self.PRICES)

    def test_needs_rebalance(self, shadow_result):
        # Drift is real — the signal recommends adjustment
        assert shadow_result.needs_rebalance is True

    def test_executable_count_zero(self, shadow_result):
        # Both buy orders round to 0 shares at reference prices
        assert shadow_result.executable_count == 0

    def test_buy_orders_exist_but_unexecutable(self, shadow_result):
        buys = [o for o in shadow_result.plan.orders if str(o.side) == "buy"]
        assert len(buys) > 0, "Strategy signals buy orders even though they're unexecutable"


# ── Scenario 5: Schedule gate — is_scheduled_rebalance_day ───────────────────

class TestScheduleGate:
    """
    Verify the is_scheduled_rebalance_day flag under different calendar conditions.

    2ME rebalance months: Jan · Mar · May · Jul · Sep · Nov (anchored 2010-01-01).

    May 2026 last business day: May 31 is Sunday → May 30 is Saturday → May 29 (Fri) ✓
    Jan 2026 last business day: Jan 31 is Saturday → Jan 30 (Fri) ✓
    Apr 2026: not a rebalance month → never scheduled, even at month-end.
    """

    NAV = 100_000.0
    # Large enough drift to guarantee orders (needed for needs_rebalance=True)
    CURRENT = {"GDX": 0.333, "IBB": 0.333, "XLE": 0.334}
    CASH_FRAC = 0.0
    TARGET = {"GDX": 0.40, "IBB": 0.35, "XLE": 0.25}

    def _snapshot(self, ts: datetime) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=ts,
            nav=self.NAV,
            positions=dict(self.CURRENT),
            cash_fraction=self.CASH_FRAC,
            source="manual",
        )

    def _target(self, d: date) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=d,
            weights=dict(self.TARGET),
        )

    def test_last_bday_of_may_is_scheduled(self, shadow_service):
        # May 29, 2026: last business day of May (a 2ME rebalance month)
        result = shadow_service.run(
            self._target(date(2026, 5, 29)),
            self._snapshot(datetime(2026, 5, 29, 16, 0)),
        )
        assert result.is_scheduled_rebalance_day is True

    def test_mid_may_is_not_scheduled(self, shadow_service):
        # May 5 is in a rebalance month but not the last business day
        result = shadow_service.run(
            self._target(date(2026, 5, 5)),
            self._snapshot(datetime(2026, 5, 5, 16, 0)),
        )
        assert result.is_scheduled_rebalance_day is False

    def test_april_month_end_is_not_scheduled(self, shadow_service):
        # April is not a 2ME rebalance month → never scheduled
        result = shadow_service.run(
            self._target(date(2026, 4, 30)),
            self._snapshot(datetime(2026, 4, 30, 16, 0)),
        )
        assert result.is_scheduled_rebalance_day is False

    def test_last_bday_of_jan_is_scheduled(self, shadow_service):
        # Jan 30, 2026: last business day of January (Jan 31 is Saturday)
        result = shadow_service.run(
            self._target(date(2026, 1, 30)),
            self._snapshot(datetime(2026, 1, 30, 16, 0)),
        )
        assert result.is_scheduled_rebalance_day is True

    def test_scheduled_day_with_drift_shows_needs_rebalance(self, shadow_service):
        # Confirm both flags are True simultaneously on a scheduled day with drift
        result = shadow_service.run(
            self._target(date(2026, 5, 29)),
            self._snapshot(datetime(2026, 5, 29, 16, 0)),
        )
        assert result.needs_rebalance is True
        assert result.is_scheduled_rebalance_day is True
