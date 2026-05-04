"""Tests for ShadowExecutionService — specifically the D.2 Suggested Notional fix.

Bug that was fixed: `buy_scale = tradeable_nav / buy_value` was applied even when
buy_value << tradeable_nav (partial rebalance), multiplying suggested notionals by
~23x instead of leaving them at the correct delta-notional value.

Correct formula: Suggested Notional = abs(delta_weight) * nav = Delta $ in Section D.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime

import pytest

from quant_stack.core.config import RiskConfig
from quant_stack.execution.adapters import DryRunExecutionAdapter
from quant_stack.execution.domain import PositionSnapshot, TargetWeights
from quant_stack.execution.service import RebalanceService
from quant_stack.execution.shadow import ShadowExecutionService, _build_summary_markdown


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


# ── Scenario 1: partial rebalance — small delta, large existing positions ──────

class TestPartialRebalanceSuggestedNotional:
    """
    Portfolio: NAV ≈ $4,954, already invested ~96% across GDX / IBB / XLE.
    Only small deltas needed (~2pp each for GDX and IBB).
    Suggested Notional must equal Delta $, NOT target-position dollar value.
    """

    NAV = 4954.0
    # Current weights (sum + cash ≈ 1.0)
    CURRENT = {"GDX": 0.2636, "IBB": 0.3376, "XLE": 0.3563}
    CASH_FRAC = 1.0 - sum(CURRENT.values())   # ≈ 0.0425
    # Target weights (small upward shifts for GDX and IBB; XLE unchanged)
    TARGET = {"GDX": 0.2847, "IBB": 0.3590, "XLE": 0.3563}

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 4, 28, 16, 0),
            nav=self.NAV,
            positions=dict(self.CURRENT),
            cash_fraction=self.CASH_FRAC,
            source="manual",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 4, 28),
            weights=dict(self.TARGET),
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot)

    def test_orders_are_buys_only(self, shadow_result):
        orders = shadow_result.plan.orders
        assert all(str(o.side) == "buy" for o in orders), (
            "Only BUY orders expected for small upward deltas"
        )

    def test_suggested_notional_equals_delta_dollar(self, shadow_result):
        """Core fix: D.2 Suggested Notional must equal Delta $ (Section D), not target value."""
        plan = shadow_result.plan
        nav = self.NAV
        summary = shadow_result.summary_text

        for o in plan.orders:
            delta_usd = o.delta_weight * nav          # Delta $ in Section D
            target_val = o.target_weight * nav        # what the wrong formula produced

            # Extract the Suggested Notional for this symbol from the markdown
            # Pattern: | GDX | BUY | $105 |  (with optional extra columns)
            pattern = rf"\|\s*{o.symbol}\s*\|\s*BUY\s*\|\s*\$([0-9,]+)"
            match = re.search(pattern, summary)
            assert match, f"Could not find D.2 BUY row for {o.symbol} in summary"
            reported = float(match.group(1).replace(",", ""))

            # Must be close to delta notional, NOT to target position value
            assert abs(reported - delta_usd) <= 2, (
                f"{o.symbol}: Suggested Notional ${reported:,.0f} should be "
                f"≈ Delta $ ${delta_usd:,.0f}, not target value ${target_val:,.0f}"
            )
            assert abs(reported - target_val) > 50, (
                f"{o.symbol}: Suggested Notional ${reported:,.0f} must NOT equal "
                f"target position value ${target_val:,.0f} — that was the bug"
            )

    def test_gdx_suggested_notional_approx_105(self, shadow_result):
        """Regression guard: GDX delta ≈ 2.11pp × $4,954 ≈ $105."""
        plan = shadow_result.plan
        nav = self.NAV
        gdx_order = next(o for o in plan.orders if o.symbol == "GDX")
        expected = gdx_order.delta_weight * nav
        assert abs(expected - 105) < 10, f"GDX delta notional should be ~$105, got ${expected:.0f}"

        summary = shadow_result.summary_text
        match = re.search(r"\|\s*GDX\s*\|\s*BUY\s*\|\s*\$([0-9,]+)", summary)
        assert match
        reported = float(match.group(1).replace(",", ""))
        assert abs(reported - expected) <= 2

    def test_ibb_suggested_notional_approx_106(self, shadow_result):
        """Regression guard: IBB delta ≈ 2.14pp × $4,954 ≈ $106."""
        plan = shadow_result.plan
        nav = self.NAV
        ibb_order = next(o for o in plan.orders if o.symbol == "IBB")
        expected = ibb_order.delta_weight * nav
        assert abs(expected - 106) < 10, f"IBB delta notional should be ~$106, got ${expected:.0f}"

        summary = shadow_result.summary_text
        match = re.search(r"\|\s*IBB\s*\|\s*BUY\s*\|\s*\$([0-9,]+)", summary)
        assert match
        reported = float(match.group(1).replace(",", ""))
        assert abs(reported - expected) <= 2


# ── Scenario 2: BUY / SELL consistency ────────────────────────────────────────

class TestBuySellNotionalConsistency:
    """
    Suggested Notional must equal abs(Delta $) for both BUY and SELL orders.
    Mixed-direction rebalance: sell XLV entirely, buy GDX.
    """

    NAV = 100_000.0
    CURRENT = {"QQQ": 0.333, "XLV": 0.333, "XLI": 0.334}
    CASH_FRAC = 0.0
    TARGET = {"QQQ": 0.40, "XLI": 0.35, "GDX": 0.25}

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 1, 31, 16, 0),
            nav=self.NAV,
            positions=dict(self.CURRENT),
            cash_fraction=self.CASH_FRAC,
            source="paper",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 1, 31),
            weights=dict(self.TARGET),
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot)

    def test_sell_notional_equals_abs_delta(self, shadow_result):
        plan = shadow_result.plan
        nav = self.NAV
        sell_orders = [o for o in plan.orders if str(o.side) == "sell"]
        assert sell_orders, "Expected at least one SELL order"
        summary = shadow_result.summary_text

        for o in sell_orders:
            expected_notional = abs(o.delta_weight) * nav
            # SELL rows appear in a separate table: | SYMBOL | SELL | $... |
            pattern = rf"\|\s*{o.symbol}\s*\|\s*SELL\s*\|\s*\$([0-9,]+)"
            match = re.search(pattern, summary)
            assert match, f"Could not find SELL row for {o.symbol}"
            reported = float(match.group(1).replace(",", ""))
            assert abs(reported - expected_notional) <= 2, (
                f"{o.symbol} SELL: reported ${reported:,.0f} vs expected ${expected_notional:,.0f}"
            )

    def test_buy_notional_equals_delta(self, shadow_result):
        plan = shadow_result.plan
        nav = self.NAV
        buy_orders = [o for o in plan.orders if str(o.side) == "buy"]
        assert buy_orders, "Expected at least one BUY order"
        summary = shadow_result.summary_text

        for o in buy_orders:
            expected_notional = o.delta_weight * nav
            pattern = rf"\|\s*{o.symbol}\s*\|\s*BUY\s*\|\s*\$([0-9,]+)"
            match = re.search(pattern, summary)
            assert match, f"Could not find BUY row for {o.symbol}"
            reported = float(match.group(1).replace(",", ""))
            assert abs(reported - expected_notional) <= 2, (
                f"{o.symbol} BUY: reported ${reported:,.0f} vs expected ${expected_notional:,.0f}"
            )


# ── Scenario 3: Est. Qty uses Suggested Notional, not target position value ───

class TestEstQtyUsesCorrectBase:
    """
    Est. Qty = floor(Suggested Notional / ref_price)
    NOT floor(target_position_value / ref_price)
    """

    NAV = 4954.0
    CURRENT = {"GDX": 0.2636, "IBB": 0.3376, "XLE": 0.3563}
    CASH_FRAC = 1.0 - sum(CURRENT.values())
    TARGET = {"GDX": 0.2847, "IBB": 0.3590, "XLE": 0.3563}
    PRICES = {"GDX": 35.50, "IBB": 165.00, "XLE": 88.00}

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 4, 28, 16, 0),
            nav=self.NAV,
            positions=dict(self.CURRENT),
            cash_fraction=self.CASH_FRAC,
            source="manual",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 4, 28),
            weights=dict(self.TARGET),
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot, latest_prices=self.PRICES)

    def test_est_qty_based_on_delta_notional(self, shadow_result):
        plan = shadow_result.plan
        nav = self.NAV
        summary = shadow_result.summary_text

        for o in [o for o in plan.orders if str(o.side) == "buy"]:
            price = self.PRICES.get(o.symbol)
            if not price:
                continue
            delta_notional = o.delta_weight * nav
            correct_qty = math.floor(delta_notional / price)
            wrong_qty = math.floor(o.target_weight * nav / price)  # what the bug produced

            # The correct and wrong quantities must differ for the test to be meaningful
            if correct_qty == wrong_qty:
                continue  # skip if they happen to match

            # Extract Est. Qty from the markdown table
            # Row format: | SYMBOL | BUY | $notional | $price | qty | $residual |
            pattern = (
                rf"\|\s*{o.symbol}\s*\|\s*BUY\s*\|"
                rf"\s*\$[0-9,]+\s*\|\s*\$[0-9,.]+\s*\|"
                rf"\s*([0-9,]+)\s*\|"
            )
            match = re.search(pattern, summary)
            assert match, f"Could not parse Est. Qty row for {o.symbol}"
            reported_qty = int(match.group(1).replace(",", ""))

            assert reported_qty == correct_qty, (
                f"{o.symbol}: Est. Qty should be {correct_qty} "
                f"(floor({delta_notional:.0f}/{price})), "
                f"not {wrong_qty} (floor(target_value/{price})). "
                f"Got {reported_qty}."
            )


# ── Scenario 4: cash note net_cash_needed ≈ sum of BUY notionals + fee ────────

class TestCashNoteConsistency:
    """
    Section D cash note: net_cash_needed = sum(BUY delta notionals) - sum(SELL proceeds) + fee
    This must align with D.2 Suggested Notionals.
    """

    NAV = 4954.0
    CURRENT = {"GDX": 0.2636, "IBB": 0.3376, "XLE": 0.3563}
    CASH_FRAC = 1.0 - sum(CURRENT.values())
    TARGET = {"GDX": 0.2847, "IBB": 0.3590, "XLE": 0.3563}

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 4, 28, 16, 0),
            nav=self.NAV,
            positions=dict(self.CURRENT),
            cash_fraction=self.CASH_FRAC,
            source="manual",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 4, 28),
            weights=dict(self.TARGET),
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot)

    def test_cash_note_aligns_with_buy_notionals(self, shadow_result):
        plan = shadow_result.plan
        nav = self.NAV
        est_cost = plan.total_turnover * plan.estimated_cost_bps / 10_000 * nav

        buy_notional = sum(o.delta_weight * nav for o in plan.orders if o.delta_weight > 0)
        sell_proceeds = sum(-o.delta_weight * nav for o in plan.orders if o.delta_weight < 0)
        expected_net = max(0.0, buy_notional - sell_proceeds) + est_cost

        # The rebalance_plan JSON also tracks this
        plan_json = json.loads(
            (shadow_result.artifacts["rebalance_plan"]).read_text(encoding="utf-8")
        )
        reported_net = plan_json["summary"]["net_cash_needed_usd"]
        assert abs(reported_net - expected_net) <= 1.0, (
            f"net_cash_needed ${reported_net:.2f} should be ≈ ${expected_net:.2f}"
        )

    def test_net_cash_needed_within_available_cash(self, shadow_result):
        """For this scenario the small delta should fit inside available cash."""
        plan = shadow_result.plan
        nav = self.NAV
        est_cost = plan.total_turnover * plan.estimated_cost_bps / 10_000 * nav
        buy_notional = sum(o.delta_weight * nav for o in plan.orders if o.delta_weight > 0)
        net_needed = buy_notional + est_cost
        available = self.CASH_FRAC * nav
        assert net_needed <= available + 1.0, (
            f"net_needed ${net_needed:.2f} exceeds available cash ${available:.2f}"
        )


# ── Scenario 5: all-cash first rebalance (buy_value large, no regression) ─────

class TestAllCashFirstRebalance:
    """
    All-cash portfolio: delta == target weight for every symbol.
    Suggested Notional must still equal delta * nav (which == target * nav here).
    The old buy_scale would have been 1.0 in this case too (tradeable_nav ≈ buy_value),
    but we verify the output is still correct after removing buy_scale.
    """

    NAV = 100_000.0

    @pytest.fixture
    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            timestamp=datetime(2026, 1, 31, 16, 0),
            nav=self.NAV,
            positions={},
            cash_fraction=1.0,
            source="manual",
        )

    @pytest.fixture
    def target(self) -> TargetWeights:
        return TargetWeights(
            strategy_name="sector_momentum_test",
            rebalance_date=date(2026, 1, 31),
            weights={"GDX": 0.333, "IBB": 0.333, "XLE": 0.334},
        )

    @pytest.fixture
    def shadow_result(self, shadow_service, target, snapshot):
        return shadow_service.run(target, snapshot)

    def test_suggested_notional_equals_target_value_when_all_cash(self, shadow_result):
        """For all-cash, delta == target, so Suggested Notional == target * nav."""
        plan = shadow_result.plan
        nav = self.NAV
        summary = shadow_result.summary_text

        for o in plan.orders:
            assert str(o.side) == "buy"
            expected = o.delta_weight * nav  # == target_weight * nav here
            pattern = rf"\|\s*{o.symbol}\s*\|\s*BUY\s*\|\s*\$([0-9,]+)"
            match = re.search(pattern, summary)
            assert match, f"Could not find BUY row for {o.symbol}"
            reported = float(match.group(1).replace(",", ""))
            assert abs(reported - expected) <= 2, (
                f"{o.symbol}: expected ${expected:,.0f}, got ${reported:,.0f}"
            )
