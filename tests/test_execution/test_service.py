"""Tests for execution domain models, service, and adapters.

All tests run without LEAN, vectorbt, or network access.
Fixtures build minimal synthetic data (no market data downloads).
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from quant_stack.core.config import RiskConfig
from quant_stack.core.schemas import PortfolioWeights
from quant_stack.execution.adapters import (
    DryRunExecutionAdapter,
    LeanExecutionAdapter,
    PaperExecutionAdapter,
)
from quant_stack.execution.domain import (
    OrderSide,
    PositionSnapshot,
    TargetWeights,
    target_weights_from_portfolio_weights,
)
from quant_stack.execution.service import (
    RebalanceService,
    check_order_plan,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def target() -> TargetWeights:
    return TargetWeights(
        strategy_name="test_strategy",
        rebalance_date=date(2025, 12, 31),
        weights={"QQQ": 0.40, "XLI": 0.35, "GDX": 0.25},
    )


@pytest.fixture
def snapshot_cash() -> PositionSnapshot:
    """All-cash portfolio (first rebalance scenario)."""
    return PositionSnapshot(
        timestamp=datetime(2025, 12, 31, 9, 30),
        nav=100_000.0,
        positions={},
        cash_fraction=1.0,
        source="manual",
    )


@pytest.fixture
def snapshot_invested() -> PositionSnapshot:
    """Partially-invested portfolio (routine rebalance scenario)."""
    return PositionSnapshot(
        timestamp=datetime(2025, 12, 31, 9, 30),
        nav=100_000.0,
        positions={"QQQ": 0.333, "XLV": 0.333, "XLI": 0.334},
        cash_fraction=0.0,
        source="paper",
    )


@pytest.fixture
def risk() -> RiskConfig:
    return RiskConfig(
        max_position_size=0.40,
        max_drawdown_halt=0.15,
        daily_loss_limit=0.03,
    )


@pytest.fixture
def dry_service(risk, tmp_path) -> RebalanceService:
    return RebalanceService(
        adapter=DryRunExecutionAdapter(),
        risk=risk,
        dry_run=True,
        artifacts_dir=tmp_path / "artifacts",
    )


# ── TargetWeights ─────────────────────────────────────────────────────────────

class TestTargetWeights:
    def test_construction(self, target) -> None:
        assert target.strategy_name == "test_strategy"
        assert target.weights["QQQ"] == pytest.approx(0.40)
        assert sum(target.weights.values()) == pytest.approx(1.0)

    def test_frozen(self, target) -> None:
        with pytest.raises(Exception):
            target.strategy_name = "other"  # type: ignore[misc]


# ── PositionSnapshot ──────────────────────────────────────────────────────────

class TestPositionSnapshot:
    def test_all_cash(self, snapshot_cash) -> None:
        assert snapshot_cash.cash_fraction == 1.0
        assert snapshot_cash.positions == {}

    def test_nav_positive(self) -> None:
        with pytest.raises(Exception):
            PositionSnapshot(
                timestamp=datetime.now(), nav=-1.0, cash_fraction=1.0
            )


# ── target_weights_from_portfolio_weights ─────────────────────────────────────

class TestBoundaryConversion:
    def test_basic_conversion(self) -> None:
        pw = PortfolioWeights(
            weights={"QQQ": 0.5, "IEF": 0.5},
            method="blend_70_30",
            rebalance_date=date(2025, 12, 31),
        )
        tw = target_weights_from_portfolio_weights(pw, strategy_name="strat")
        assert tw.strategy_name == "strat"
        assert tw.rebalance_date == date(2025, 12, 31)
        assert tw.weights == {"QQQ": 0.5, "IEF": 0.5}

    def test_cash_column_excluded(self) -> None:
        pw = PortfolioWeights(
            weights={"QQQ": 0.6, "CASH": 0.4},
            rebalance_date=date(2025, 12, 31),
        )
        tw = target_weights_from_portfolio_weights(pw, strategy_name="strat")
        assert "CASH" not in tw.weights
        assert tw.weights == {"QQQ": 0.6}

    def test_source_record_id_passed(self) -> None:
        pw = PortfolioWeights(
            weights={"A": 0.5, "B": 0.5},
            rebalance_date=date(2025, 12, 31),
        )
        tw = target_weights_from_portfolio_weights(
            pw, strategy_name="s", source_record_id="abc-123"
        )
        assert tw.source_record_id == "abc-123"

    def test_rebalance_date_defaults_to_today_when_none(self) -> None:
        pw = PortfolioWeights(weights={"A": 1.0})
        tw = target_weights_from_portfolio_weights(pw, strategy_name="s")
        assert tw.rebalance_date == date.today()


# ── RebalanceService._build_plan ──────────────────────────────────────────────

class TestBuildPlan:
    def test_all_cash_creates_buy_orders(
        self, dry_service, target, snapshot_cash
    ) -> None:
        plan, _ = dry_service.run(target, snapshot_cash)
        sides = {o.side for o in plan.orders}
        assert sides == {OrderSide.BUY}

    def test_orders_match_target_symbols(
        self, dry_service, target, snapshot_cash
    ) -> None:
        plan, _ = dry_service.run(target, snapshot_cash)
        order_syms = {o.symbol for o in plan.orders}
        # All target symbols should appear as orders (from 0 → target)
        assert order_syms == set(target.weights)

    def test_delta_values_correct(
        self, dry_service, target, snapshot_cash
    ) -> None:
        plan, _ = dry_service.run(target, snapshot_cash)
        for order in plan.orders:
            # From all-cash, delta == target_weight
            assert abs(order.delta_weight - target.weights[order.symbol]) < 1e-9

    def test_partial_rebalance(
        self, dry_service, target, snapshot_invested
    ) -> None:
        plan, _ = dry_service.run(target, snapshot_invested)
        # XLV is held at 0.333 but not in target → should appear as SELL order
        xlv_order = next((o for o in plan.orders if o.symbol == "XLV"), None)
        assert xlv_order is not None
        assert xlv_order.side == OrderSide.SELL
        assert xlv_order.target_weight == pytest.approx(0.0)

    def test_min_trade_size_filters_small_diffs(
        self, risk, target, snapshot_invested, tmp_path
    ) -> None:
        # snapshot_invested has QQQ=0.333; target has QQQ=0.40
        # delta = 0.067 → exceeds default min_trade_size=0.005 → included
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=risk,
            dry_run=True,
            min_trade_size=0.10,  # very high threshold
            artifacts_dir=tmp_path,
        )
        plan, _ = svc.run(target, snapshot_invested)
        # Only orders with |delta| >= 0.10 should appear
        for order in plan.orders:
            assert abs(order.delta_weight) >= 0.10

    def test_turnover_correct(
        self, dry_service, target, snapshot_cash
    ) -> None:
        plan, _ = dry_service.run(target, snapshot_cash)
        expected = sum(abs(w) for w in target.weights.values())
        assert abs(plan.total_turnover - expected) < 1e-6

    def test_turnover_partial_rebalance_lower_than_full(
        self, dry_service, target, snapshot_cash, snapshot_invested
    ) -> None:
        plan_full, _ = dry_service.run(target, snapshot_cash)
        plan_partial, _ = dry_service.run(target, snapshot_invested)
        # Partial rebalance should move less weight
        assert plan_partial.total_turnover < plan_full.total_turnover


# ── check_order_plan (risk checks) ────────────────────────────────────────────

class TestRiskChecks:
    def test_passes_clean_plan(
        self, dry_service, target, snapshot_cash, risk
    ) -> None:
        plan, result = dry_service.run(target, snapshot_cash)
        assert result.risk_check is not None
        assert result.risk_check.passed

    def test_max_position_size_violation(
        self, snapshot_cash, tmp_path
    ) -> None:
        # Target has QQQ=0.60 which exceeds max_position_size=0.50
        target_over = TargetWeights(
            strategy_name="over_alloc",
            rebalance_date=date.today(),
            weights={"QQQ": 0.60, "SPY": 0.40},
        )
        strict_risk = RiskConfig(
            max_position_size=0.50,
            max_drawdown_halt=0.15,
            daily_loss_limit=0.03,
        )
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=strict_risk,
            dry_run=True,
            artifacts_dir=tmp_path,
        )
        _, result = svc.run(target_over, snapshot_cash)
        assert result.risk_check is not None
        assert not result.risk_check.passed
        assert any(v.rule == "max_position_size" for v in result.risk_check.violations)
        assert not result.success

    def test_max_turnover_violation(self, risk, snapshot_cash, tmp_path) -> None:
        target_full = TargetWeights(
            strategy_name="s",
            rebalance_date=date.today(),
            weights={"A": 0.5, "B": 0.5},
        )
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=risk,
            dry_run=True,
            max_turnover=0.01,   # absurdly tight
            artifacts_dir=tmp_path,
        )
        snap = PositionSnapshot(
            timestamp=datetime.now(), nav=100_000.0, positions={}, cash_fraction=1.0
        )
        _, result = svc.run(target_full, snap)
        assert result.risk_check is not None
        assert not result.risk_check.passed
        assert any(v.rule == "max_turnover" for v in result.risk_check.violations)


# ── Position reconciliation & cash checks ────────────────────────────────────

class TestSafetyWarnings:
    def test_reconciliation_warning_logged(
        self, target, risk, tmp_path
    ) -> None:
        # positions(0.80) + cash(0.80) = 1.60 → over 100%
        bad_snapshot = PositionSnapshot(
            timestamp=datetime.now(),
            nav=100_000.0,
            positions={"QQQ": 0.80},
            cash_fraction=0.80,
        )
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=risk,
            dry_run=True,
            artifacts_dir=tmp_path,
        )
        _, result = svc.run(target, bad_snapshot)
        assert any("RECONCILIATION" in e for e in result.log_entries)
        assert result.success  # warning only — does not block

    def test_low_cash_warning_logged(
        self, target, risk, tmp_path
    ) -> None:
        # All invested, zero cash — turnover cost will exceed cash
        invested_snapshot = PositionSnapshot(
            timestamp=datetime.now(),
            nav=100_000.0,
            positions={"QQQ": 0.999},
            cash_fraction=0.0,
        )
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=risk,
            dry_run=True,
            artifacts_dir=tmp_path,
        )
        _, result = svc.run(target, invested_snapshot)
        assert any("LOW CASH" in e for e in result.log_entries)
        assert result.success  # warning only — does not block


# ── Kill switch ────────────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_blocks_all_execution(
        self, target, snapshot_cash, risk, tmp_path
    ) -> None:
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=risk,
            dry_run=False,
            kill_switch=True,
            artifacts_dir=tmp_path,
        )
        _, result = svc.run(target, snapshot_cash)
        assert not result.success
        assert result.adapter_mode == "blocked"


# ── Duplicate guard ────────────────────────────────────────────────────────────

class TestDuplicateGuard:
    def test_second_identical_run_skipped(
        self, target, snapshot_cash, risk, tmp_path
    ) -> None:
        svc = RebalanceService(
            adapter=PaperExecutionAdapter(),
            risk=risk,
            dry_run=False,
            artifacts_dir=tmp_path,
        )
        _, r1 = svc.run(target, snapshot_cash)
        assert r1.success

        # Same target weights → duplicate guard should fire
        _, r2 = svc.run(target, snapshot_cash)
        # Second run is still "success" but with 0 orders attempted
        assert r2.success
        assert r2.orders_attempted == 0


# ── DryRunExecutionAdapter ────────────────────────────────────────────────────

class TestDryRunAdapter:
    def test_returns_success(self, dry_service, target, snapshot_cash) -> None:
        _, result = dry_service.run(target, snapshot_cash)
        assert result.success

    def test_no_orders_filled(self, dry_service, target, snapshot_cash) -> None:
        _, result = dry_service.run(target, snapshot_cash)
        assert result.orders_filled == 0

    def test_adapter_mode_label(self, dry_service, target, snapshot_cash) -> None:
        _, result = dry_service.run(target, snapshot_cash)
        assert result.adapter_mode == "dry_run"

    def test_log_contains_symbol_info(
        self, dry_service, target, snapshot_cash
    ) -> None:
        _, result = dry_service.run(target, snapshot_cash)
        log_text = " ".join(result.log_entries)
        for sym in target.weights:
            assert sym in log_text


# ── PaperExecutionAdapter ──────────────────────────────────────────────────────

class TestPaperAdapter:
    def test_fills_orders(self, target, snapshot_cash, risk, tmp_path) -> None:
        adapter = PaperExecutionAdapter()
        svc = RebalanceService(
            adapter=adapter, risk=risk, dry_run=False, artifacts_dir=tmp_path
        )
        _, result = svc.run(target, snapshot_cash)
        assert result.orders_filled == len(target.weights)
        assert result.orders_rejected == 0

    def test_positions_updated(self, target, snapshot_cash, risk, tmp_path) -> None:
        adapter = PaperExecutionAdapter()
        svc = RebalanceService(
            adapter=adapter, risk=risk, dry_run=False, artifacts_dir=tmp_path
        )
        svc.run(target, snapshot_cash)
        for sym, w in target.weights.items():
            assert adapter.positions.get(sym) == pytest.approx(w)

    def test_dry_run_does_not_update_positions(
        self, target, snapshot_cash, risk, tmp_path
    ) -> None:
        adapter = PaperExecutionAdapter()
        svc = RebalanceService(
            adapter=adapter, risk=risk, dry_run=True, artifacts_dir=tmp_path
        )
        svc.run(target, snapshot_cash)
        assert adapter.positions == {}


# ── LeanExecutionAdapter ──────────────────────────────────────────────────────

class TestLeanAdapter:
    def test_payload_contains_weights(
        self, target, snapshot_cash, risk, tmp_path
    ) -> None:
        adapter = LeanExecutionAdapter(output_dir=tmp_path / "lean")
        svc = RebalanceService(
            adapter=adapter, risk=risk, dry_run=True, artifacts_dir=tmp_path
        )
        _, result = svc.run(target, snapshot_cash)
        payload = result.lean_payload
        assert "weights" in payload
        assert "strategy_name" in payload
        assert payload["strategy_name"] == target.strategy_name

    def test_dry_run_does_not_write_file(
        self, target, snapshot_cash, risk, tmp_path
    ) -> None:
        lean_dir = tmp_path / "lean"
        adapter = LeanExecutionAdapter(output_dir=lean_dir)
        svc = RebalanceService(
            adapter=adapter, risk=risk, dry_run=True, artifacts_dir=tmp_path
        )
        svc.run(target, snapshot_cash)
        assert not (lean_dir / "target_weights.json").exists()

    def test_execute_writes_file(
        self, target, snapshot_cash, risk, tmp_path
    ) -> None:
        lean_dir = tmp_path / "lean"
        adapter = LeanExecutionAdapter(output_dir=lean_dir)
        svc = RebalanceService(
            adapter=adapter, risk=risk, dry_run=False, artifacts_dir=tmp_path
        )
        svc.run(target, snapshot_cash)
        out = lean_dir / "target_weights.json"
        assert out.exists()
        data = __import__("json").loads(out.read_text())
        assert "weights" in data
        assert data["metadata"]["risk_checks_passed"] is True

    def test_payload_metadata_fields(
        self, target, snapshot_cash, risk, tmp_path
    ) -> None:
        adapter = LeanExecutionAdapter(output_dir=tmp_path / "lean")
        svc = RebalanceService(
            adapter=adapter, risk=risk, dry_run=True, artifacts_dir=tmp_path
        )
        _, result = svc.run(target, snapshot_cash)
        meta = result.lean_payload["metadata"]
        assert meta["source"] == "quant_stack_execution_layer"
        assert meta["adapter"] == "LeanExecutionAdapter"
        assert "plan_id" in meta
        assert "total_turnover" in meta


# ── Artifact persistence ──────────────────────────────────────────────────────

class TestArtifacts:
    def test_order_plan_json_written(
        self, dry_service, target, snapshot_cash, tmp_path
    ) -> None:
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=RiskConfig(
                max_position_size=0.40, max_drawdown_halt=0.15, daily_loss_limit=0.03
            ),
            dry_run=True,
            artifacts_dir=tmp_path / "arts",
        )
        svc.run(target, snapshot_cash)
        json_files = list((tmp_path / "arts").glob("*_order_plan.json"))
        assert len(json_files) == 1

    def test_execution_log_written(
        self, target, snapshot_cash, tmp_path
    ) -> None:
        arts = tmp_path / "arts"
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=RiskConfig(
                max_position_size=0.40, max_drawdown_halt=0.15, daily_loss_limit=0.03
            ),
            dry_run=True,
            artifacts_dir=arts,
        )
        svc.run(target, snapshot_cash)
        log_files = list(arts.glob("*_execution_log.txt"))
        assert len(log_files) == 1

    def test_order_plan_json_structure(
        self, target, snapshot_cash, tmp_path
    ) -> None:
        arts = tmp_path / "arts"
        svc = RebalanceService(
            adapter=DryRunExecutionAdapter(),
            risk=RiskConfig(
                max_position_size=0.40, max_drawdown_halt=0.15, daily_loss_limit=0.03
            ),
            dry_run=True,
            artifacts_dir=arts,
        )
        svc.run(target, snapshot_cash)
        import json as _json
        plan_file = next(arts.glob("*_order_plan.json"))
        data = _json.loads(plan_file.read_text())
        assert "plan_id" in data
        assert "orders" in data
        assert "strategy_name" in data
        assert data["strategy_name"] == target.strategy_name
