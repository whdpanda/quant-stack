"""Shadow execution service — wraps RebalanceService with a rich artifact set.

ShadowExecutionService adds on top of RebalanceService:
  - A run-specific directory with a unique timestamped run ID
  - current_positions_snapshot.json  — serialised PositionSnapshot
  - target_weights_snapshot.json     — serialised TargetWeights
  - rebalance_plan.json              — full order plan + all diffs
  - risk_check_result.json           — every risk gate result (pass/warn/fail)
  - shadow_execution_summary.md      — human review document
  - execution_log.jsonl              — structured event log for audit/replay
  - shadow_artifacts/latest/         — copy of latest run (overwritten each time)

No orders are ever submitted. dry_run=True is enforced on construction.
"""
from __future__ import annotations

import json
import math
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from quant_stack.execution.domain import (
    ExecutionResult,
    OrderPlan,
    PositionSnapshot,
    TargetWeights,
)
from quant_stack.execution.service import RebalanceService


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ShadowRunResult:
    """All outputs from a single shadow run."""

    run_id: str
    run_dir: Path
    latest_dir: Path
    plan: OrderPlan
    result: ExecutionResult
    artifacts: dict[str, Path] = field(default_factory=dict)
    summary_text: str = ""
    needs_rebalance: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_bdays(start: date, end: date) -> int:
    """Count Mon-Fri business days elapsed between start (exclusive) and end (inclusive)."""
    count = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return count


def _fmt_cost(usd: float) -> str:
    """Format a transaction cost amount — always 2 dp, never rounds sub-dollar values to $0."""
    return f"${usd:,.2f}"


# ── Service ───────────────────────────────────────────────────────────────────

class ShadowExecutionService:
    """Dry-run orchestration with full artifact writing.

    Every call to run() creates an isolated, timestamped subdirectory inside
    shadow_dir.  A `latest/` sibling is also refreshed so downstream tools
    can always find the most recent run without knowing the timestamp.

    Args:
        service:    A pre-configured RebalanceService. Must have dry_run=True.
        shadow_dir: Root directory for all shadow run artifact directories.
    """

    def __init__(
        self,
        service: RebalanceService,
        shadow_dir: str | Path = "./shadow_artifacts",
    ) -> None:
        if not service.dry_run:
            raise ValueError(
                "ShadowExecutionService requires a RebalanceService with dry_run=True. "
                "Set dry_run=True when constructing RebalanceService."
            )
        self.service = service
        self.shadow_dir = Path(shadow_dir)
        self._run_count: int = 0  # tracks how many times run() has been called

    def run(
        self,
        target: TargetWeights,
        snapshot: PositionSnapshot,
        weighting_method: str = "",
        universe: list[str] | None = None,
        universe_type: str = "",
        latest_prices: dict[str, float] | None = None,
    ) -> ShadowRunResult:
        """Run one shadow execution cycle and persist the full artifact set.

        Steps
        -----
        1. Call RebalanceService.run(target, snapshot) with dry_run=True.
        2. Write current_positions_snapshot.json
        3. Write target_weights_snapshot.json
        4. Write rebalance_plan.json
        5. Write risk_check_result.json
        6. Write shadow_execution_summary.md
        7. Write execution_log.jsonl
        8. Copy everything to shadow_artifacts/latest/

        Args:
            target:           TargetWeights from the research layer.
            snapshot:         Current portfolio snapshot.
            weighting_method: Human-readable weighting scheme label for the summary.
            universe:         Full candidate universe used by the strategy.

        Returns:
            ShadowRunResult with all artifact paths and a human-readable summary.
        """
        self._run_count += 1
        run_count = self._run_count

        ts = datetime.now()
        run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = self.shadow_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        log_events: list[dict[str, Any]] = []

        def log_event(event: str, level: str = "INFO", **data: Any) -> None:
            log_events.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "level": level,
                "event": event,
                **data,
            })

        log_event(
            "shadow_run_started",
            run_id=run_id,
            run_count=run_count,
            strategy=target.strategy_name,
            market_data_date=str(target.rebalance_date),
            nav=snapshot.nav,
            position_count=len(snapshot.positions),
        )

        # ── 1. Run the service ─────────────────────────────────────────────
        plan, result = self.service.run(target, snapshot)
        needs_rebalance = len(plan.orders) > 0 and result.success

        log_event(
            "service_completed",
            plan_id=plan.plan_id[:8],
            orders=len(plan.orders),
            turnover=round(plan.total_turnover, 4),
            success=result.success,
            needs_rebalance=needs_rebalance,
        )

        artifacts: dict[str, Path] = {}

        # ── 2. current_positions_snapshot.json ────────────────────────────
        pos_path = run_dir / "current_positions_snapshot.json"
        pos_data = _build_positions_artifact(snapshot)
        pos_path.write_text(json.dumps(pos_data, indent=2), encoding="utf-8")
        artifacts["current_positions_snapshot"] = pos_path
        log_event("artifact_written", file="current_positions_snapshot.json")

        # ── 3. target_weights_snapshot.json ───────────────────────────────
        tw_path = run_dir / "target_weights_snapshot.json"
        tw_data = _build_target_weights_artifact(target, weighting_method, universe)
        tw_path.write_text(json.dumps(tw_data, indent=2), encoding="utf-8")
        artifacts["target_weights_snapshot"] = tw_path
        log_event("artifact_written", file="target_weights_snapshot.json")

        # ── 4. rebalance_plan.json ─────────────────────────────────────────
        plan_path = run_dir / "rebalance_plan.json"
        plan_data = _build_rebalance_plan_artifact(plan, result, run_id, needs_rebalance)
        plan_path.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")
        artifacts["rebalance_plan"] = plan_path
        log_event(
            "artifact_written",
            file="rebalance_plan.json",
            order_count=len(plan.orders),
            turnover=round(plan.total_turnover, 4),
        )

        # ── 5. risk_check_result.json ──────────────────────────────────────
        risk_path = run_dir / "risk_check_result.json"
        risk_data = _build_risk_check_artifact(
            result, plan, snapshot, target, self.service, run_count
        )
        risk_path.write_text(json.dumps(risk_data, indent=2), encoding="utf-8")
        artifacts["risk_check_result"] = risk_path
        log_event(
            "artifact_written",
            file="risk_check_result.json",
            all_passed=risk_data["all_passed"],
            warnings=risk_data["warnings"],
            violations=risk_data["violations"],
        )

        # ── 6. shadow_execution_summary.md ────────────────────────────────
        summary_text = _build_summary_markdown(
            run_id=run_id,
            target=target,
            snapshot=snapshot,
            plan=plan,
            result=result,
            risk_data=risk_data,
            needs_rebalance=needs_rebalance,
            ts=ts,
            weighting_method=weighting_method,
            universe=universe,
            universe_type=universe_type,
            latest_prices=latest_prices,
        )
        summary_path = run_dir / "shadow_execution_summary.md"
        summary_path.write_text(summary_text, encoding="utf-8")
        artifacts["shadow_execution_summary"] = summary_path
        log_event("artifact_written", file="shadow_execution_summary.md")

        # ── 7. manual_execution_log_template.json ────────────────────────
        if plan.orders:
            tpl_path = run_dir / "manual_execution_log_template.json"
            tpl_data = _build_execution_log_template(
                target, snapshot, plan, run_id, ts, latest_prices
            )
            tpl_path.write_text(json.dumps(tpl_data, indent=2), encoding="utf-8")
            artifacts["manual_execution_log_template"] = tpl_path
            log_event("artifact_written", file="manual_execution_log_template.json")

        # ── 8. execution_log.jsonl ─────────────────────────────────────────
        log_event(
            "shadow_run_completed",
            run_id=run_id,
            needs_rebalance=needs_rebalance,
            artifacts_dir=str(run_dir),
        )
        log_path = run_dir / "execution_log.jsonl"
        log_path.write_text(
            "\n".join(json.dumps(e) for e in log_events) + "\n",
            encoding="utf-8",
        )
        artifacts["execution_log"] = log_path

        # ── 8. Sync to latest/ ────────────────────────────────────────────
        latest_dir = self.shadow_dir / "latest"
        if latest_dir.exists():
            # ignore_errors=True lets rmtree clear the contents even when
            # Windows file-watchers keep the directory handle open.
            shutil.rmtree(latest_dir, ignore_errors=True)
        # dirs_exist_ok=True handles the case where rmtree left an empty dir.
        shutil.copytree(run_dir, latest_dir, dirs_exist_ok=True)

        return ShadowRunResult(
            run_id=run_id,
            run_dir=run_dir,
            latest_dir=latest_dir,
            plan=plan,
            result=result,
            artifacts=artifacts,
            summary_text=summary_text,
            needs_rebalance=needs_rebalance,
        )


# ── Artifact builders ─────────────────────────────────────────────────────────

def _build_positions_artifact(snapshot: PositionSnapshot) -> dict:
    nav = snapshot.nav
    positions_out: dict = {}
    for sym, w in sorted(snapshot.positions.items(), key=lambda x: -x[1]):
        entry: dict = {
            "weight": round(w, 8),          # full precision — derived from market_value/nav
            "market_value_usd": round(w * nav, 2),
        }
        meta = snapshot.position_metadata.get(sym, {})
        if "quantity" in meta:
            entry["quantity"] = meta["quantity"]
        if "last_price_usd" in meta:
            entry["last_price_usd"] = round(meta["last_price_usd"], 4)
        positions_out[sym] = entry

    return {
        "as_of": snapshot.timestamp.isoformat(timespec="seconds"),
        "source": snapshot.source,
        "input_format": "amount_driven" if snapshot.position_metadata else "legacy_weight",
        "nav": nav,
        "invested_fraction": round(sum(snapshot.positions.values()), 8),
        "cash_fraction": round(snapshot.cash_fraction, 8),
        "cash_usd": round(snapshot.cash_fraction * nav, 2),
        "positions": positions_out,
    }


def _build_target_weights_artifact(
    target: TargetWeights,
    weighting_method: str = "",
    universe: list[str] | None = None,
) -> dict:
    weight_sum = sum(target.weights.values())
    data: dict = {
        "strategy_name": target.strategy_name,
        "market_data_date": str(target.rebalance_date),
        "source_record_id": target.source_record_id,
        "weighting_method": weighting_method,
        "universe": universe or [],
        "weights": {
            sym: round(w, 6)
            for sym, w in sorted(target.weights.items(), key=lambda x: -x[1])
        },
        "weight_sum": round(weight_sum, 6),
        "implicit_cash_fraction": round(max(0.0, 1.0 - weight_sum), 6),
    }
    return data


def _build_rebalance_plan_artifact(
    plan: OrderPlan,
    result: ExecutionResult,
    run_id: str,
    needs_rebalance: bool,
) -> dict:
    snapshot = plan.decision.snapshot
    nav = snapshot.nav

    orders = [
        {
            "symbol": o.symbol,
            "side": str(o.side),
            "current_weight": round(snapshot.positions.get(o.symbol, 0.0), 6),
            "target_weight": round(o.target_weight, 6),
            "delta_weight": round(o.delta_weight, 6),
            "current_value_usd": round(snapshot.positions.get(o.symbol, 0.0) * nav, 2),
            "target_value_usd": round(o.target_weight * nav, 2),
            "delta_value_usd": round(o.delta_weight * nav, 2),
        }
        for o in sorted(plan.orders, key=lambda x: x.symbol)
    ]

    actionable_symbols = {d.symbol for d in plan.decision.actionable}
    all_diffs = [
        {
            "symbol": d.symbol,
            "current_weight": round(d.current_weight, 6),
            "target_weight": round(d.target_weight, 6),
            "delta_weight": round(d.delta_weight, 6),
            "action": str(d.side) if d.symbol in actionable_symbols else "hold",
        }
        for d in plan.decision.all_diffs
    ]

    buy_count = sum(1 for o in plan.orders if str(o.side) == "buy")
    sell_count = len(plan.orders) - buy_count
    est_cost_usd = plan.total_turnover * plan.estimated_cost_bps / 10_000 * nav
    buy_value = sum(o.delta_weight * nav for o in plan.orders if o.delta_weight > 0)
    sell_value = sum(-o.delta_weight * nav for o in plan.orders if o.delta_weight < 0)
    principal_cash_needed = max(0.0, buy_value - sell_value)
    est_total_cash_needed = principal_cash_needed + est_cost_usd

    return {
        "run_id": run_id,
        "plan_id": plan.plan_id,
        "created_at": plan.created_at.isoformat(timespec="seconds"),
        "strategy_name": plan.decision.target.strategy_name,
        "market_data_date": str(plan.decision.target.rebalance_date),
        "mode": result.adapter_mode,
        "dry_run": not plan.approved,
        "needs_rebalance": needs_rebalance,
        "nav": nav,
        "summary": {
            "order_count": len(plan.orders),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": len(plan.decision.all_diffs) - len(plan.orders),
            "total_turnover": round(plan.total_turnover, 6),
            "buy_value_usd": round(buy_value, 2),
            "sell_proceeds_usd": round(sell_value, 2),
            "estimated_cost_bps": plan.estimated_cost_bps,
            "estimated_cost_usd": round(est_cost_usd, 2),
            "principal_cash_needed_usd": round(principal_cash_needed, 2),
            "est_total_cash_needed_usd": round(est_total_cash_needed, 2),
            "available_cash_usd": round(snapshot.cash_fraction * nav, 2),
        },
        "orders": orders,
        "all_position_diffs": all_diffs,
        "risk_check_passed": result.risk_check.passed if result.risk_check else None,
        "success": result.success,
    }


def _build_risk_check_artifact(
    result: ExecutionResult,
    plan: OrderPlan,
    snapshot: PositionSnapshot,
    target: TargetWeights,
    service: RebalanceService,
    run_count: int = 1,
) -> dict:
    checks: list[dict] = []
    nav = snapshot.nav

    # 1. Kill switch
    ks_ok = not service.kill_switch
    checks.append({
        "check": "kill_switch",
        "severity": "error",
        "passed": ks_ok,
        "detail": (
            "kill_switch=False -- execution gate open"
            if ks_ok
            else "BLOCKED: kill_switch=True -- all execution permanently halted"
        ),
    })

    # 2. Stale market data — compare today against the last market close used for signals.
    # target.rebalance_date is the last price bar date, not the time this object was created.
    market_data_date = target.rebalance_date
    market_data_age = (datetime.now().date() - market_data_date).days
    stale_ok = market_data_age <= service.stale_signal_days
    checks.append({
        "check": "stale_signal",
        "severity": "warning",
        "passed": stale_ok,
        "market_data_date": str(market_data_date),
        "market_data_age_days": market_data_age,
        "limit_days": service.stale_signal_days,
        "detail": (
            f"Market data {market_data_age}d old (last close: {market_data_date}) -- within {service.stale_signal_days}d limit"
            if stale_ok
            else f"WARNING: Market data {market_data_age}d old (last close: {market_data_date}) -- exceeds {service.stale_signal_days}d limit, verify signal is still current"
        ),
    })

    # 3. Position reconciliation — two-sided: total should be in [90%, 105%].
    # Under 90% may indicate untracked positions; over 105% is likely an accounting error.
    total_alloc = sum(snapshot.positions.values())
    implied_total = total_alloc + snapshot.cash_fraction
    if implied_total > 1.05:
        recon_ok = False
        detail = (
            f"WARNING: positions ({total_alloc:.2%}) + cash ({snapshot.cash_fraction:.2%})"
            f" = {implied_total:.2%} > 105% -- possible double-counting or accounting error"
        )
    elif implied_total < 0.90:
        recon_ok = False
        detail = (
            f"WARNING: positions ({total_alloc:.2%}) + cash ({snapshot.cash_fraction:.2%})"
            f" = {implied_total:.2%} < 90% -- verify all holdings are accounted for"
        )
    else:
        recon_ok = True
        detail = (
            f"positions ({total_alloc:.2%}) + cash ({snapshot.cash_fraction:.2%})"
            f" = {implied_total:.2%} (within normal range)"
        )
    checks.append({
        "check": "position_reconciliation",
        "severity": "warning",
        "passed": recon_ok,
        "value": round(implied_total, 6),
        "detail": detail,
    })

    # 4. Cash sufficiency — two-part check.
    # Hard gate: principal coverage only (buy notional net of sell proceeds).
    # Soft note: estimated fee buffer (whether fee is also covered by available cash).
    # Rationale: fees are approximate and may be deducted from sale proceeds or inside
    # the broker's fill mechanism; blocking on fee coverage alone would be too strict.
    fee_shortfall_usd = 0.0
    if plan.total_turnover == 0:
        cash_ok = True
        cash_detail = "No orders -- cash check not applicable"
    else:
        buy_value = sum(o.delta_weight * nav for o in plan.orders if o.delta_weight > 0)
        sell_value = sum(-o.delta_weight * nav for o in plan.orders if o.delta_weight < 0)
        est_cost_usd = plan.total_turnover * service.cost_bps / 10_000 * nav
        available_cash = snapshot.cash_fraction * nav
        net_principal = max(0.0, buy_value - sell_value)
        net_with_fee = net_principal + est_cost_usd
        # Hard gate: principal only (0.01 float-rounding tolerance)
        principal_covered = net_principal <= available_cash + 0.01
        fee_covered = net_with_fee <= available_cash + 0.01
        cash_ok = principal_covered
        fee_shortfall_usd = round(max(0.0, net_with_fee - available_cash), 2)

        sell_note = f" - sell proceeds ${sell_value:,.2f}" if sell_value > 0 else ""
        if not principal_covered:
            shortage = net_principal - available_cash
            cash_detail = (
                f"Principal coverage: FAIL -- need ${net_principal:,.2f}{sell_note}"
                f" but only ${available_cash:,.2f} available (short by ${shortage:,.2f})."
            )
        elif fee_covered:
            cash_detail = (
                f"Principal coverage: PASS -- buy ${buy_value:,.2f}{sell_note}"
                f" <= available ${available_cash:,.2f}. "
                f"Estimated fee buffer: covered"
                f" (total ${net_with_fee:,.2f} <= available ${available_cash:,.2f})."
            )
        else:
            cash_detail = (
                f"Principal coverage: PASS -- buy ${buy_value:,.2f}{sell_note}"
                f" <= available ${available_cash:,.2f}. "
                f"Estimated fee buffer: SHORTFALL -- buy + est. fee"
                f" ${net_with_fee:,.2f} vs available ${available_cash:,.2f}"
                f" (short by ${fee_shortfall_usd:,.2f})."
            )

    checks.append({
        "check": "cash_sufficiency",
        "severity": "warning",
        "passed": cash_ok,
        "detail": cash_detail,
        "fee_shortfall_usd": fee_shortfall_usd,
    })

    # 5. Duplicate execution guard.
    # In dry_run mode the service never updates its fingerprint, so dup_skipped is always
    # False.  We use run_count to distinguish first vs subsequent shadow runs.
    dup_skipped = any("[DUPLICATE]" in e for e in result.log_entries)
    if dup_skipped:
        dup_detail = "SKIPPED: Same plan as last execution -- no changes detected"
    elif run_count == 1:
        dup_detail = "No duplicate -- no prior shadow run found (first run for this session)"
    else:
        dup_detail = "No duplicate -- plan fingerprint differs from previous shadow run"
    checks.append({
        "check": "duplicate_guard",
        "severity": "info",
        "passed": not dup_skipped,
        "detail": dup_detail,
    })

    # 6. Minimum order threshold — result-oriented.
    filtered_count = len(plan.decision.all_diffs) - len(plan.decision.actionable)
    if filtered_count == 0:
        mot_detail = (
            f"All diffs are at or above the {service.min_trade_size:.1%} minimum threshold"
            f" -- no trades filtered"
        )
    else:
        mot_detail = (
            f"{filtered_count} sub-threshold diff(s) below {service.min_trade_size:.1%} filtered;"
            f" {len(plan.decision.actionable)} actionable trade(s) remain"
        )
    checks.append({
        "check": "min_order_threshold",
        "severity": "info",
        "passed": True,
        "threshold": service.min_trade_size,
        "filtered_count": filtered_count,
        "detail": mot_detail,
    })

    # 7-9. Hard risk checks from RiskCheckResult (max_position_size, max_turnover, max_orders).
    if result.risk_check:
        violation_rules = {v.rule for v in result.risk_check.violations}

        mps_ok = "max_position_size" not in violation_rules
        mps_msgs = [v.message for v in result.risk_check.violations if v.rule == "max_position_size"]
        checks.append({
            "check": "max_position_size",
            "severity": "error",
            "passed": mps_ok,
            "limit": service.risk.max_position_size,
            "detail": (
                f"All target positions <= {service.risk.max_position_size:.0%}"
                if mps_ok
                else f"VIOLATION: {'; '.join(mps_msgs)}"
            ),
        })

        mt_ok = "max_turnover" not in violation_rules
        checks.append({
            "check": "max_turnover",
            "severity": "error",
            "passed": mt_ok,
            "value": round(plan.total_turnover, 4),
            "limit": service.max_turnover,
            "detail": (
                f"Turnover {plan.total_turnover:.2%} <= {service.max_turnover:.0%} limit"
                if mt_ok
                else f"VIOLATION: turnover {plan.total_turnover:.2%} > {service.max_turnover:.0%} limit"
            ),
        })

        moc_ok = "max_order_count" not in violation_rules
        checks.append({
            "check": "max_order_count",
            "severity": "error",
            "passed": moc_ok,
            "value": len(plan.orders),
            "limit": service.max_orders,
            "detail": (
                f"{len(plan.orders)} orders <= {service.max_orders} limit"
                if moc_ok
                else f"VIOLATION: {len(plan.orders)} orders > {service.max_orders} limit"
            ),
        })

    hard_failures = [c["check"] for c in checks if not c["passed"] and c["severity"] == "error"]
    soft_warnings = [c["check"] for c in checks if not c["passed"] and c["severity"] == "warning"]

    return {
        "all_passed": len(hard_failures) == 0,
        "checks": checks,
        "violations": hard_failures,
        "warnings": soft_warnings,
        "risk_check_result": result.risk_check.model_dump() if result.risk_check else None,
    }


def _build_summary_markdown(
    run_id: str,
    target: TargetWeights,
    snapshot: PositionSnapshot,
    plan: OrderPlan,
    result: ExecutionResult,
    risk_data: dict,
    needs_rebalance: bool,
    ts: datetime,
    weighting_method: str = "",
    universe: list[str] | None = None,
    universe_type: str = "",
    latest_prices: dict[str, float] | None = None,
) -> str:
    nav = snapshot.nav
    buy_orders = [o for o in plan.orders if str(o.side) == "buy"]
    sell_orders = [o for o in plan.orders if str(o.side) == "sell"]
    weight_sum = sum(target.weights.values())
    traded_notional = plan.total_turnover * nav
    est_cost = traded_notional * plan.estimated_cost_bps / 10_000

    # Enhancement 1: time fields
    market_data_date = target.rebalance_date
    calendar_age = (ts.date() - market_data_date).days
    trading_age = _count_bdays(market_data_date, ts.date())

    has_cash_warning = "cash_sufficiency" in risk_data.get("warnings", [])
    _cash_check = next((c for c in risk_data["checks"] if c["check"] == "cash_sufficiency"), {})
    has_fee_shortfall = _cash_check.get("fee_shortfall_usd", 0.0) > 0.01

    if not result.success:
        recommendation = "BLOCKED -- execution blocked (kill switch or risk violation). Review risk checks."
        action = "**Do NOT execute** until violations are resolved."
    elif not needs_rebalance:
        recommendation = "No rebalance required -- portfolio already matches target allocation."
        action = "No action needed this cycle."
    elif has_cash_warning:
        recommendation = (
            f"Strategy recommends {len(plan.orders)} trade adjustment(s), "
            "subject to resolving the cash shortfall noted in Section D."
        )
        action = (
            "Verify your account has sufficient buying power (see Section D cash note) "
            "before placing orders, then execute manually at your broker."
        )
    else:
        recommendation = f"Strategy recommends {len(plan.orders)} trade adjustment(s)."
        action = "Review the order plan, then execute manually at your broker if approved."

    # ── Section A: Basic Information ──────────────────────────────────────────
    L: list[str] = [
        "# Shadow Execution Summary",
        "",
        "## A. Basic Information",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Strategy | `{target.strategy_name}` |",
    ]

    if weighting_method:
        L.append(f"| Weighting Method | {weighting_method} |")

    if universe:
        if universe_type:
            L.append(f"| Universe Type | {universe_type} |")
        L.append(f"| Universe Members | {', '.join(universe)} |")

    age_str = f"{trading_age} trading day(s) / {calendar_age} calendar day(s)"
    L += [
        f"| US Market Close Date | {market_data_date} |",
        f"| Market Data Age | {age_str} |",
        f"| Run At (Local Time) | {ts.strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| Shadow Run ID | `{run_id}` |",
        f"| Current NAV | ${nav:,.2f} |",
        f"| Execution Mode | `{result.adapter_mode}` (dry-run, no orders submitted) |",
    ]

    # ── Section B: Current Positions ──────────────────────────────────────────
    L += [
        "",
        "## B. Current Positions",
        "",
    ]

    if snapshot.positions:
        has_qty = any(
            "quantity" in snapshot.position_metadata.get(sym, {})
            for sym in snapshot.positions
        )
        if has_qty:
            # Amount-driven format: show quantity, price, exact market value
            L += [
                "| Symbol | Qty | Last Price | Market Value | Weight |",
                "|--------|----:|----------:|-------------:|-------:|",
            ]
            for sym, w in sorted(snapshot.positions.items(), key=lambda x: -x[1]):
                mv = w * nav          # always use w × nav as display value
                meta = snapshot.position_metadata.get(sym, {})
                qty = meta.get("quantity", "—")
                price_str = f"${meta['last_price_usd']:,.2f}" if "last_price_usd" in meta else "—"
                L.append(f"| {sym} | {qty} | {price_str} | ${mv:,.2f} | {w:.2%} |")
            L.append(
                f"| **CASH** | — | — | **${snapshot.cash_fraction * nav:,.2f}** "
                f"| **{snapshot.cash_fraction:.2%}** |"
            )
            L.append("")
            L.append(
                "_Market values are the source-of-truth inputs; displayed weights are derived from "
                "precise market_value_usd / nav_usd and may be rounded._"
            )
        else:
            # Legacy format: weight-fraction display
            L += [
                "| Symbol | Weight | Value (USD) |",
                "|--------|--------|-------------|",
            ]
            for sym, w in sorted(snapshot.positions.items(), key=lambda x: -x[1]):
                L.append(f"| {sym} | {w:.2%} | ${w * nav:,.0f} |")
            L.append(
                f"| **CASH** | **{snapshot.cash_fraction:.2%}** "
                f"| **${snapshot.cash_fraction * nav:,.0f}** |"
            )
    else:
        L += [
            "| Symbol | Weight | Value (USD) |",
            "|--------|--------|-------------|",
            f"| CASH | 100.00% | ${nav:,.0f} |",
            "",
            "_Portfolio is all-cash (first rebalance)._",
        ]

    # ── Section C: Target Positions ───────────────────────────────────────────
    L += [
        "",
        "## C. Target Positions",
        "",
        f"_Theoretical target weights (sum = {weight_sum:.2%} of NAV)._",
        "_See Section D for the cash/cost note before placing orders._",
        "",
        "| Symbol | Weight | Value (USD) |",
        "|--------|--------|-------------|",
    ]
    for sym, w in sorted(target.weights.items(), key=lambda x: -x[1]):
        L.append(f"| {sym} | {w:.2%} | ${w * nav:,.0f} |")
    implicit_cash = max(0.0, 1.0 - weight_sum)
    if implicit_cash > 0.001:
        L.append(f"| CASH (implicit buffer) | {implicit_cash:.2%} | ${implicit_cash * nav:,.0f} |")

    # ── Section D: Rebalance Plan ─────────────────────────────────────────────
    L += [
        "",
        "## D. Rebalance Plan",
        "",
    ]

    if not plan.orders:
        L += [
            "_No orders required. All position diffs are below the minimum trade "
            f"threshold ({plan.estimated_cost_bps:.0f} bps / {self_min_trade_size_display(plan)})._",
        ]
    else:
        L += [
            "| Symbol | Action | Current | Target | Delta | Delta $ |",
            "|--------|--------|---------|--------|-------|---------|",
        ]
        for o in sorted(plan.orders, key=lambda x: x.symbol):
            cur = snapshot.positions.get(o.symbol, 0.0)
            delta_usd = o.delta_weight * nav
            L.append(
                f"| {o.symbol} | **{str(o.side).upper()}** "
                f"| {cur:.2%} | {o.target_weight:.2%} "
                f"| {o.delta_weight:+.2%} | ${delta_usd:+,.2f} |"
            )

        # Turnover and cost
        buy_value = sum(o.delta_weight * nav for o in plan.orders if o.delta_weight > 0)
        sell_value = sum(-o.delta_weight * nav for o in plan.orders if o.delta_weight < 0)
        net_principal = max(0.0, buy_value - sell_value)
        net_with_fee = net_principal + est_cost
        available_cash = snapshot.cash_fraction * nav
        principal_covered = net_principal <= available_cash + 0.01
        fee_covered = net_with_fee <= available_cash + 0.01
        sell_note_md = f" - sell ${sell_value:,.2f}" if sell_value > 0 else ""

        L += [
            "",
            f"**Total Turnover:** {plan.total_turnover:.2%}  ",
            f"**Estimated Cost:** ~{_fmt_cost(est_cost)}  ({plan.estimated_cost_bps:.0f} bps × traded notional of ${traded_notional:,.2f})",
        ]

        # Cash note — two-part: principal coverage (hard gate) then fee buffer (soft)
        L += [""]
        if not principal_covered:
            shortage = net_principal - available_cash
            L += [
                "> **Cash note (principal):** FAIL -- buy"
                + (f" ${buy_value:,.2f}{sell_note_md} = ${net_principal:,.2f}" if sell_value > 0 else f" ${buy_value:,.2f}")
                + f" exceeds available ${available_cash:,.2f} (short by ${shortage:,.2f}).",
                "> Reduce order sizes or deposit additional funds before executing.",
            ]
        elif fee_covered:
            L += [
                "> **Cash note (principal):** PASS -- buy"
                + (f" ${buy_value:,.2f}{sell_note_md} = ${net_principal:,.2f}" if sell_value > 0 else f" ${buy_value:,.2f}")
                + f" <= available ${available_cash:,.2f}.  ",
                f"> **Estimated fee buffer:** covered -- total ${net_with_fee:,.2f} <= available ${available_cash:,.2f}.",
            ]
        else:
            fee_shortfall = net_with_fee - available_cash
            L += [
                "> **Cash note (principal):** PASS -- buy"
                + (f" ${buy_value:,.2f}{sell_note_md} = ${net_principal:,.2f}" if sell_value > 0 else f" ${buy_value:,.2f}")
                + f" <= available ${available_cash:,.2f}.  ",
                f"> **Estimated fee buffer:** SHORTFALL -- buy + est. fee ${net_with_fee:,.2f}"
                f" vs available ${available_cash:,.2f} (short by ${fee_shortfall:,.2f}).",
                f"> Est. fee ({_fmt_cost(est_cost)}) may be deducted from proceeds;"
                " verify your broker's fee settlement method.",
            ]

        # ── D.1 Cash Reserve Summary ───────────────────────────────────────────
        post_cost_nav = max(0.0, nav - est_cost)
        L += [
            "",
            "**D.1 Cash Reserve Summary**",
            "",
            "| Item | Amount |",
            "|------|--------|",
            f"| Current NAV | ${nav:,.2f} |",
            f"| Estimated Cost Buffer | ~{_fmt_cost(est_cost)} ({plan.estimated_cost_bps:.0f} bps × traded notional) |",
            f"| **Post-Cost Effective NAV** | **${post_cost_nav:,.2f}** |",
            "",
            "_Post-Cost Effective NAV = NAV minus estimated cost buffer (cash planning view only)._"
            " _Strategy target weights are sized against full NAV; this row does not affect order sizing._",
        ]

        # ── D.2 Human Execution Suggestion ─────────────────────────────────────
        buy_orders_sorted = sorted(
            [o for o in plan.orders if o.delta_weight > 0], key=lambda x: x.symbol
        )
        sell_orders_sorted = sorted(
            [o for o in plan.orders if o.delta_weight < 0], key=lambda x: x.symbol
        )
        has_prices = latest_prices is not None and any(
            o.symbol in latest_prices for o in plan.orders
        )
        L += [
            "",
            "**D.2 Human Execution Suggestion** "
            "_(manual trading aid only -- not formal strategy output)_",
            "",
            "**Three-tier price framework:**",
            "",
            "| Price Type | Definition | How to use |",
            "|------------|------------|------------|",
            f"| Planning Reference Price | Last US market close ({market_data_date}) | Notional and qty estimates below only |",
            "| Real-time Execution Price | Price shown at your broker at order submission | Use this to derive final share qty |",
            "| Actual Fill Price | Price your order actually filled at | Record in execution log template (see Section G) |",
            "",
            "> **Notional-first execution rule:** Target `Suggested Notional` as your order size.",
            "> At your broker, read the live ask (buy) or bid (sell) price, then compute:",
            "> `executable qty = floor(Suggested Notional / real-time price)`.",
            "> Do NOT mechanically submit the `Est. Qty` column -- it is a planning estimate",
            "> based on yesterday's close and will be wrong if the price has moved.",
        ]

        zero_qty_symbols: list[str] = []

        if buy_orders_sorted:
            if has_prices:
                L += [
                    "",
                    f"| Symbol | Action | Suggested Notional* | Planning Ref Price | Est. Qty (plan only) | Est. Residual |",
                    f"|--------|--------|--------------------:|-------------------:|---------------------:|--------------:|",
                ]
                total_residual = 0.0
                for o in buy_orders_sorted:
                    sug_notional = o.delta_weight * nav
                    price = latest_prices.get(o.symbol) if latest_prices else None
                    if price and price > 0:
                        qty = math.floor(sug_notional / price)
                        residual = sug_notional - qty * price
                        total_residual += residual
                        if qty == 0:
                            zero_qty_symbols.append(o.symbol)
                        L.append(
                            f"| {o.symbol} | BUY | ${sug_notional:,.2f} "
                            f"| ${price:,.2f} | {qty:,} | ${residual:,.2f} |"
                        )
                    else:
                        L.append(
                            f"| {o.symbol} | BUY | ${sug_notional:,.2f} "
                            f"| N/A | N/A | N/A |"
                        )
                L.append(
                    f"\n_* Suggested Notional = delta-weight × NAV = Delta $ in Section D."
                    f" Est. residual from floor-rounding at ref price: ~${total_residual:,.2f}._"
                )
                if zero_qty_symbols:
                    L += [
                        "",
                        f"> **Whole-share constraint:** {', '.join(zero_qty_symbols)} — "
                        f"the strategy-level notional exists but is below the cost of one share at the reference price. "
                        f"Est. Qty = 0; this trade is **not broker-executable** under whole-share constraints. "
                        f"To execute, you would need a notional override, accumulated residual cash, or fractional-share support.",
                    ]
            else:
                L += [
                    "",
                    "| Symbol | Action | Suggested Notional* |",
                    "|--------|--------|--------------------:|",
                ]
                for o in buy_orders_sorted:
                    sug_notional = o.delta_weight * nav
                    L.append(f"| {o.symbol} | BUY | ${sug_notional:,.2f} |")
                L.append(
                    "\n_* Suggested Notional = delta-weight × NAV = Delta $ in Section D."
                    " Reference prices unavailable -- calculate qty at your broker using real-time price._"
                )

        if sell_orders_sorted:
            L += [
                "",
                "| Symbol | Action | Sell Notional (planning est.) | Planning Ref Price |",
                "|--------|--------|------------------------------:|-------------------:|",
            ]
            for o in sell_orders_sorted:
                sell_notional = abs(o.delta_weight) * nav
                price = latest_prices.get(o.symbol) if latest_prices else None
                price_str = f"${price:,.2f}" if price else "N/A"
                L.append(
                    f"| {o.symbol} | SELL | ${sell_notional:,.0f} | {price_str} |"
                )

    # ── Section E: Risk Checks ────────────────────────────────────────────────
    L += [
        "",
        "## E. Risk Check Results",
        "",
        "| Check | Status | Detail |",
        "|-------|--------|--------|",
    ]
    for check in risk_data["checks"]:
        if check["passed"]:
            status = "PASS"
        elif check["severity"] == "error":
            status = "FAIL"
        elif check["severity"] == "warning":
            status = "WARN"
        else:
            status = "INFO"
        L.append(f"| `{check['check']}` | {status} | {check['detail']} |")

    if not risk_data["all_passed"]:
        parts = []
        if risk_data["violations"]:
            parts.append(f"{len(risk_data['violations'])} violation(s): `{'`, `'.join(risk_data['violations'])}`")
        if risk_data["warnings"]:
            parts.append(f"{len(risk_data['warnings'])} warning(s): `{'`, `'.join(risk_data['warnings'])}`")
        overall = f"**FAILED -- {'; '.join(parts)}**"
    elif risk_data["warnings"]:
        overall = (
            f"**PASSED WITH WARNINGS** -- "
            f"{len(risk_data['warnings'])} warning(s) require manual review: "
            f"`{'`, `'.join(risk_data['warnings'])}`"
        )
    else:
        overall = "**All checks passed.**"
    L += ["", f"Overall: {overall}"]

    # ── Section F: Human Review Summary ──────────────────────────────────────
    L += [
        "",
        "## F. Human Review Summary",
        "",
        f"**Recommendation:** {recommendation}",
        "",
        f"**Action:** {action}",
        "",
    ]

    if plan.orders:
        entering = [o.symbol for o in buy_orders if snapshot.positions.get(o.symbol, 0) < 0.005]
        exiting = [o.symbol for o in sell_orders if o.target_weight < 0.005]
        reweighting = [
            o.symbol for o in plan.orders
            if o.symbol not in entering and o.symbol not in exiting
        ]
        if entering:
            L.append(f"- **Entering (new positions):** {', '.join(entering)}")
        if exiting:
            L.append(f"- **Exiting (full liquidation):** {', '.join(exiting)}")
        if reweighting:
            L.append(f"- **Reweighting (partial change):** {', '.join(reweighting)}")
        if not sell_orders:
            L.append("- **Cash-only rebalance:** no sells required — all trades funded from available cash.")
        if zero_qty_symbols:
            executable_count = len(plan.orders) - len(zero_qty_symbols)
            L.append(
                f"- **Whole-share note:** strategy recommends {len(plan.orders)} trade adjustment(s); "
                f"{len(zero_qty_symbols)} ({', '.join(zero_qty_symbols)}) round(s) to 0 shares at the reference price — "
                f"not broker-executable under whole-share constraints. "
                f"{executable_count} broker-executable order(s); see D.2 for execution options on the remainder."
            )
        L.append("")

    if not result.success:
        L += [
            "> **BLOCKED:** Resolve the violations listed in Section E before executing.",
            "",
        ]
    elif not needs_rebalance:
        L += [
            "Current portfolio matches the target allocation"
            f" (all diffs below the {self_min_trade_size_display(plan)} minimum threshold).",
            "No action required this cycle.",
            "",
        ]
    else:
        L += [
            "> This is a **dry-run**. No orders have been or will be submitted automatically.",
            "> Review the plan in Section D, confirm the cash note, then place orders manually at your broker.",
            "",
        ]

    # ── Section G: Human Execution Rules ──────────────────────────────────────
    L += [
        "",
        "## G. Human Execution Rules",
        "",
        "_These rules are manual execution guidelines only._"
        " _They do not modify target weights, order sizing, or strategy logic._",
        "",
        "### G.1 Price Deviation Handling",
        "",
        "Before placing each order, compare the **real-time price** at your broker"
        " to the **Planning Reference Price** in Section D.2."
        " Apply the rule for the matching deviation band:",
        "",
        "**BUY orders** -- deviation = `(real_time / ref_price) - 1`",
        "",
        "| Deviation | Rule |",
        "|-----------|------|",
        "| <= +1% | **PROCEED** -- execute at real-time price |",
        "| +1% to +2% | **REVIEW** -- price has moved up; decide whether to proceed or reduce size |",
        "| > +2% | **DEFER** -- exceeds acceptable deviation; wait or skip this cycle |",
        "",
        "**SELL orders** -- deviation = `(real_time / ref_price) - 1`",
        "",
        "| Deviation | Rule |",
        "|-----------|------|",
        "| >= -1% | **PROCEED** -- execute at real-time price |",
        "| -1% to -2% | **REVIEW** -- price has dropped; decide whether to proceed |",
        "| < -2% | **DEFER** -- exceeds acceptable deviation; wait or skip this cycle |",
    ]

    # Per-order deviation threshold table (only when prices are available)
    orders_with_prices = [
        o for o in plan.orders
        if latest_prices and latest_prices.get(o.symbol)
    ]
    if orders_with_prices:
        L += [
            "",
            "### G.2 Per-Order Deviation Limits",
            "",
            "| Symbol | Action | Planning Ref | PROCEED if | REVIEW if | DEFER if |",
            "|--------|--------|-------------:|-----------:|----------:|---------:|",
        ]
        for o in sorted(orders_with_prices, key=lambda x: x.symbol):
            ref = latest_prices[o.symbol]  # type: ignore[index]
            side = str(o.side)
            if side == "buy":
                proceed = f"<= ${ref * 1.01:,.2f}"
                review  = f"${ref * 1.01:,.2f} -- ${ref * 1.02:,.2f}"
                defer   = f"> ${ref * 1.02:,.2f}"
            else:
                proceed = f">= ${ref * 0.99:,.2f}"
                review  = f"${ref * 0.98:,.2f} -- ${ref * 0.99:,.2f}"
                defer   = f"< ${ref * 0.98:,.2f}"
            L.append(
                f"| {o.symbol} | {side.upper()} | ${ref:,.2f}"
                f" | {proceed} | {review} | {defer} |"
            )

    L += [
        "",
        "### G.3 Execution Checklist",
        "",
        "1. Confirm all risk checks in Section E have passed (no FAIL).",
        "2. For each order: check the real-time price vs Planning Ref Price (G.2).",
        "3. If PROCEED: enter `floor(Suggested Notional / real-time price)` shares.",
        "4. If REVIEW: use judgment; document reasoning in execution log.",
        "5. If DEFER: skip this order; document in execution log.",
        "6. After all orders: fill in `manual_execution_log_template.json`"
        f" in the run artifact directory.",
        "",
        "### G.4 Actual Fill Price Recording",
        "",
        "After execution, open the template and save your fills:",
        "",
        f"    {run_id}/manual_execution_log_template.json",
        "",
        "Fields to complete: `real_time_quote_seen`, `actual_fill_price`,"
        " `actual_quantity`, `actual_notional_usd`, `execution_time`,"
        " `deviation_from_ref_pct`, `rule_applied`, `notes`.",
    ]

    L += [
        "",
        "---",
        "",
        f"*Generated by quant-stack shadow execution | {ts.strftime('%Y-%m-%d %H:%M:%S')}*  ",
        f"*Run ID: `{run_id}`*",
    ]

    return "\n".join(L)


def self_min_trade_size_display(plan: OrderPlan) -> str:
    """Return a human-readable min trade size string from the plan context."""
    if not plan.orders:
        return "0.5%"
    return f"{min(abs(o.delta_weight) for o in plan.decision.actionable):.1%}" if plan.decision.actionable else "0.5%"


def _build_execution_log_template(
    target: TargetWeights,
    snapshot: PositionSnapshot,
    plan: OrderPlan,
    run_id: str,
    ts: datetime,
    latest_prices: dict[str, float] | None = None,
) -> dict:
    """Build a fill-in-the-blank template for recording actual manual fills."""
    entries = []
    for o in sorted(plan.orders, key=lambda x: x.symbol):
        ref_price = (latest_prices or {}).get(o.symbol)
        planned_notional = round(abs(o.delta_weight) * snapshot.nav, 2)
        entries.append({
            "symbol": o.symbol,
            "action": str(o.side).upper(),
            "planning_ref_price": round(ref_price, 4) if ref_price else None,
            "planned_notional_usd": planned_notional,
            "planned_delta_weight": round(o.delta_weight, 6),
            # ── Fill in after execution ──────────────────────────────────────
            "real_time_quote_seen": None,
            "actual_fill_price": None,
            "actual_quantity": None,
            "actual_notional_usd": None,
            "execution_time": None,
            "deviation_from_ref_pct": None,
            "rule_applied": None,
            "notes": None,
        })

    return {
        "_instructions": [
            "Fill in every null field after manual execution.",
            "Save a completed copy as 'manual_execution_log.json' in this directory.",
            "real_time_quote_seen  : price you observed at your broker before submitting",
            "actual_fill_price     : price your order filled at (from broker confirmation)",
            "actual_quantity       : whole shares actually bought or sold",
            "actual_notional_usd   : actual_fill_price x actual_quantity",
            "deviation_from_ref_pct: (actual_fill_price / planning_ref_price - 1) * 100",
            "rule_applied          : PROCEED / REVIEW / DEFER (per Section G of summary)",
            "notes                 : any context (market conditions, partial fills, etc.)",
        ],
        "strategy_name": target.strategy_name,
        "signal_date": str(target.rebalance_date),
        "run_id": run_id,
        "run_at": ts.isoformat(timespec="seconds"),
        "nav": snapshot.nav,
        "orders": entries,
    }
