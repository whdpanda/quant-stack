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
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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

    def run(
        self,
        target: TargetWeights,
        snapshot: PositionSnapshot,
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
            target:   TargetWeights from the research layer.
            snapshot: Current portfolio snapshot from positions loader or paper adapter.

        Returns:
            ShadowRunResult with all artifact paths and a human-readable summary.
        """
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
            strategy=target.strategy_name,
            rebalance_date=str(target.rebalance_date),
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
        tw_data = _build_target_weights_artifact(target)
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
        risk_data = _build_risk_check_artifact(result, plan, snapshot, target, self.service)
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
        )
        summary_path = run_dir / "shadow_execution_summary.md"
        summary_path.write_text(summary_text, encoding="utf-8")
        artifacts["shadow_execution_summary"] = summary_path
        log_event("artifact_written", file="shadow_execution_summary.md")

        # ── 7. execution_log.jsonl ─────────────────────────────────────────
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
            shutil.rmtree(latest_dir)
        shutil.copytree(run_dir, latest_dir)

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
    return {
        "as_of": snapshot.timestamp.isoformat(timespec="seconds"),
        "source": snapshot.source,
        "nav": nav,
        "invested_fraction": round(sum(snapshot.positions.values()), 6),
        "cash_fraction": round(snapshot.cash_fraction, 6),
        "cash_value_usd": round(snapshot.cash_fraction * nav, 2),
        "positions": {
            sym: {
                "weight": round(w, 6),
                "value_usd": round(w * nav, 2),
            }
            for sym, w in sorted(snapshot.positions.items(), key=lambda x: -x[1])
        },
    }


def _build_target_weights_artifact(target: TargetWeights) -> dict:
    weight_sum = sum(target.weights.values())
    return {
        "strategy_name": target.strategy_name,
        "rebalance_date": str(target.rebalance_date),
        "generated_at": target.generated_at.isoformat(timespec="seconds"),
        "source_record_id": target.source_record_id,
        "weights": {
            sym: round(w, 6)
            for sym, w in sorted(target.weights.items(), key=lambda x: -x[1])
        },
        "weight_sum": round(weight_sum, 6),
        "implicit_cash_fraction": round(max(0.0, 1.0 - weight_sum), 6),
    }


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

    return {
        "run_id": run_id,
        "plan_id": plan.plan_id,
        "created_at": plan.created_at.isoformat(timespec="seconds"),
        "strategy_name": plan.decision.target.strategy_name,
        "rebalance_date": str(plan.decision.target.rebalance_date),
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
            "estimated_cost_bps": plan.estimated_cost_bps,
            "estimated_cost_usd": round(est_cost_usd, 2),
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
) -> dict:
    checks: list[dict] = []

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

    # 2. Stale signal
    age_days = (datetime.now() - target.generated_at).total_seconds() / 86400
    stale_ok = age_days <= service.stale_signal_days
    checks.append({
        "check": "stale_signal",
        "severity": "warning",
        "passed": stale_ok,
        "value_days": round(age_days, 2),
        "limit_days": service.stale_signal_days,
        "detail": (
            f"Signal age {age_days:.1f}d <= {service.stale_signal_days}d limit"
            if stale_ok
            else f"WARNING: Signal age {age_days:.1f}d > {service.stale_signal_days}d -- verify freshness"
        ),
    })

    # 3. Position reconciliation
    total_alloc = sum(snapshot.positions.values())
    implied_total = total_alloc + snapshot.cash_fraction
    recon_ok = implied_total <= 1.05
    checks.append({
        "check": "position_reconciliation",
        "severity": "warning",
        "passed": recon_ok,
        "value": round(implied_total, 6),
        "limit": 1.05,
        "detail": (
            f"positions ({total_alloc:.2%}) + cash ({snapshot.cash_fraction:.2%}) = {implied_total:.2%} <= 105%"
            if recon_ok
            else f"WARNING: positions + cash = {implied_total:.2%} > 105% -- possible accounting error"
        ),
    })

    # 4. Cash buffer
    est_fee = plan.total_turnover * service.cost_bps / 10_000
    cash_ok = plan.total_turnover == 0 or snapshot.cash_fraction >= est_fee
    checks.append({
        "check": "cash_buffer",
        "severity": "warning",
        "passed": cash_ok,
        "cash_fraction": round(snapshot.cash_fraction, 6),
        "fee_estimate_fraction": round(est_fee, 6),
        "detail": (
            f"cash {snapshot.cash_fraction:.3%} >= fee estimate {est_fee:.3%}"
            if cash_ok
            else f"WARNING: cash {snapshot.cash_fraction:.3%} < fee estimate {est_fee:.3%} -- consider adding a cash buffer"
        ),
    })

    # 5. Duplicate execution guard
    dup_skipped = any("[DUPLICATE]" in e for e in result.log_entries)
    checks.append({
        "check": "duplicate_guard",
        "severity": "info",
        "passed": not dup_skipped,
        "detail": (
            "No duplicate -- plan fingerprint differs from last execution"
            if not dup_skipped
            else "SKIPPED: Same fingerprint as last execution -- no rebalance needed"
        ),
    })

    # 6. Minimum order threshold
    checks.append({
        "check": "min_order_threshold",
        "severity": "info",
        "passed": True,
        "threshold": service.min_trade_size,
        "detail": f"Diffs < {service.min_trade_size:.1%} filtered automatically (reduces noise trading)",
    })

    # 7–9. Hard risk checks from RiskCheckResult
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
) -> str:
    nav = snapshot.nav
    buy_orders = [o for o in plan.orders if str(o.side) == "buy"]
    sell_orders = [o for o in plan.orders if str(o.side) == "sell"]
    est_cost = plan.total_turnover * plan.estimated_cost_bps / 10_000 * nav

    if not result.success:
        recommendation = "BLOCKED -- execution blocked (kill switch or risk violation). Review risk checks."
        action = "**Do NOT execute** until violations are resolved."
    elif not needs_rebalance:
        recommendation = "No rebalance required -- portfolio already matches target allocation."
        action = "No action needed this cycle."
    else:
        recommendation = f"Rebalance recommended -- {len(plan.orders)} order(s) to execute."
        action = "Review the order plan, then execute manually at your broker if approved."

    L: list[str] = [
        "# Shadow Execution Summary",
        "",
        "## A. Basic Information",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Strategy | `{target.strategy_name}` |",
        f"| Rebalance Date | {target.rebalance_date} |",
        f"| Signal Generated At | {target.generated_at.strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| Shadow Run ID | `{run_id}` |",
        f"| Run At | {ts.strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| Current NAV | ${nav:,.2f} |",
        f"| Execution Mode | `{result.adapter_mode}` (dry-run, no orders submitted) |",
        "",
        "## B. Current Positions",
        "",
    ]

    if snapshot.positions:
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

    L += [
        "",
        "## C. Target Positions",
        "",
        "| Symbol | Weight | Value (USD) |",
        "|--------|--------|-------------|",
    ]
    for sym, w in sorted(target.weights.items(), key=lambda x: -x[1]):
        L.append(f"| {sym} | {w:.2%} | ${w * nav:,.0f} |")
    implicit_cash = max(0.0, 1.0 - sum(target.weights.values()))
    if implicit_cash > 0.001:
        L.append(f"| CASH (implicit buffer) | {implicit_cash:.2%} | ${implicit_cash * nav:,.0f} |")

    L += [
        "",
        "## D. Rebalance Plan",
        "",
    ]

    if not plan.orders:
        L.append(
            "_No orders required. All position diffs are below the minimum trade "
            f"threshold ({plan.estimated_cost_bps:.0f} bps)._"
        )
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
                f"| {o.delta_weight:+.2%} | ${delta_usd:+,.0f} |"
            )
        L += [
            "",
            f"**Total Turnover:** {plan.total_turnover:.2%}  ",
            f"**Estimated Cost:** ~${est_cost:,.0f} "
            f"({plan.estimated_cost_bps:.0f} bps of NAV)",
        ]

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

    if risk_data["all_passed"]:
        overall = "**All checks passed.**"
    else:
        parts = []
        if risk_data["violations"]:
            parts.append(f"{len(risk_data['violations'])} violation(s): `{'`, `'.join(risk_data['violations'])}`")
        if risk_data["warnings"]:
            parts.append(f"{len(risk_data['warnings'])} warning(s): `{'`, `'.join(risk_data['warnings'])}`")
        overall = f"**Issues found — {'; '.join(parts)}**"
    L += ["", f"Overall: {overall}"]

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
        L.append("")

    if not result.success:
        L += [
            "> **BLOCKED:** Resolve the violations listed in Section E before executing.",
            "",
        ]
    elif not needs_rebalance:
        L += [
            "Current portfolio matches the target allocation (all diffs below the minimum "
            "trade threshold). No action required this cycle.",
            "",
        ]
    else:
        L += [
            "> This is a **dry-run**. No orders have been or will be submitted automatically.",
            "> Review the plan in Section D, then place orders manually at your broker.",
            "",
        ]

    L += [
        "---",
        "",
        f"*Generated by quant-stack shadow execution | {ts.strftime('%Y-%m-%d %H:%M:%S')}*  ",
        f"*Run ID: `{run_id}`*",
    ]

    return "\n".join(L)
