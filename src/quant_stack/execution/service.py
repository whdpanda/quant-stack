"""RebalanceService — research-to-execution orchestration.

Responsibilities (in order):
  1. Kill-switch gate     — hard-stop all execution if set
  2. Stale-signal check   — warn if target weights are too old
  3. Cash-buffer check    — warn if snapshot has insufficient cash for fees
  4. Position diff        — compute current vs target deltas per symbol
  5. Min-trade-size filter— drop diffs below threshold
  6. Build OrderPlan      — convert actionable diffs to OrderIntents
  7. Risk checks          — validate against RiskConfig hard limits
  8. Duplicate guard      — no-op if plan is identical to the last executed one
  9. Adapter execution    — delegate to DryRun / Paper / Lean adapter
 10. Artifact persistence — write order plan + log to execution_artifacts/

Safety defaults
───────────────
``dry_run=True`` is the default.  Nothing reaches an adapter's "real"
execution path unless the caller explicitly sets ``dry_run=False``.
``kill_switch=False`` by default; set True to block all execution permanently
(e.g., on a circuit-breaker event).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from quant_stack.core.config import RiskConfig
from quant_stack.execution.domain import (
    ExecutionResult,
    OrderIntent,
    OrderPlan,
    PositionDiff,
    PositionSnapshot,
    RebalanceDecision,
    RiskCheckResult,
    RiskViolation,
    TargetWeights,
)


# ── Risk checks ────────────────────────────────────────────────────────────────

def check_order_plan(
    plan: OrderPlan,
    risk: RiskConfig,
    max_turnover: float = 1.5,
    max_orders: int = 20,
) -> RiskCheckResult:
    """Run all hard risk rules against an OrderPlan.

    Rules checked:
      max_position_size  (from RiskConfig)  — per-symbol target weight limit
      max_turnover                           — total |delta| this cycle
      max_order_count                        — number of order intents

    Returns a RiskCheckResult with ``passed=True`` iff all rules are satisfied.
    """
    violations: list[RiskViolation] = []

    # Rule 1: no single target weight exceeds RiskConfig.max_position_size
    for order in plan.orders:
        if order.target_weight > risk.max_position_size:
            violations.append(RiskViolation(
                rule="max_position_size",
                value=round(order.target_weight, 4),
                limit=risk.max_position_size,
                message=(
                    f"{order.symbol}: target_weight={order.target_weight:.2%}"
                    f" > max_position_size={risk.max_position_size:.2%}"
                ),
            ))

    # Rule 2: total one-way turnover within one cycle
    if plan.total_turnover > max_turnover:
        violations.append(RiskViolation(
            rule="max_turnover",
            value=round(plan.total_turnover, 4),
            limit=max_turnover,
            message=(
                f"total_turnover={plan.total_turnover:.2%} > max={max_turnover:.2%}"
                " — unusually large rebalance; verify positions and signal freshness"
            ),
        ))

    # Rule 3: order count guard
    if len(plan.orders) > max_orders:
        violations.append(RiskViolation(
            rule="max_order_count",
            value=float(len(plan.orders)),
            limit=float(max_orders),
            message=f"order count {len(plan.orders)} > max {max_orders}",
        ))

    return RiskCheckResult(passed=len(violations) == 0, violations=violations)


# ── Duplicate-execution guard ─────────────────────────────────────────────────

def _plan_fingerprint(plan: OrderPlan) -> str:
    """Stable hash of the plan's weights for duplicate detection."""
    key = json.dumps(
        {o.symbol: round(o.target_weight, 4) for o in plan.orders},
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── RebalanceService ──────────────────────────────────────────────────────────

class RebalanceService:
    """Orchestrate one rebalance cycle end-to-end.

    The service is stateless per-call except for duplicate-execution tracking.
    All inputs are passed explicitly so the same service instance can be reused
    across multiple rebalance cycles.

    Parameters
    ----------
    adapter:
        One of DryRunExecutionAdapter, PaperExecutionAdapter, or
        LeanExecutionAdapter (from quant_stack.execution.adapters).
    risk:
        Hard risk limits from AppConfig.execution.risk.  Defaults to
        RiskConfig() which uses the project-wide defaults.
    dry_run:
        If True (default), execution is fully suppressed: plan is built,
        risk-checked, and logged, but the adapter performs no real action.
    kill_switch:
        If True, ALL execution is blocked regardless of dry_run.
    stale_signal_days:
        Warn if target weights were generated more than N days ago.
    min_trade_size:
        Minimum |delta_weight| to include an order.  Diffs smaller than
        this are skipped (reduces noise trading).
    max_turnover:
        Hard limit on total one-way turnover per cycle (risk guard).
    max_orders:
        Hard limit on number of orders per cycle (risk guard).
    cost_bps:
        Informational cost estimate for the execution report.  Does not
        affect order logic.
    artifacts_dir:
        Directory for order plan JSON and execution log files.
    """

    def __init__(
        self,
        adapter: Any,
        risk: RiskConfig | None = None,
        dry_run: bool = True,
        kill_switch: bool = False,
        stale_signal_days: int = 5,
        min_trade_size: float = 0.005,
        max_turnover: float = 1.5,
        max_orders: int = 20,
        cost_bps: float = 20.0,
        artifacts_dir: str | Path = "./execution_artifacts",
    ) -> None:
        self.adapter = adapter
        self.risk = risk or RiskConfig()
        self.dry_run = dry_run
        self.kill_switch = kill_switch
        self.stale_signal_days = stale_signal_days
        self.min_trade_size = min_trade_size
        self.max_turnover = max_turnover
        self.max_orders = max_orders
        self.cost_bps = cost_bps
        self.artifacts_dir = Path(artifacts_dir)
        self._last_fingerprint: str | None = None   # duplicate guard

    def run(
        self,
        target: TargetWeights,
        snapshot: PositionSnapshot,
    ) -> tuple[OrderPlan, ExecutionResult]:
        """Run a single rebalance cycle.

        Always returns an (OrderPlan, ExecutionResult) pair even when execution
        is blocked — the plan contains the full diff for human review, and the
        result explains why execution was skipped.

        Args:
            target:   TargetWeights from research layer (via
                      target_weights_from_portfolio_weights).
            snapshot: Current portfolio positions from broker / paper / manual.

        Returns:
            (plan, result) tuple.
        """
        service_log: list[str] = []

        # ── Guard: kill switch ─────────────────────────────────────────────
        if self.kill_switch:
            msg = "[KILL SWITCH] All execution permanently blocked."
            logger.error(msg)
            plan = self._build_plan(target, snapshot)
            return plan, ExecutionResult(
                plan_id=plan.plan_id,
                adapter_mode="blocked",
                success=False,
                log_entries=[msg],
            )

        # ── Guard: stale signal ────────────────────────────────────────────
        age_days = (datetime.now() - target.generated_at).total_seconds() / 86400
        if age_days > self.stale_signal_days:
            msg = (
                f"[STALE SIGNAL] Target weights are {age_days:.1f}d old"
                f" (limit: {self.stale_signal_days}d)."
                " Proceeding with caution — verify signal is still current."
            )
            logger.warning(msg)
            service_log.append(msg)

        # ── Guard: position reconciliation ─────────────────────────────────
        total_allocated = sum(snapshot.positions.values())
        implied_total = total_allocated + snapshot.cash_fraction
        if implied_total > 1.0 + 0.05:
            msg = (
                f"[RECONCILIATION] positions ({total_allocated:.1%})"
                f" + cash ({snapshot.cash_fraction:.1%})"
                f" = {implied_total:.1%} > 100%."
                " Snapshot may contain an accounting error or stale data."
            )
            logger.warning(msg)
            service_log.append(msg)

        # ── Build plan ─────────────────────────────────────────────────────
        plan = self._build_plan(target, snapshot)
        service_log.append(
            f"[PLAN] {plan.plan_id[:8]}  strategy={target.strategy_name}"
            f"  rebal_date={target.rebalance_date}"
            f"  nav=${snapshot.nav:,.0f}"
            f"  orders={len(plan.orders)}"
            f"  turnover={plan.total_turnover:.2%}"
        )

        # ── Guard: cash buffer ────────────────────────────────────────────
        est_fee_fraction = plan.total_turnover * self.cost_bps / 10_000
        if snapshot.cash_fraction < est_fee_fraction:
            msg = (
                f"[LOW CASH] cash_fraction={snapshot.cash_fraction:.2%}"
                f" may be insufficient to cover estimated fees"
                f" (~{est_fee_fraction:.2%} of NAV)."
                " Consider keeping a cash buffer before rebalancing."
            )
            logger.warning(msg)
            service_log.append(msg)

        # ── Guard: duplicate execution ─────────────────────────────────────
        fp = _plan_fingerprint(plan)
        if fp == self._last_fingerprint:
            msg = (
                "[DUPLICATE] Target weights unchanged since last execution."
                " Skipping rebalance."
            )
            logger.info(msg)
            service_log.append(msg)
            return plan, ExecutionResult(
                plan_id=plan.plan_id,
                adapter_mode=self.adapter.mode,
                orders_attempted=0,
                success=True,
                log_entries=service_log,
            )

        # ── Risk checks ────────────────────────────────────────────────────
        risk_result = check_order_plan(
            plan, self.risk, self.max_turnover, self.max_orders
        )
        service_log.append(f"[RISK] {risk_result.summary}")

        if not risk_result.passed:
            msg = f"[RISK BLOCK] {risk_result.summary}"
            logger.error(msg)
            service_log.append(msg)
            result = ExecutionResult(
                plan_id=plan.plan_id,
                adapter_mode=self.adapter.mode,
                risk_check=risk_result,
                success=False,
                log_entries=service_log,
            )
            self._save_artifacts(plan, result)
            return plan, result

        # ── Dry-run gate ───────────────────────────────────────────────────
        if self.dry_run:
            service_log.append(
                "[DRY RUN] dry_run=True — plan validated, execution suppressed."
                " Set dry_run=False on RebalanceService to execute."
            )

        # ── Execute via adapter ────────────────────────────────────────────
        result = self.adapter.execute(
            plan, dry_run=self.dry_run, risk_check=risk_result
        )
        result.risk_check = risk_result
        result.log_entries = service_log + result.log_entries

        if not self.dry_run and result.success:
            self._last_fingerprint = fp
            plan.approved = True    # mutate: plan was actually executed

        self._save_artifacts(plan, result)
        return plan, result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_plan(
        self,
        target: TargetWeights,
        snapshot: PositionSnapshot,
    ) -> OrderPlan:
        """Compute diffs, apply min_trade_size, construct OrderPlan."""
        universe = set(target.weights) | set(snapshot.positions)

        all_diffs: list[PositionDiff] = []
        for sym in sorted(universe):
            cur = snapshot.positions.get(sym, 0.0)
            tgt = target.weights.get(sym, 0.0)
            all_diffs.append(PositionDiff(
                symbol=sym,
                current_weight=cur,
                target_weight=tgt,
                delta_weight=tgt - cur,
            ))

        actionable = [
            d for d in all_diffs if abs(d.delta_weight) >= self.min_trade_size
        ]

        decision = RebalanceDecision(
            target=target,
            snapshot=snapshot,
            all_diffs=all_diffs,
            actionable=actionable,
        )

        orders = [
            OrderIntent(
                symbol=d.symbol,
                side=d.side,
                target_weight=d.target_weight,
                delta_weight=d.delta_weight,
                nav=snapshot.nav,
            )
            for d in actionable
        ]

        turnover = sum(abs(d.delta_weight) for d in actionable)

        return OrderPlan(
            decision=decision,
            orders=orders,
            total_turnover=turnover,
            estimated_cost_bps=self.cost_bps,
        )

    def _save_artifacts(self, plan: OrderPlan, result: ExecutionResult) -> None:
        """Write order_plan.json and execution_log.txt to artifacts_dir."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = result.adapter_mode  # include adapter mode to prevent same-second collision

        # ── Execution log ──────────────────────────────────────────────────
        log_path = self.artifacts_dir / f"{ts}_{mode}_execution_log.txt"
        log_path.write_text("\n".join(result.log_entries), encoding="utf-8")

        # ── Order plan (JSON) ──────────────────────────────────────────────
        plan_path = self.artifacts_dir / f"{ts}_{mode}_order_plan.json"
        plan_data: dict = {
            "plan_id": plan.plan_id,
            "created_at": plan.created_at.isoformat(),
            "strategy_name": plan.decision.target.strategy_name,
            "rebalance_date": str(plan.decision.target.rebalance_date),
            "source_record_id": plan.decision.target.source_record_id,
            "adapter_mode": result.adapter_mode,
            "dry_run": not plan.approved,
            "nav": plan.decision.snapshot.nav,
            "total_turnover": round(plan.total_turnover, 4),
            "estimated_cost_bps": plan.estimated_cost_bps,
            "risk_check_passed": result.risk_check.passed if result.risk_check else None,
            "success": result.success,
            "orders": [
                {
                    "symbol": o.symbol,
                    "side": str(o.side),
                    "current_weight": round(
                        plan.decision.snapshot.positions.get(o.symbol, 0.0), 4
                    ),
                    "target_weight": round(o.target_weight, 4),
                    "delta_weight": round(o.delta_weight, 4),
                    "delta_value_usd": round(o.delta_value, 2),
                }
                for o in plan.orders
            ],
        }
        plan_path.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

        logger.debug(
            f"Execution artifacts saved: {log_path.name}, {plan_path.name}"
        )
