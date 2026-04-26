"""Three execution adapters: DryRun, Paper, Lean.

All adapters share a common interface::

    result = adapter.execute(plan, dry_run=False)

They differ only in what happens when ``dry_run=False``:

DryRunExecutionAdapter
    Logs every order intent.  Never modifies any state regardless of dry_run.
    The safe default for human review before committing to paper or live.

PaperExecutionAdapter
    Simulates fills against a virtual NAV.  Tracks paper positions in memory.
    When dry_run=False: updates internal positions after "fills".
    When dry_run=True:  simulates but discards the state update.
    Wire _submit_order() to a broker's paper API for real paper trading.

LeanExecutionAdapter
    Translates the OrderPlan into a LEAN-compatible JSON payload.
    When dry_run=False: writes payload to ``output_dir/target_weights.json``.
    When dry_run=True:  builds and returns payload without writing to disk.
    The LEAN algorithm skeleton (lean/SectorMomentumAlgorithm.py) reads
    this file at each scheduled rebalance date.

    Payload structure::

        {
            "strategy_name": "sector_momentum_210d_top3",
            "rebalance_date": "2025-12-30",
            "generated_at": "...",
            "weights": {"QQQ": 0.3333, "XLI": 0.3333, "GDX": 0.3333},
            "all_target_weights": {...},    # includes zeros for liquidated symbols
            "metadata": {...}
        }

    To connect to a live LEAN engine:
        Replace _write_payload() with a call to LEAN's REST API, or place
        the output file in LEAN's algorithm data directory.  The LEAN
        algorithm is scheduled independently of this adapter — ensure the
        adapter writes before the algorithm's timer fires.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from loguru import logger

from quant_stack.execution.domain import (
    ExecutionResult,
    OrderPlan,
    RiskCheckResult,
)


# ── DryRunExecutionAdapter ────────────────────────────────────────────────────

class DryRunExecutionAdapter:
    """Log-only adapter — never submits orders, never modifies state.

    Provides a human-readable order preview that matches exactly what
    PaperExecutionAdapter or LeanExecutionAdapter would execute.
    """

    mode = "dry_run"

    def execute(
        self,
        plan: OrderPlan,
        dry_run: bool = True,
        **_: object,
    ) -> ExecutionResult:
        nav = plan.decision.snapshot.nav
        est_cost = (
            plan.total_turnover * plan.estimated_cost_bps / 10_000 * nav
        )
        log: list[str] = [
            f"[DRY RUN] Plan {plan.plan_id[:8]}",
            f"  strategy : {plan.decision.target.strategy_name}",
            f"  rebal    : {plan.decision.target.rebalance_date}",
            f"  nav      : ${nav:,.0f}",
            f"  orders   : {len(plan.orders)}",
            f"  turnover : {plan.total_turnover:.2%}",
            f"  est cost : {plan.estimated_cost_bps:.0f} bps (~${est_cost:,.0f})",
            "  ─────────────────────────────────────────────────────────",
        ]
        for order in sorted(plan.orders, key=lambda o: o.symbol):
            cur = plan.decision.snapshot.positions.get(order.symbol, 0.0)
            log.append(
                f"  [{order.side.upper():4s}] {order.symbol:6s}"
                f"  {cur:.2%} → {order.target_weight:.2%}"
                f"  Δ={order.delta_weight:+.2%}"
                f"  Δ$={order.delta_value:+,.0f}"
            )
        for entry in log:
            logger.info(entry)

        return ExecutionResult(
            plan_id=plan.plan_id,
            adapter_mode=self.mode,
            orders_attempted=len(plan.orders),
            orders_filled=0,
            orders_rejected=0,
            estimated_cost=est_cost,
            log_entries=log,
            success=True,
        )


# ── PaperExecutionAdapter ─────────────────────────────────────────────────────

class PaperExecutionAdapter:
    """Paper trading adapter — simulates fills, tracks virtual positions.

    Internal state (``_paper_positions``, ``_paper_nav``) is updated in-place
    when ``dry_run=False``.  When ``dry_run=True`` the simulation runs but
    no state is persisted.

    Fills are assumed immediate at NAV (no market-impact model).
    Override ``_submit_order()`` to wire to a real paper broker API.
    """

    mode = "paper"

    def __init__(self) -> None:
        self._paper_positions: dict[str, float] = {}
        self._paper_nav: float = 100_000.0

    def execute(
        self,
        plan: OrderPlan,
        dry_run: bool = False,
        **_: object,
    ) -> ExecutionResult:
        nav = plan.decision.snapshot.nav
        mode_label = "[PAPER-DRY]" if dry_run else "[PAPER]"
        log: list[str] = [
            f"{mode_label} Plan {plan.plan_id[:8]}"
            f"  strategy={plan.decision.target.strategy_name}"
            f"  nav=${nav:,.0f}",
        ]
        if dry_run:
            log.append(
                "  dry_run=True — fills simulated but positions not updated"
            )

        filled = 0
        rejected = 0

        for order in sorted(plan.orders, key=lambda o: o.symbol):
            try:
                self._submit_order(order.symbol, order.target_weight)
                log.append(
                    f"  [FILL] {order.side.upper():4s} {order.symbol:6s}"
                    f"  → {order.target_weight:.2%}"
                    f"  Δ={order.delta_weight:+.2%}"
                )
                filled += 1
                if not dry_run:
                    self._paper_positions[order.symbol] = order.target_weight
            except Exception as exc:  # noqa: BLE001
                log.append(f"  [REJECT] {order.symbol}: {exc}")
                rejected += 1

        # Carry over held positions: target symbols skipped by min_trade_size filter.
        # These symbols were not in plan.orders (delta too small), so the fills loop
        # never wrote them.  Without this step they would vanish from internal state.
        if not dry_run:
            for sym, cur_w in plan.decision.snapshot.positions.items():
                if (
                    sym in plan.decision.target.weights
                    and sym not in self._paper_positions
                    and cur_w > 0
                ):
                    self._paper_positions[sym] = cur_w
                    log.append(f"  [HOLD]      {sym:6s}  {cur_w:.2%} (unchanged)")

        # Liquidate symbols no longer in target
        if not dry_run:
            for sym in list(self._paper_positions):
                if sym not in plan.decision.target.weights:
                    del self._paper_positions[sym]
                    log.append(f"  [LIQUIDATE] {sym}")
            self._paper_nav = nav

        for entry in log:
            logger.info(entry)

        est_cost = plan.total_turnover * plan.estimated_cost_bps / 10_000 * nav
        return ExecutionResult(
            plan_id=plan.plan_id,
            adapter_mode=self.mode,
            orders_attempted=len(plan.orders),
            orders_filled=filled,
            orders_rejected=rejected,
            estimated_cost=est_cost,
            log_entries=log,
            success=rejected == 0,
        )

    def _submit_order(self, symbol: str, target_weight: float) -> None:
        """Override to wire to a real paper broker API."""
        logger.debug(f"[PAPER] SetHoldings({symbol!r}, {target_weight:.4f})")

    @property
    def positions(self) -> dict[str, float]:
        """Current paper positions (fraction of NAV)."""
        return dict(self._paper_positions)


# ── LeanExecutionAdapter ──────────────────────────────────────────────────────

class LeanExecutionAdapter:
    """Translate OrderPlan into a LEAN-compatible JSON payload.

    This adapter does NOT connect to a live LEAN engine.  It produces a
    structured JSON file that the LEAN algorithm skeleton reads at each
    scheduled rebalance:

        lean/SectorMomentumAlgorithm.py  →  reads  output_dir/target_weights.json

    To connect to a live LEAN engine:
        1. Replace ``_write_payload()`` with a LEAN REST API call.
        2. Or place ``output_dir`` inside LEAN's algorithm data directory so
           the algorithm finds it at runtime.

    When ``dry_run=False``: payload is written to ``output_dir/target_weights.json``.
    When ``dry_run=True`` : payload is built and returned but not written.
    """

    mode = "lean"

    def __init__(self, output_dir: str | Path = "./lean_output") -> None:
        self.output_dir = Path(output_dir)

    def execute(
        self,
        plan: OrderPlan,
        dry_run: bool = False,
        risk_check: RiskCheckResult | None = None,
        **_: object,
    ) -> ExecutionResult:
        payload = self._build_payload(plan, risk_check)
        mode_label = "[LEAN-DRY]" if dry_run else "[LEAN]"
        log: list[str] = [
            f"{mode_label} Plan {plan.plan_id[:8]}",
            f"  strategy : {plan.decision.target.strategy_name}",
            f"  date     : {plan.decision.target.rebalance_date}",
            f"  weights  : { {k: f'{v:.2%}' for k, v in payload['weights'].items()} }",
        ]

        if not dry_run:
            unique_path, latest_path = self._write_payload(payload)
            log.append(f"  written  : {unique_path.resolve()}")
            log.append(f"  latest   : {latest_path.resolve()}")
        else:
            log.append("  dry_run=True — payload not written to disk")

        for entry in log:
            logger.info(entry)

        nav = plan.decision.snapshot.nav
        est_cost = plan.total_turnover * plan.estimated_cost_bps / 10_000 * nav
        return ExecutionResult(
            plan_id=plan.plan_id,
            adapter_mode=self.mode,
            orders_attempted=len(plan.orders),
            orders_filled=len(plan.orders) if not dry_run else 0,
            orders_rejected=0,
            estimated_cost=est_cost,
            lean_payload=payload,
            log_entries=log,
            success=True,
        )

    def _build_payload(
        self, plan: OrderPlan, risk_check: RiskCheckResult | None = None
    ) -> dict:
        """Build the LEAN-compatible weight payload dict."""
        # Only include symbols with non-zero target weight
        weights = {
            o.symbol: round(o.target_weight, 6)
            for o in plan.orders
            if o.target_weight > 0
        }
        # Full target weight map including zeros (for LEAN liquidation logic)
        all_weights = {
            k: round(v, 6)
            for k, v in plan.decision.target.weights.items()
        }
        return {
            "strategy_name": plan.decision.target.strategy_name,
            "rebalance_date": str(plan.decision.target.rebalance_date),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "weights": weights,
            "all_target_weights": all_weights,
            "metadata": {
                "source": "quant_stack_execution_layer",
                "adapter": "LeanExecutionAdapter",
                "plan_id": plan.plan_id,
                "nav": plan.decision.snapshot.nav,
                "total_turnover": round(plan.total_turnover, 4),
                "estimated_cost_bps": plan.estimated_cost_bps,
                "risk_checks_passed": risk_check.passed if risk_check else None,
                "source_record_id": plan.decision.target.source_record_id,
            },
        }

    def _write_payload(self, payload: dict) -> tuple[Path, Path]:
        """Write the JSON payload to disk for LEAN algorithm to read.

        Writes two files:
          1. Unique file: ``{rebalance_date}_{strategy}_{plan_id[:8]}_weights.json``
             — permanent audit record, never overwritten.
          2. ``target_weights.json`` — latest-alias read by SectorMomentumAlgorithm.

        Returns:
            (unique_path, latest_path)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rebalance_date = payload.get("rebalance_date", "unknown")
        strategy_raw = payload.get("strategy_name", "strategy")
        plan_id = payload.get("metadata", {}).get("plan_id", "")[:8]
        safe_strategy = re.sub(r"[^\w]", "_", strategy_raw)[:30]
        unique_name = f"{rebalance_date}_{safe_strategy}_{plan_id}_weights.json"

        content = json.dumps(payload, indent=2)
        unique_path = self.output_dir / unique_name
        unique_path.write_text(content, encoding="utf-8")

        latest_path = self.output_dir / "target_weights.json"
        latest_path.write_text(content, encoding="utf-8")

        logger.info(f"[LEAN] Payload written → {unique_path.resolve()}")
        logger.info(f"[LEAN] Latest alias    → {latest_path.resolve()}")
        return unique_path, latest_path
