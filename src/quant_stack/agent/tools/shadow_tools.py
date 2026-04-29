"""Shadow execution and audit tools.

All tools in this module enforce dry_run=True at construction time.
No broker API is accessed. No orders are submitted.

  - load_current_positions_tool  : load current portfolio from JSON file
  - build_rebalance_plan_tool    : lightweight order preview (no file writes)
  - summarize_shadow_run_tool    : full ShadowExecutionService run + artifact write
  - review_execution_artifacts_tool: read any artifact from the latest / named run
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from quant_stack.agent.tools._context import ToolContext

# Formal strategy constants — mirror shadow_run.py; never expose to agent input
_STRATEGY_NAME = "sector_momentum_210d_top3"
_WEIGHTING_DISPLAY = "BLEND_70_30 (70% equal + 30% inverse-vol)"
_UNIVERSE_TYPE_DISPLAY = "Sector / industry / thematic ETFs"
_SHADOW_DIR = Path("shadow_artifacts")
_EXECUTION_ARTIFACTS_DIR = Path("execution_artifacts")

_ARTIFACT_FILENAMES: dict[str, str] = {
    "current_positions_snapshot": "current_positions_snapshot.json",
    "target_weights_snapshot": "target_weights_snapshot.json",
    "rebalance_plan": "rebalance_plan.json",
    "risk_check_result": "risk_check_result.json",
    "shadow_execution_summary": "shadow_execution_summary.md",
    "execution_log": "execution_log.jsonl",
    "manual_execution_log_template": "manual_execution_log_template.json",
}


# ── Tool implementations ──────────────────────────────────────────────────────

def load_current_positions(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Load the current portfolio snapshot from a JSON positions file."""
    path_str = inputs.get("path", "data/current_positions.json")
    nav_override = inputs.get("nav_override")

    try:
        from quant_stack.execution.positions import load_positions_json
        snapshot = load_positions_json(path_str)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"Failed to load positions: {e}"}

    if nav_override is not None:
        from quant_stack.execution.domain import PositionSnapshot
        snapshot = PositionSnapshot(
            timestamp=snapshot.timestamp,
            nav=float(nav_override),
            positions=snapshot.positions,
            cash_fraction=snapshot.cash_fraction,
            source=snapshot.source,
        )

    ctx.positions_snapshot = snapshot

    return {
        "status": "ok",
        "path": path_str,
        "nav": snapshot.nav,
        "positions": snapshot.positions,
        "cash_fraction": round(snapshot.cash_fraction, 4),
        "source": snapshot.source,
    }


def build_rebalance_plan(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Compute a lightweight order preview (no file writes, no risk checks).

    Takes target weights from ctx.target_weights (or inputs) and current positions
    from ctx.positions_snapshot (or loads from inputs["current_positions_path"]).
    This is a preview tool — use summarize_shadow_run_tool for the full risk-checked run.
    """
    target_weights: dict[str, float] = inputs.get("target_weights") or ctx.target_weights
    if not target_weights:
        return {
            "status": "error",
            "error": (
                "No target weights. Call allocate_portfolio_tool first, "
                "or pass 'target_weights' in inputs."
            ),
        }

    # Load positions if not in context
    if ctx.positions_snapshot is None:
        pos_path = inputs.get("current_positions_path", "data/current_positions.json")
        try:
            from quant_stack.execution.positions import load_positions_json
            ctx.positions_snapshot = load_positions_json(pos_path)
        except FileNotFoundError as e:
            return {"status": "error", "error": str(e)}

    snapshot = ctx.positions_snapshot
    nav = snapshot.nav
    current = snapshot.positions
    min_trade_size = float(inputs.get("min_trade_size", 0.005))

    all_symbols = sorted(set(target_weights) | set(current))
    orders: list[dict[str, Any]] = []
    total_turnover = 0.0

    for sym in all_symbols:
        tgt = target_weights.get(sym, 0.0)
        cur = current.get(sym, 0.0)
        delta = tgt - cur
        if abs(delta) < min_trade_size:
            continue
        action = "BUY" if delta > 0 else "SELL"
        orders.append({
            "symbol": sym,
            "action": action,
            "current_weight": round(cur, 4),
            "target_weight": round(tgt, 4),
            "delta_weight": round(delta, 4),
            "delta_usd": round(delta * nav),
        })
        total_turnover += abs(delta)

    estimated_cost_usd = round(total_turnover * 20.0 / 10_000 * nav)

    return {
        "status": "ok",
        "nav": nav,
        "orders": orders,
        "order_count": len(orders),
        "total_turnover": round(total_turnover, 4),
        "estimated_cost_usd": estimated_cost_usd,
        "note": "Preview only — no risk checks run. Use summarize_shadow_run_tool for full validation.",
    }


def summarize_shadow_run(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Run a full shadow execution cycle and write all 7 artifacts to disk.

    Requires ctx.target_weights (from allocate_portfolio_tool).
    Loads positions from ctx.positions_snapshot or inputs["positions_path"].

    Hard-coded safety constraints (not exposed to agent inputs):
      - dry_run=True always
      - kill_switch=False (respects the live kill_switch in AppConfig if set)
      - adapter=DryRunExecutionAdapter (no broker connection)
    """
    if not ctx.target_weights:
        return {
            "status": "error",
            "error": "No target weights in context. Call allocate_portfolio_tool first.",
        }

    # ── Get or load positions ──────────────────────────────────────────
    if ctx.positions_snapshot is None:
        pos_path = inputs.get("positions_path", "data/current_positions.json")
        try:
            from quant_stack.execution.positions import load_positions_json
            ctx.positions_snapshot = load_positions_json(pos_path)
        except FileNotFoundError as e:
            return {"status": "error", "error": str(e)}

    snapshot = ctx.positions_snapshot
    nav_override = inputs.get("nav_override")
    if nav_override is not None:
        from quant_stack.execution.domain import PositionSnapshot
        snapshot = PositionSnapshot(
            timestamp=snapshot.timestamp,
            nav=float(nav_override),
            positions=snapshot.positions,
            cash_fraction=snapshot.cash_fraction,
            source=snapshot.source,
        )

    # ── Build TargetWeights from context ───────────────────────────────
    try:
        from quant_stack.core.schemas import PortfolioWeights
        from quant_stack.execution.domain import target_weights_from_portfolio_weights

        signal_date_str = ctx.signal_date or str(date.today())

        pw = PortfolioWeights(
            weights=ctx.target_weights,
            method="blend_70_30",
            rebalance_date=date.fromisoformat(signal_date_str),
        )
        target = target_weights_from_portfolio_weights(
            pw,
            strategy_name=_STRATEGY_NAME,
            source_record_id="",
        )
    except Exception as e:
        return {"status": "error", "error": f"Failed to build TargetWeights: {e}"}

    # ── Build ShadowExecutionService (dry_run=True enforced) ──────────
    try:
        from quant_stack.core.config import AppConfig, load_config
        from quant_stack.execution.adapters import DryRunExecutionAdapter
        from quant_stack.execution.service import RebalanceService
        from quant_stack.execution.shadow import ShadowExecutionService

        try:
            app_cfg = load_config("config/settings.yaml")
        except FileNotFoundError:
            app_cfg = AppConfig()
        risk_cfg = app_cfg.execution.risk

        adapter = DryRunExecutionAdapter()
        service = RebalanceService(
            adapter=adapter,
            risk=risk_cfg,
            dry_run=True,           # ← hardcoded; not exposed to agent
            kill_switch=False,
            stale_signal_days=5,
            min_trade_size=0.005,
            max_turnover=1.5,
            max_orders=20,
            cost_bps=20.0,
            artifacts_dir=str(_EXECUTION_ARTIFACTS_DIR),
        )
        shadow_svc = ShadowExecutionService(service=service, shadow_dir=_SHADOW_DIR)
    except Exception as e:
        return {"status": "error", "error": f"Failed to build shadow service: {e}"}

    # ── Extract latest prices for D.2 section ─────────────────────────
    latest_prices: dict[str, float] | None = None
    if ctx.close_df is not None:
        latest_row = ctx.close_df.iloc[-1]
        latest_prices = {
            sym: float(latest_row[sym])
            for sym in ctx.close_df.columns
            if not math.isnan(float(latest_row[sym]))
        }

    universe = (
        list(ctx.close_df.columns)
        if ctx.close_df is not None
        else list(ctx.target_weights.keys())
    )

    # ── Run shadow ─────────────────────────────────────────────────────
    try:
        shadow_result = shadow_svc.run(
            target,
            snapshot,
            weighting_method=_WEIGHTING_DISPLAY,
            universe=universe,
            universe_type=_UNIVERSE_TYPE_DISPLAY,
            latest_prices=latest_prices,
        )
    except Exception as e:
        return {"status": "error", "error": f"Shadow run failed: {e}"}

    ctx.shadow_result = shadow_result

    # ── Parse risk summary from artifact ──────────────────────────────
    risk_overall = "UNKNOWN"
    warnings: list[str] = []
    try:
        risk_path = shadow_result.artifacts.get("risk_check_result")
        if risk_path and risk_path.exists():
            risk_data = json.loads(risk_path.read_text(encoding="utf-8"))
            warnings = risk_data.get("warnings", [])
            failures = risk_data.get("failures", [])
            if failures:
                risk_overall = "FAILED"
            elif warnings:
                risk_overall = "PASSED_WITH_WARNINGS"
            else:
                risk_overall = "ALL_CHECKS_PASSED"
    except Exception:
        pass

    return {
        "status": "ok",
        "run_id": shadow_result.run_id,
        "needs_rebalance": shadow_result.needs_rebalance,
        "risk_overall": risk_overall,
        "warnings": warnings,
        "order_count": len(shadow_result.plan.orders),
        "summary_path": str(shadow_result.artifacts.get("shadow_execution_summary", "")),
        "template_path": str(shadow_result.artifacts.get("manual_execution_log_template", "")),
        "run_dir": str(shadow_result.run_dir),
    }


def review_execution_artifacts(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Read a specific artifact from a shadow run directory.

    Use run_id='latest' to always read the most recent run.
    """
    run_id: str = inputs.get("run_id", "latest")
    artifact: str = inputs.get("artifact", "risk_check_result")

    filename = _ARTIFACT_FILENAMES.get(artifact)
    if filename is None:
        return {
            "status": "error",
            "error": f"Unknown artifact '{artifact}'. Valid values: {list(_ARTIFACT_FILENAMES)}",
        }

    if run_id == "latest":
        run_dir = _SHADOW_DIR / "latest"
    else:
        run_dir = _SHADOW_DIR / run_id

    artifact_path = run_dir / filename
    if not artifact_path.exists():
        return {
            "status": "error",
            "error": f"Artifact not found: {artifact_path}. Run summarize_shadow_run_tool first.",
        }

    try:
        content_str = artifact_path.read_text(encoding="utf-8")
        if filename.endswith(".json"):
            content: Any = json.loads(content_str)
        elif filename.endswith(".jsonl"):
            content = [json.loads(line) for line in content_str.splitlines() if line.strip()]
        else:
            content = content_str  # .md — return raw text
    except Exception as e:
        return {"status": "error", "error": f"Failed to read artifact: {e}"}

    return {
        "status": "ok",
        "run_id": run_id,
        "artifact": artifact,
        "path": str(artifact_path),
        "content": content,
    }


# ── Anthropic tool schemas ────────────────────────────────────────────────────

SHADOW_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "load_current_positions_tool",
        "description": (
            "Load the current portfolio snapshot from a JSON positions file. "
            "Stores the result in session context for subsequent tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to current_positions.json (default: data/current_positions.json)",
                },
                "nav_override": {
                    "type": "number",
                    "description": "Override NAV from file (e.g. 150000)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "build_rebalance_plan_tool",
        "description": (
            "Compute a lightweight order preview: current vs target weights, delta, and estimated cost. "
            "No file writes. No risk checks. Use summarize_shadow_run_tool for full validation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_weights": {
                    "type": "object",
                    "description": "Target weight dict (symbol → weight). Defaults to context value.",
                    "additionalProperties": {"type": "number"},
                },
                "current_positions_path": {
                    "type": "string",
                    "description": "Positions file path (used only if context has no snapshot)",
                },
                "min_trade_size": {
                    "type": "number",
                    "description": "Minimum |delta_weight| to include an order (default 0.005)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "summarize_shadow_run_tool",
        "description": (
            "Run a full shadow execution cycle: risk checks, order plan, and all 7 artifact files. "
            "Requires allocate_portfolio_tool to have been called first. "
            "dry_run=True is always enforced — no orders are submitted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "positions_path": {
                    "type": "string",
                    "description": "Positions file (used only if context has no snapshot)",
                },
                "nav_override": {
                    "type": "number",
                    "description": "Override NAV for this run only",
                },
            },
            "required": [],
        },
    },
    {
        "name": "review_execution_artifacts_tool",
        "description": (
            "Read a specific artifact from the latest (or named) shadow run directory. "
            "Valid artifacts: current_positions_snapshot, target_weights_snapshot, "
            "rebalance_plan, risk_check_result, shadow_execution_summary, "
            "execution_log, manual_execution_log_template."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Shadow run ID or 'latest' (default: 'latest')",
                },
                "artifact": {
                    "type": "string",
                    "enum": list(_ARTIFACT_FILENAMES.keys()),
                    "description": "Which artifact to read",
                },
            },
            "required": [],
        },
    },
]
