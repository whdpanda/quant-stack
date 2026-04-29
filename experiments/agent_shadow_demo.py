"""Agent Shadow Demo — direct tool call chain (no LLM, no external scheduler).

This script shows how the 11 agent tools chain together to complete a full
shadow_run cycle. All tool calls are made directly in Python — there is no
LLM planner or Anthropic API involved here.

The chain demonstrated:

    load_market_data_tool          →  download prices
    build_factors_tool             →  compute 210-day momentum scores
    generate_signals_tool          →  select top-3 ETFs
    allocate_portfolio_tool        →  compute BLEND_70_30 weights
    load_current_positions_tool    →  read current holdings from JSON
    build_rebalance_plan_tool      →  preview orders (no file writes)
    summarize_shadow_run_tool      →  full shadow run + 7 artifact files
    review_execution_artifacts_tool→  read risk_check_result.json

Artifacts written to: shadow_artifacts/{run_id}/ and shadow_artifacts/latest/

Usage
-----
    # First run (all-cash):
    python experiments/agent_shadow_demo.py

    # With an existing portfolio:
    python experiments/agent_shadow_demo.py --positions data/current_positions.json

    # Override NAV:
    python experiments/agent_shadow_demo.py --nav 150000
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from quant_stack.agent.tools import ToolContext, dispatch
from quant_stack.research.strategies.sector_momentum import RISK_ON_UNIVERSE


# ── Display helpers ───────────────────────────────────────────────────────────

def _fmt_val(val: Any, max_len: int = 100) -> str:
    if isinstance(val, dict):
        s = json.dumps(val, ensure_ascii=False)
    elif isinstance(val, list) and val and isinstance(val[0], dict):
        s = json.dumps(val, ensure_ascii=False)
    else:
        s = str(val)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def _print_step(step_num: int, total: int, name: str, result: dict[str, Any]) -> None:
    status = result.get("status", "?")
    icon = "[OK]" if status == "ok" else "[ERR]"
    print(f"\n[{step_num}/{total}] {icon}  {name}")
    if status == "error":
        print(f"         ERROR: {result.get('error', 'unknown error')}")
        return
    for key, val in result.items():
        if key == "status":
            continue
        print(f"         {key}: {_fmt_val(val)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent shadow demo — direct tool chain (no LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--positions",
        default="data/current_positions.json",
        help="Path to current_positions.json (default: data/current_positions.json)",
    )
    parser.add_argument(
        "--nav",
        type=float,
        default=None,
        help="Override NAV (e.g. --nav 150000)",
    )
    args = parser.parse_args()

    TOTAL_STEPS = 8
    print("=" * 70)
    print("  Agent Shadow Demo — Direct Tool Chain (no LLM)")
    print("  Universe:", RISK_ON_UNIVERSE)
    print("=" * 70)

    ctx = ToolContext()

    # ── Step 1: Load market data ──────────────────────────────────────
    r1 = dispatch("load_market_data_tool", {"symbols": RISK_ON_UNIVERSE, "lookback_days": 350}, ctx)
    _print_step(1, TOTAL_STEPS, "load_market_data_tool", r1)
    if r1["status"] != "ok":
        sys.exit(1)

    # ── Step 2: Build factors (210-day momentum) ──────────────────────
    r2 = dispatch("build_factors_tool", {"momentum_window": 210}, ctx)
    _print_step(2, TOTAL_STEPS, "build_factors_tool", r2)
    if r2["status"] != "ok":
        sys.exit(1)

    # ── Step 3: Generate signals (top-3) ─────────────────────────────
    r3 = dispatch("generate_signals_tool", {"top_n": 3}, ctx)
    _print_step(3, TOTAL_STEPS, "generate_signals_tool", r3)
    if r3["status"] != "ok":
        sys.exit(1)

    # ── Step 4: Allocate portfolio (BLEND_70_30) ──────────────────────
    r4 = dispatch("allocate_portfolio_tool", {"weighting": "BLEND_70_30"}, ctx)
    _print_step(4, TOTAL_STEPS, "allocate_portfolio_tool", r4)
    if r4["status"] != "ok":
        sys.exit(1)

    # ── Step 5: Load current positions ────────────────────────────────
    pos_inputs: dict[str, Any] = {"path": args.positions}
    if args.nav is not None:
        pos_inputs["nav_override"] = args.nav
    r5 = dispatch("load_current_positions_tool", pos_inputs, ctx)
    _print_step(5, TOTAL_STEPS, "load_current_positions_tool", r5)
    if r5["status"] != "ok":
        sys.exit(1)

    # ── Step 6: Build rebalance plan (preview, no file writes) ────────
    r6 = dispatch("build_rebalance_plan_tool", {}, ctx)
    _print_step(6, TOTAL_STEPS, "build_rebalance_plan_tool", r6)
    # Non-fatal: preview failing doesn't block the full shadow run

    # ── Step 7: Full shadow run (writes 7 artifact files) ─────────────
    shadow_inputs: dict[str, Any] = {}
    if args.nav is not None:
        shadow_inputs["nav_override"] = args.nav
    r7 = dispatch("summarize_shadow_run_tool", shadow_inputs, ctx)
    _print_step(7, TOTAL_STEPS, "summarize_shadow_run_tool", r7)
    if r7["status"] != "ok":
        sys.exit(1)

    # ── Step 8: Read risk_check_result from latest run ────────────────
    r8 = dispatch(
        "review_execution_artifacts_tool",
        {"run_id": "latest", "artifact": "risk_check_result"},
        ctx,
    )
    _print_step(8, TOTAL_STEPS, "review_execution_artifacts_tool", r8)

    # ── Final summary ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    if r7["status"] == "ok":
        print(f"  Run ID         : {r7.get('run_id', 'N/A')}")
        print(f"  Needs rebalance: {r7.get('needs_rebalance', 'N/A')}")
        print(f"  Risk overall   : {r7.get('risk_overall', 'N/A')}")
        warnings = r7.get("warnings", [])
        if warnings:
            print(f"  Warnings       : {', '.join(warnings)}")
        print(f"  Orders         : {r7.get('order_count', 0)}")
        print()
        print("  Artifacts written:")
        print(f"    {r7.get('run_dir', '')}")
        print(f"    shadow_artifacts/latest/")
        print()
        print("  Human review:")
        print(f"    {r7.get('summary_path', '')}")
        if r7.get("template_path"):
            print(f"    {r7.get('template_path', '')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
