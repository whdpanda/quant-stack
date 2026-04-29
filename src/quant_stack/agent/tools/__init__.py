"""Agent tools package for the shadow_run phase.

Public API:
    ToolContext  — shared session state (pass to every dispatch call)
    dispatch()   — call any tool by name
    ALL_TOOLS    — dict of tool_name → callable
    TOOL_SCHEMAS — list of Anthropic-format tool schemas for all 11 tools

11 tools across three categories:

  Research (5):
    load_market_data_tool, build_factors_tool, generate_signals_tool,
    allocate_portfolio_tool, run_research_backtest_tool

  Report (2):
    generate_report_tool, compare_experiments_tool

  Shadow / Audit (4):
    load_current_positions_tool, build_rebalance_plan_tool,
    summarize_shadow_run_tool, review_execution_artifacts_tool

Safety guarantee: no tool in this package submits orders or connects to a broker.
summarize_shadow_run_tool enforces dry_run=True unconditionally.
"""
from __future__ import annotations

from typing import Any, Callable

from quant_stack.agent.tools._context import ToolContext
from quant_stack.agent.tools.report_tools import (
    REPORT_TOOL_SCHEMAS,
    compare_experiments,
    generate_report,
)
from quant_stack.agent.tools.research_tools import (
    RESEARCH_TOOL_SCHEMAS,
    allocate_portfolio,
    build_factors,
    generate_signals,
    load_market_data,
    run_research_backtest,
)
from quant_stack.agent.tools.shadow_tools import (
    SHADOW_TOOL_SCHEMAS,
    build_rebalance_plan,
    load_current_positions,
    review_execution_artifacts,
    summarize_shadow_run,
)

__all__ = [
    "ToolContext",
    "dispatch",
    "ALL_TOOLS",
    "TOOL_SCHEMAS",
]

ALL_TOOLS: dict[str, Callable[[dict[str, Any], ToolContext], dict[str, Any]]] = {
    # ── Research ──────────────────────────────────────────────────────
    "load_market_data_tool":      load_market_data,
    "build_factors_tool":         build_factors,
    "generate_signals_tool":      generate_signals,
    "allocate_portfolio_tool":    allocate_portfolio,
    "run_research_backtest_tool": run_research_backtest,
    # ── Report ────────────────────────────────────────────────────────
    "generate_report_tool":       generate_report,
    "compare_experiments_tool":   compare_experiments,
    # ── Shadow / Audit ────────────────────────────────────────────────
    "load_current_positions_tool":      load_current_positions,
    "build_rebalance_plan_tool":        build_rebalance_plan,
    "summarize_shadow_run_tool":        summarize_shadow_run,
    "review_execution_artifacts_tool":  review_execution_artifacts,
}

TOOL_SCHEMAS: list[dict[str, Any]] = (
    RESEARCH_TOOL_SCHEMAS + REPORT_TOOL_SCHEMAS + SHADOW_TOOL_SCHEMAS
)


def dispatch(
    tool_name: str,
    inputs: dict[str, Any],
    ctx: ToolContext,
) -> dict[str, Any]:
    """Call a registered tool by name and return its result dict.

    Always returns a dict with at least {"status": "ok"|"error"}.
    On unknown tool name, returns {"status": "error", "error": "..."}.
    """
    fn = ALL_TOOLS.get(tool_name)
    if fn is None:
        known = sorted(ALL_TOOLS.keys())
        return {
            "status": "error",
            "error": f"Unknown tool '{tool_name}'. Known tools: {known}",
        }
    return fn(inputs, ctx)
