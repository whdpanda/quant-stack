"""Shared session context passed between agent tool calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ToolContext:
    """In-memory state shared across tool calls in one agent session.

    Tools read/write this object to pass intermediate results without
    serialising large DataFrames into JSON tool_result messages.

    Lifecycle: create one ToolContext per session (or per demo run),
    pass it to every dispatch() call.
    """

    # ── market data ────────────────────────────────────────────────────
    close_df: Any = None          # pd.DataFrame  — from load_market_data_tool
    momentum_df: Any = None       # pd.DataFrame  — raw ROC, from build_factors_tool
    signals_df: Any = None        # pd.DataFrame  — binary 0/1, from generate_signals_tool
    signal_date: str = ""         # ISO date string of the latest signal bar

    # ── derived outputs ────────────────────────────────────────────────
    selected_symbols: list[str] = field(default_factory=list)       # from generate_signals_tool
    target_weights: dict[str, float] = field(default_factory=dict)  # from allocate_portfolio_tool

    # ── execution state ────────────────────────────────────────────────
    positions_snapshot: Any = None   # PositionSnapshot — from load_current_positions_tool
    shadow_result: Any = None        # ShadowRunResult  — from summarize_shadow_run_tool

    # ── experiment tracking ────────────────────────────────────────────
    last_record_path: str | None = None  # from run_research_backtest_tool
