"""LLM-driven shadow execution agent.

Wraps the 11 shadow_run agent tools with an Anthropic tool-use loop,
providing natural language access to shadow execution analysis.

Architecture
------------
    ShadowAgentContext  — project-level defaults injected into the system prompt
    ShadowAgent         — LLM runtime + tool dispatch + safety guardrails
    ALLOWED_TOOL_NAMES  — explicit whitelist (all 11 shadow_run tools)
    FORBIDDEN_TOOLS     — explicit blacklist (broker / order tools)

Safety guarantee
----------------
    - Only tools in ALLOWED_TOOL_NAMES can be dispatched.
    - FORBIDDEN_TOOLS calls return a "blocked" error, never execute.
    - summarize_shadow_run_tool enforces dry_run=True unconditionally (in the tool layer).
    - No tool in this package submits orders or connects to a broker.

Usage
-----
    from quant_stack.agent.shadow_agent import ShadowAgent, ShadowAgentContext

    ctx = ShadowAgentContext(positions_path="data/current_positions.json")
    agent = ShadowAgent(agent_ctx=ctx)
    print(agent.run("读取 latest shadow run，告诉我今天是否适合人工执行"))

    # Multi-turn (history preserved within one ShadowAgent instance)
    agent.run("还有什么风险需要注意？")

    # Start fresh
    agent.reset()
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_stack.agent.tools import ToolContext, dispatch
from quant_stack.agent.tools import TOOL_SCHEMAS
from quant_stack.research.strategies.sector_momentum import RISK_ON_UNIVERSE


# ── Safety boundary ───────────────────────────────────────────────────────────

ALLOWED_TOOL_NAMES: frozenset[str] = frozenset({
    "load_market_data_tool",
    "build_factors_tool",
    "generate_signals_tool",
    "allocate_portfolio_tool",
    "run_research_backtest_tool",
    "generate_report_tool",
    "compare_experiments_tool",
    "load_current_positions_tool",
    "build_rebalance_plan_tool",
    "summarize_shadow_run_tool",
    "review_execution_artifacts_tool",
})

FORBIDDEN_TOOLS: frozenset[str] = frozenset({
    "submit_order_tool",
    "cancel_order_tool",
    "modify_order_tool",
    "place_order_tool",
    "send_order_tool",
    "execute_trade_tool",
    "live_broker_tool",
    "broker_connect_tool",
    "account_trade_tool",
    "live_execution_tool",
})

ALLOWED_TOOL_SCHEMAS: list[dict[str, Any]] = [
    s for s in TOOL_SCHEMAS if s["name"] in ALLOWED_TOOL_NAMES
]


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are a shadow execution assistant for the quant-stack sector momentum strategy.
Your job is to analyze, explain, audit, and summarize shadow execution results.
You help the portfolio manager understand the current rebalance recommendation
before they make a manual trading decision.

## Current Project State
{context_block}

## What You Can Do
- Run the full shadow execution pipeline and explain the results
- Explain why certain ETFs were selected (momentum ranking, weights)
- Summarize risk check results and highlight any warnings
- Read and interpret shadow run artifacts
- Compare two experiments or shadow runs
- Generate human-readable audit summaries
- Suggest next steps for the portfolio manager to review

## Hard Boundaries — Never Violate These
1. You CANNOT submit, cancel, or modify orders of any kind.
2. You CANNOT make the final trading decision — the human always decides.
3. You CANNOT bypass or override risk checks in the execution layer.
4. You CANNOT modify strategy parameters, universe, or weighting scheme.
5. You CANNOT access broker APIs or live account data.
6. If the user asks "should I execute?" — provide analysis and risks only.
   Always end with: "最终执行决定由您人工做出。"

## Response Guidelines
- Respond in the user's language (Chinese or English — match the query)
- Lead with the key conclusion, then supporting details
- Surface WARN or FAIL risk checks prominently (bold or at the top)
- End with 1-3 concrete next-step suggestions for the portfolio manager
- Never phrase a suggestion as a trade instruction (e.g., avoid "Buy GDX now")
- Suggestions should be review actions (e.g., "请检查 Section D.2 的实时价格偏差规则")
"""


# ── Project context ───────────────────────────────────────────────────────────

@dataclass
class ShadowAgentContext:
    """Project-level defaults for the shadow execution agent.

    Injected into the system prompt so the LLM knows the current project state.
    """
    strategy_name: str = "sector_momentum_210d_top3"
    universe: list[str] = field(default_factory=lambda: list(RISK_ON_UNIVERSE))
    positions_path: str = "data/current_positions.json"
    shadow_artifacts_dir: str = "shadow_artifacts"
    latest_dir: str = "shadow_artifacts/latest"
    weighting_method: str = "BLEND_70_30 (70% equal + 30% inverse-vol)"
    momentum_window: int = 210
    top_n: int = 3

    def latest_summary_exists(self) -> bool:
        """True if the latest shadow run has produced a summary file."""
        return (Path(self.latest_dir) / "shadow_execution_summary.md").exists()

    def to_context_block(self) -> str:
        """Multi-line text block for the system prompt."""
        has_run = self.latest_summary_exists()
        lines = [
            f"Strategy   : {self.strategy_name}",
            f"Universe   : {', '.join(self.universe)}",
            f"Weighting  : {self.weighting_method}",
            f"Params     : momentum_window={self.momentum_window}d, top_n={self.top_n}",
            f"Positions  : {self.positions_path}",
            f"Latest run : {self.latest_dir}/ ({'ready' if has_run else 'no run yet — call summarize_shadow_run_tool first'})",
        ]
        return "\n".join(f"  {line}" for line in lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ShadowAgent:
    """LLM-driven agent for shadow execution analysis.

    Uses the Anthropic tool-use API to orchestrate the 11 shadow_run tools
    based on natural language queries.

    Multi-turn support: conversation history and ToolContext persist across
    run() calls within the same ShadowAgent instance.  Call reset() to start
    a fresh conversation.

    Args:
        agent_ctx:      Project-level defaults (strategy, paths). Defaults to
                        ShadowAgentContext() which reads the current project layout.
        model:          Anthropic model ID (default claude-sonnet-4-6).
        max_tokens:     Max tokens per LLM response (default 4096).
        max_tool_rounds:Hard limit on tool-use iterations per run() call.
                        Prevents infinite loops if the model keeps calling tools.
    """

    def __init__(
        self,
        agent_ctx: ShadowAgentContext | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        max_tool_rounds: int = 10,
    ) -> None:
        self.agent_ctx = agent_ctx or ShadowAgentContext()
        self.model = model
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.tool_ctx = ToolContext()
        self._history: list[dict[str, Any]] = []
        self._client = _build_anthropic_client()

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, query: str, *, verbose: bool = False) -> str:
        """Process one user message and return the agent's text response.

        Internally executes the Anthropic tool-use loop: the LLM decides which
        tools to call, results are fed back, until the LLM produces a final text
        response.  Conversation history is appended automatically.

        Args:
            query:   Natural language question or instruction.
            verbose: If True, print tool call details to stderr.

        Returns:
            The agent's final text response.
        """
        self._history.append({"role": "user", "content": query})
        system = _SYSTEM_PROMPT_TEMPLATE.format(
            context_block=self.agent_ctx.to_context_block(),
        )

        for round_idx in range(self.max_tool_rounds):
            if verbose:
                print(f"\n  [agent] LLM round {round_idx + 1}/{self.max_tool_rounds}", file=sys.stderr)

            response = self._client.messages.create(
                model=self.model,
                system=system,
                messages=self._history,
                tools=ALLOWED_TOOL_SCHEMAS,
                max_tokens=self.max_tokens,
            )

            # Append assistant turn (preserve ContentBlock objects — SDK accepts them)
            self._history.append({"role": "assistant", "content": response.content})

            if response.stop_reason in ("end_turn", "stop_sequence", "max_tokens"):
                return _extract_text(response)

            if response.stop_reason != "tool_use":
                return _extract_text(response)

            # ── Execute all tool calls in this response ────────────────
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if not hasattr(block, "type") or block.type != "tool_use":
                    continue

                if verbose:
                    input_preview = json.dumps(block.input, ensure_ascii=False)
                    if len(input_preview) > 80:
                        input_preview = input_preview[:77] + "..."
                    print(f"  [tool] {block.name}({input_preview})", file=sys.stderr)

                result = self._dispatch_tool(block.name, block.input)

                if verbose:
                    print(f"         → status={result.get('status', '?')}", file=sys.stderr)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

            if not tool_results:
                # No tool_use blocks despite tool_use stop reason — shouldn't happen
                break

            self._history.append({"role": "user", "content": tool_results})

        return "[Agent: maximum tool-use rounds reached without a final response.]"

    def reset(self) -> None:
        """Clear conversation history and shared ToolContext.

        Call this to start a completely fresh conversation without creating
        a new ShadowAgent instance.
        """
        self._history.clear()
        self.tool_ctx = ToolContext()

    @property
    def turn_count(self) -> int:
        """Number of user turns in the current conversation."""
        return sum(1 for m in self._history if m["role"] == "user")

    # ── Internal ──────────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call with whitelist + blacklist enforcement."""
        if name in FORBIDDEN_TOOLS:
            return {
                "status": "blocked",
                "error": (
                    f"Tool '{name}' is blocked in the shadow_run agent. "
                    "This agent cannot submit, cancel, or modify orders, "
                    "and cannot control live trading in any way. "
                    "Provide analysis and suggestions only."
                ),
            }
        if name not in ALLOWED_TOOL_NAMES:
            return {
                "status": "blocked",
                "error": (
                    f"Tool '{name}' is not in the approved tool list for this agent. "
                    f"Allowed: {sorted(ALLOWED_TOOL_NAMES)}"
                ),
            }
        return dispatch(name, inputs, self.tool_ctx)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_anthropic_client() -> Any:
    """Build an Anthropic client; raise ImportError with install instructions."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The 'anthropic' package is required.\n"
            "Install: pip install anthropic\n"
            "API key : set ANTHROPIC_API_KEY environment variable."
        ) from e
    return anthropic.Anthropic()


def _extract_text(response: Any) -> str:
    """Extract concatenated text from all TextBlock items in a response."""
    parts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(parts).strip() or "(no text response)"
