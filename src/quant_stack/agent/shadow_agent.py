"""LLM-driven shadow execution agent.

Wraps the 11 shadow_run agent tools with a tool-use loop, providing
natural language access to shadow execution analysis.  The LLM provider
is pluggable: Anthropic, OpenAI, DeepSeek, Groq, Ollama, or any
OpenAI-compatible endpoint.

Architecture
------------
    ShadowAgentContext  — project-level defaults (strategy, paths)
    ShadowAgent         — LLM runtime + tool dispatch + safety guardrails
    ALLOWED_TOOL_NAMES  — explicit whitelist (all 11 shadow_run tools)
    FORBIDDEN_TOOLS     — explicit blacklist (broker / order tools)
    LLMBackend          — provider abstraction (see providers.py)

Safety guarantee
----------------
    - Only tools in ALLOWED_TOOL_NAMES can be dispatched.
    - FORBIDDEN_TOOLS calls return status=blocked, never execute.
    - summarize_shadow_run_tool enforces dry_run=True at the tool layer.
    - No tool submits orders or connects to a broker.

Usage
-----
    from quant_stack.agent.shadow_agent import ShadowAgent, ShadowAgentContext

    # Default: Anthropic Claude
    agent = ShadowAgent()
    print(agent.run("读取 latest shadow run，告诉我今天是否适合人工执行"))

    # Switch to OpenAI
    agent = ShadowAgent(provider="openai", model="gpt-4o")

    # Switch to DeepSeek
    agent = ShadowAgent(provider="deepseek")

    # Switch to local Ollama
    agent = ShadowAgent(provider="ollama", model="qwen2.5:14b")

    # Bring your own backend
    from quant_stack.agent.providers import create_backend
    agent = ShadowAgent(backend=create_backend("groq", model="llama-3.3-70b-versatile"))

    # Multi-turn (history preserved within one instance)
    agent.run("还有什么风险需要注意？")
    agent.reset()   # start fresh
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_stack.agent.providers import LLMBackend, ToolResult, create_backend
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
- Surface WARN or FAIL risk checks prominently
- End with 1-3 concrete next-step suggestions for the portfolio manager
- Never phrase a suggestion as a trade instruction (avoid "Buy GDX now")
- Suggestions should be review actions (e.g., "请检查 Section G.2 的价格偏差规则")
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
        return (Path(self.latest_dir) / "shadow_execution_summary.md").exists()

    def to_context_block(self) -> str:
        has_run = self.latest_summary_exists()
        lines = [
            f"Strategy   : {self.strategy_name}",
            f"Universe   : {', '.join(self.universe)}",
            f"Weighting  : {self.weighting_method}",
            f"Params     : momentum_window={self.momentum_window}d, top_n={self.top_n}",
            f"Positions  : {self.positions_path}",
            f"Latest run : {self.latest_dir}/ "
            f"({'ready' if has_run else 'no run yet — call summarize_shadow_run_tool first'})",
        ]
        return "\n".join(f"  {line}" for line in lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ShadowAgent:
    """LLM-driven agent for shadow execution analysis.

    Provider-agnostic: Anthropic, OpenAI, DeepSeek, Groq, Ollama, or any
    OpenAI-compatible endpoint — controlled by the ``provider`` / ``backend`` args.

    Conversation history and ToolContext persist across run() calls.
    Call reset() to start a fresh conversation.

    Args:
        agent_ctx:      Project context (strategy, paths). Defaults to current layout.
        backend:        Pre-built LLMBackend instance. Overrides provider/model.
        provider:       Provider name: 'anthropic' (default), 'openai', 'deepseek',
                        'groq', 'ollama', 'openai-compatible'.
        model:          Model name override (uses each provider's default if None).
        max_tokens:     Max tokens per LLM response (default 4096).
        max_tool_rounds:Hard limit on tool-use iterations per run() call.
    """

    def __init__(
        self,
        agent_ctx: ShadowAgentContext | None = None,
        backend: LLMBackend | None = None,
        provider: str = "anthropic",
        model: str | None = None,
        max_tokens: int = 4096,
        max_tool_rounds: int = 10,
    ) -> None:
        self.agent_ctx = agent_ctx or ShadowAgentContext()
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.tool_ctx = ToolContext()
        self._backend: LLMBackend = backend or create_backend(provider=provider, model=model)

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, query: str, *, verbose: bool = False) -> str:
        """Process one user message and return the agent's text response.

        Internally runs the tool-use loop: the LLM decides which tools to call,
        results are fed back, until a final text response is produced.
        Conversation history is appended automatically for multi-turn use.

        Args:
            query:   Natural language question or instruction.
            verbose: If True, print tool call details to stderr.

        Returns:
            The agent's final text response.
        """
        self._backend.add_user_message(query)
        system = _SYSTEM_PROMPT_TEMPLATE.format(
            context_block=self.agent_ctx.to_context_block(),
        )

        for round_idx in range(self.max_tool_rounds):
            if verbose:
                print(f"\n  [{self._backend.provider_name}] round {round_idx + 1}/{self.max_tool_rounds}", file=sys.stderr)

            response = self._backend.chat(
                system=system,
                tools=ALLOWED_TOOL_SCHEMAS,
                max_tokens=self.max_tokens,
            )

            if not response.needs_tool_use:
                return response.text or "(no response text)"

            # ── Execute all tool calls ─────────────────────────────────
            tool_results: list[ToolResult] = []
            for tc in response.tool_calls:
                if verbose:
                    input_preview = json.dumps(tc.inputs, ensure_ascii=False)
                    if len(input_preview) > 80:
                        input_preview = input_preview[:77] + "..."
                    print(f"  [tool] {tc.name}({input_preview})", file=sys.stderr)

                result = self._dispatch_tool(tc.name, tc.inputs)

                if verbose:
                    print(f"         → status={result.get('status', '?')}", file=sys.stderr)

                tool_results.append(ToolResult(
                    id=tc.id,
                    content=json.dumps(result, ensure_ascii=False, default=str),
                ))

            if not tool_results:
                break

            self._backend.add_tool_results(tool_results)

        return "[Agent: maximum tool-use rounds reached without a final response.]"

    def reset(self) -> None:
        """Clear conversation history and shared ToolContext for a fresh session."""
        self._backend.reset_history()
        self.tool_ctx = ToolContext()

    @property
    def provider_name(self) -> str:
        """Human-readable name of the active LLM provider."""
        return self._backend.provider_name

    @property
    def turn_count(self) -> int:
        """Number of run() calls in the current conversation (approximate)."""
        return getattr(self._backend, "_turn_count", 0)

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
                    f"Tool '{name}' is not in the approved tool list. "
                    f"Allowed: {sorted(ALLOWED_TOOL_NAMES)}"
                ),
            }
        return dispatch(name, inputs, self.tool_ctx)
