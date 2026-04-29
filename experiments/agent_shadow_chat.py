"""Shadow Agent CLI — natural language interface for shadow execution.

Wraps ShadowAgent with a command-line interface supporting both
single-shot queries and interactive multi-turn conversations.

Usage
-----
    # Single-shot query
    python experiments/agent_shadow_chat.py --query "读取 latest shadow run，告诉我今天是否适合人工执行"

    # Interactive mode (multi-turn)
    python experiments/agent_shadow_chat.py

    # With a specific positions file or NAV override
    python experiments/agent_shadow_chat.py --positions data/current_positions.json
    python experiments/agent_shadow_chat.py --query "..." --verbose

Interactive commands
--------------------
    exit / quit / q     — exit the chat
    reset               — clear conversation history and tool context
    help                — show example queries

Example queries
---------------
    "帮我解释今天 shadow run 为什么建议买 GDX / IBB / XLE"
    "读取 latest shadow run，告诉我今天是否值得人工执行"
    "检查 current_positions 和 target_weights 是否一致"
    "读取 risk_check_result 并用中文解释每一项检查"
    "帮我生成一份完整的 shadow run 分析，从数据下载到 artifacts 审阅"
    "如果我现在是全现金，今天建议怎么执行（仅分析，不下单）"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project src is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_stack.agent.shadow_agent import ShadowAgent, ShadowAgentContext


_API_KEY_ENVS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": "",   # no key needed
    "openai-compatible": "",  # varies; let the backend handle it
}

_EXAMPLE_QUERIES = [
    '帮我解释今天 shadow run 为什么建议买这些 ETF',
    '读取 latest shadow run，告诉我今天是否适合人工执行',
    '读取 risk_check_result 并用中文解释每一项检查',
    '帮我运行完整的 shadow execution（从数据下载到 artifacts）',
    '比较最近这次 shadow run 的持仓和上次有什么不同',
    '如果我是全现金仓位，今天的执行建议是什么',
    'Explain today shadow run result in English',
]


def _check_api_key(provider: str) -> None:
    """Exit early with a clear message if the required API key is missing."""
    env_var = _API_KEY_ENVS.get(provider, "")
    if not env_var:
        return   # ollama / openai-compatible don't need a key check here
    if not os.environ.get(env_var):
        print(
            f"Error: {env_var} is not set.\n"
            f"Set it with: export {env_var}=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_agent(args: argparse.Namespace) -> ShadowAgent:
    ctx = ShadowAgentContext(positions_path=args.positions)
    return ShadowAgent(agent_ctx=ctx, provider=args.provider, model=args.model or None)


def _print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def _run_single_shot(agent: ShadowAgent, query: str, verbose: bool) -> None:
    """Run one query and print the response."""
    _print_separator()
    print(f"Query: {query}")
    _print_separator()
    response = agent.run(query, verbose=verbose)
    print(response)
    _print_separator()


def _run_interactive(agent: ShadowAgent, verbose: bool) -> None:
    """Interactive multi-turn chat loop."""
    _print_separator("=")
    print(f"  Shadow Agent — Interactive Mode  [{agent.provider_name}]")
    print("  Type 'help' for example queries, 'reset' to clear context, 'exit' to quit.")
    _print_separator("=")

    while True:
        # Prompt
        try:
            raw = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exiting]")
            break

        if not raw:
            continue

        # Built-in commands
        lower = raw.lower()
        if lower in ("exit", "quit", "q"):
            print("[Exiting]")
            break

        if lower == "reset":
            agent.reset()
            print("[Context cleared — fresh conversation started]")
            continue

        if lower == "help":
            print("\nExample queries:")
            for q in _EXAMPLE_QUERIES:
                print(f"  • {q}")
            continue

        # Run the agent
        print()
        response = agent.run(raw, verbose=verbose)
        _print_separator()
        print(f"Agent > {response}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shadow Agent CLI — natural language interface for shadow execution analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single-shot query (omit for interactive mode)",
    )
    parser.add_argument(
        "--positions",
        type=str,
        default="data/current_positions.json",
        help="Path to current_positions.json (default: data/current_positions.json)",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "deepseek", "groq", "ollama", "openai-compatible"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name override (uses each provider's default if omitted)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print tool call details to stderr during execution",
    )
    args = parser.parse_args()

    # Check for API key early with provider-specific guidance
    _check_api_key(args.provider)

    agent = _build_agent(args)

    if args.query:
        _run_single_shot(agent, args.query, verbose=args.verbose)
    else:
        _run_interactive(agent, verbose=args.verbose)


if __name__ == "__main__":
    main()
