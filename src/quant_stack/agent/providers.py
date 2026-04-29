"""LLM provider abstraction layer.

Decouples ShadowAgent from any specific LLM vendor.
Adding a new provider = subclass LLMBackend, implement 4 methods.

Supported out of the box
------------------------
    anthropic        — Claude models via Anthropic API
    openai           — OpenAI models (GPT-4o, o1, …)
    deepseek         — DeepSeek chat via OpenAI-compatible endpoint
    groq             — Groq inference via OpenAI-compatible endpoint
    ollama           — Local models via Ollama's OpenAI-compatible server
    openai-compatible— Any URL that speaks the OpenAI chat completions API

Tool schema format
------------------
The project stores tool schemas in Anthropic format (input_schema).
OpenAI-compatible backends auto-convert via _anthropic_to_openai_tools().
Callers always pass Anthropic-format schemas; conversion is internal.

Usage
-----
    from quant_stack.agent.providers import create_backend

    # Anthropic (default)
    backend = create_backend("anthropic", model="claude-sonnet-4-6")

    # OpenAI
    backend = create_backend("openai", model="gpt-4o")

    # DeepSeek
    backend = create_backend("deepseek", model="deepseek-chat")

    # Groq
    backend = create_backend("groq", model="llama-3.3-70b-versatile")

    # Ollama (local)
    backend = create_backend("ollama", model="qwen2.5:14b")

    # Any OpenAI-compatible endpoint
    backend = create_backend(
        "openai-compatible",
        model="your-model",
        base_url="http://...",
        api_key="...",
    )
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Normalised interchange types ──────────────────────────────────────────────

@dataclass
class ToolCall:
    """One tool invocation requested by the LLM."""
    id: str
    name: str
    inputs: dict[str, Any]


@dataclass
class ToolResult:
    """Result to feed back to the LLM for one tool call."""
    id: str            # Must match ToolCall.id
    content: str       # JSON-serialised result dict


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    needs_tool_use: bool = False


# ── Abstract backend ──────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """Protocol for a stateful, single-conversation LLM backend.

    Each backend owns its conversation history in its native message format.
    Callers interact through the four methods below; they never touch raw
    API objects or format-specific message dicts.

    History lifecycle
    -----------------
    add_user_message()  — append one user turn
    chat()              — call the API, append the assistant turn, return response
    add_tool_results()  — append tool results (must follow a chat() with tool calls)
    reset_history()     — clear everything for a fresh conversation
    """

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """Append a user turn to the conversation history."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None:
        """Append tool results after a tool-use response."""

    @abstractmethod
    def chat(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],   # Anthropic-format schemas
        max_tokens: int,
    ) -> LLMResponse:
        """Send the current history to the LLM and return a normalised response.

        Implementations must append the assistant turn to history before returning.
        tools are always provided in Anthropic format; backends convert as needed.
        """

    @abstractmethod
    def reset_history(self) -> None:
        """Discard conversation history."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'anthropic', 'openai')."""


# ── Anthropic backend ─────────────────────────────────────────────────────────

class AnthropicBackend(LLMBackend):
    """Anthropic Claude via the official anthropic Python SDK.

    Passes tool schemas unchanged (native Anthropic format).
    History is stored as Anthropic-format message dicts.
    System prompt is passed as the dedicated system parameter.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model
        self._history: list[dict[str, Any]] = []
        self._client: Any = None  # lazy: created on first chat() call

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def add_user_message(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self._history.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.id,
                    "content": r.content,
                }
                for r in results
            ],
        })

    def chat(self, *, system: str, tools: list[dict[str, Any]], max_tokens: int) -> LLMResponse:
        if self._client is None:
            self._client = self._build_client()
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=self._history,
            tools=tools,           # Anthropic format — no conversion
            max_tokens=max_tokens,
        )

        # Append assistant turn (SDK ContentBlock objects accepted by next call)
        self._history.append({"role": "assistant", "content": response.content})

        text = " ".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()

        tool_calls = [
            ToolCall(id=block.id, name=block.name, inputs=block.input)
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            needs_tool_use=response.stop_reason == "tool_use",
        )

    def reset_history(self) -> None:
        self._history.clear()

    @staticmethod
    def _build_client() -> Any:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "pip install anthropic\n"
                "Then set ANTHROPIC_API_KEY environment variable."
            ) from e
        return anthropic.Anthropic()


# ── OpenAI-compatible backend ─────────────────────────────────────────────────

class OpenAICompatibleBackend(LLMBackend):
    """OpenAI chat completions API — works with any compatible endpoint.

    Compatible providers (set base_url and the matching api_key env var):
        OpenAI   — no base_url needed; reads OPENAI_API_KEY
        DeepSeek — base_url="https://api.deepseek.com/v1"; reads DEEPSEEK_API_KEY
        Groq     — base_url="https://api.groq.com/openai/v1"; reads GROQ_API_KEY
        Ollama   — base_url="http://localhost:11434/v1"; api_key="ollama"

    Tool schemas are auto-converted from Anthropic format to OpenAI function format.
    System prompt is prepended as a system-role message at call time (not stored
    in history, so it can be regenerated dynamically per-call).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        provider_label: str = "openai",
    ) -> None:
        self.model = model
        self._provider_label = provider_label
        self._history: list[dict[str, Any]] = []
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None  # lazy: created on first chat() call

    @property
    def provider_name(self) -> str:
        return self._provider_label

    def add_user_message(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        # OpenAI: one message per tool result, role="tool"
        for r in results:
            self._history.append({
                "role": "tool",
                "tool_call_id": r.id,
                "content": r.content,
            })

    def chat(self, *, system: str, tools: list[dict[str, Any]], max_tokens: int) -> LLMResponse:
        if self._client is None:
            self._client = self._build_client(api_key=self._api_key, base_url=self._base_url)
        messages = [{"role": "system", "content": system}] + self._history
        openai_tools = _anthropic_to_openai_tools(tools)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        msg = choice.message

        # Build assistant history entry as a plain dict
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        self._history.append(assistant_entry)

        text = (msg.content or "").strip()

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    inputs = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inputs = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, inputs=inputs))

        needs_tool_use = choice.finish_reason == "tool_calls"

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            needs_tool_use=needs_tool_use,
        )

    def reset_history(self) -> None:
        self._history.clear()

    @staticmethod
    def _build_client(api_key: str | None, base_url: str | None) -> Any:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "pip install openai\n"
                "Then set the provider's API key environment variable."
            ) from e
        return openai.OpenAI(api_key=api_key, base_url=base_url)


# ── Schema conversion ─────────────────────────────────────────────────────────

def _anthropic_to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-format tool schemas to OpenAI function-calling format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


# ── Factory ───────────────────────────────────────────────────────────────────

_PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "class": "anthropic",
        "default_model": "claude-sonnet-4-6",
    },
    "openai": {
        "class": "openai",
        "default_model": "gpt-4o",
        "base_url": None,
    },
    "deepseek": {
        "class": "openai",
        "default_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "groq": {
        "class": "openai",
        "default_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "ollama": {
        "class": "openai",
        "default_model": "qwen2.5:14b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",   # Ollama doesn't need a real key
    },
}


def create_backend(
    provider: str = "anthropic",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMBackend:
    """Factory: create an LLMBackend by provider name.

    Args:
        provider: One of 'anthropic', 'openai', 'deepseek', 'groq', 'ollama',
                  or 'openai-compatible' (requires base_url).
        model:    Model name override. If None, uses each provider's default.
        base_url: API base URL override (useful for 'openai-compatible').
        api_key:  API key override. If None, reads from the standard env var.

    Returns:
        A ready-to-use LLMBackend instance.

    Examples:
        create_backend("anthropic")
        create_backend("openai", model="gpt-4o-mini")
        create_backend("deepseek")
        create_backend("groq", model="mixtral-8x7b-32768")
        create_backend("ollama", model="llama3.2:3b")
        create_backend("openai-compatible", model="my-model", base_url="http://...")
    """
    import os

    cfg = _PROVIDER_CONFIGS.get(provider)

    if provider == "openai-compatible":
        if not base_url:
            raise ValueError("'openai-compatible' provider requires base_url")
        return OpenAICompatibleBackend(
            model=model or "default",
            base_url=base_url,
            api_key=api_key,
            provider_label="openai-compatible",
        )

    if cfg is None:
        known = list(_PROVIDER_CONFIGS) + ["openai-compatible"]
        raise ValueError(f"Unknown provider '{provider}'. Known: {known}")

    resolved_model = model or cfg["default_model"]

    if cfg["class"] == "anthropic":
        return AnthropicBackend(model=resolved_model)

    # OpenAI-compatible
    resolved_url = base_url or cfg.get("base_url")
    resolved_key = api_key or cfg.get("api_key") or (
        os.environ.get(cfg["api_key_env"]) if "api_key_env" in cfg else None
    )
    return OpenAICompatibleBackend(
        model=resolved_model,
        base_url=resolved_url,
        api_key=resolved_key,
        provider_label=provider,
    )
