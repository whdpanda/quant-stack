"""AI research assistant — asks Claude to analyse backtest results and suggest improvements."""

from __future__ import annotations

import json

from loguru import logger

from quant_stack.core.exceptions import AgentError
from quant_stack.core.schemas import BacktestResult


_SYSTEM_PROMPT = """\
You are a quantitative research assistant. You receive backtest performance statistics
and help the researcher understand results, identify weaknesses, and suggest parameter
or strategy improvements. Be concise, rigorous, and cite statistical limitations.
"""


class Researcher:
    """Wraps the Anthropic API to provide research commentary on backtest results."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 2048) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = self._build_client()

    def analyse_backtest(
        self,
        result: BacktestResult,
        context: str = "",
    ) -> str:
        """Ask the model to comment on a backtest result.

        Args:
            result: Completed backtest metrics.
            context: Optional free-text context (e.g. universe, market regime).

        Returns:
            Model's analysis as a markdown string.
        """
        payload = result.model_dump(exclude={"metadata"})
        user_msg = (
            f"Here are the backtest results for strategy '{result.strategy_name}':\n\n"
            f"```json\n{json.dumps(payload, indent=2)}\n```\n\n"
            f"{('Additional context: ' + context) if context else ''}\n\n"
            "Please analyse these results: explain what the metrics imply, "
            "flag any concerns, and suggest 2-3 concrete improvements."
        )

        logger.debug(f"Sending backtest analysis request to {self.model}")
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------

    @staticmethod
    def _build_client() -> "anthropic.Anthropic":  # type: ignore[name-defined]
        try:
            import anthropic
        except ImportError as e:
            raise AgentError("anthropic is not installed: pip install anthropic") from e
        return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
