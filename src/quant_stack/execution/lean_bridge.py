"""LEAN Engine bridge — paper/live execution stub.

LEAN (https://www.lean.io/) is QuantConnect's open-source backtesting and
live-trading engine. Full integration requires:
  1. A running LEAN container or local installation.
  2. A configured lean/config.json with broker credentials.
  3. QuantConnect.Lean Python wrapper or REST API calls.

This stub validates the interface and logs intent without placing real orders.
Replace the body of `rebalance` and `get_positions` when wiring to LEAN.
"""

from __future__ import annotations

from loguru import logger

from quant_stack.core.schemas import ExecutionConfig, PortfolioWeights
from quant_stack.execution.base import Executor


class LeanBridge(Executor):
    """Stub executor that logs target weights without sending orders."""

    def __init__(self, config: ExecutionConfig) -> None:
        super().__init__(config)
        logger.warning(
            "LeanBridge is a stub — no real orders will be placed. "
            "Implement _send_order() to connect to LEAN."
        )

    def rebalance(self, weights: PortfolioWeights) -> None:
        logger.info(f"[LEAN stub] Target weights: {weights.weights}")
        for symbol, w in weights.weights.items():
            self._send_order(symbol, w)

    def get_positions(self) -> dict[str, float]:
        logger.info("[LEAN stub] get_positions → returning empty dict")
        return {}

    def _send_order(self, symbol: str, target_weight: float) -> None:
        """Override this to submit an order to LEAN via its REST API."""
        logger.debug(f"[LEAN stub] Would set {symbol} → {target_weight:.2%}")
