"""Abstract executor interface consumed by the deployment layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from quant_stack.core.schemas import ExecutionConfig, PortfolioWeights


class Executor(ABC):
    """Submit target portfolio weights to a broker / engine."""

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    @abstractmethod
    def rebalance(self, weights: PortfolioWeights) -> None:
        """Rebalance the live (or paper) portfolio to the target weights.

        Implementations must be idempotent: calling with the same weights
        a second time should result in no additional orders.
        """

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Return current positions as {symbol: fraction of NAV}."""
