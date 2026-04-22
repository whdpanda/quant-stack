"""Hierarchical Risk Parity allocator wrapping PyPortfolioOpt."""

from __future__ import annotations

import pandas as pd

from quant_stack.portfolio.allocators.base import AllocationConstraints, BaseAllocator


class HRPAllocator(BaseAllocator):
    """Hierarchical Risk Parity via PyPortfolioOpt.

    HRP does not require expected-return estimates, making it more robust
    than mean-variance for out-of-sample allocation.

    Requires the ``[portfolio]`` extra: ``pip install 'quant-stack[portfolio]'``.
    Falls back to equal weight on failure.
    """

    name = "hrp"

    def __init__(self, constraints: AllocationConstraints | None = None) -> None:
        super().__init__(constraints)

    def _compute_raw_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        try:
            from pypfopt.hierarchical_portfolio import HRPOpt
        except ImportError as exc:
            raise RuntimeError(
                "PyPortfolioOpt not installed: pip install 'quant-stack[portfolio]'"
            ) from exc

        hrp = HRPOpt(returns=returns)
        hrp.optimize()
        cleaned = hrp.clean_weights()
        return {k: v for k, v in cleaned.items() if v > 1e-6}
