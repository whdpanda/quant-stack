"""Mean-variance allocator wrapping PyPortfolioOpt."""

from __future__ import annotations

import pandas as pd

from quant_stack.core.schemas import PortfolioMethod
from quant_stack.portfolio.allocators.base import AllocationConstraints, BaseAllocator


class MeanVarianceAllocator(BaseAllocator):
    """PyPortfolioOpt-backed mean-variance optimiser.

    Requires the ``[portfolio]`` extra: ``pip install 'quant-stack[portfolio]'``.
    Falls back to equal weight if PyPortfolioOpt is unavailable or the
    optimisation fails (e.g., singular covariance matrix).
    """

    name = "mean_variance"

    def __init__(
        self,
        method: PortfolioMethod = PortfolioMethod.MAX_SHARPE,
        risk_free_rate: float = 0.05,
        target_volatility: float | None = None,
        constraints: AllocationConstraints | None = None,
    ) -> None:
        super().__init__(constraints)
        self.method = method
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

        # Derive per-asset max_weight from constraints for pypfopt bounds
        c = self.constraints
        self._weight_bounds = (c.min_weight, c.max_weight)

    def _compute_raw_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        try:
            from pypfopt import expected_returns, risk_models
            from pypfopt.efficient_frontier import EfficientFrontier
        except ImportError as exc:
            raise RuntimeError(
                "PyPortfolioOpt not installed: pip install 'quant-stack[portfolio]'"
            ) from exc

        mu = expected_returns.mean_historical_return(returns, returns_data=True)
        S = risk_models.sample_cov(returns, returns_data=True)
        ef = EfficientFrontier(mu, S, weight_bounds=self._weight_bounds)

        match self.method:
            case PortfolioMethod.MAX_SHARPE:
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            case PortfolioMethod.MIN_VOLATILITY:
                ef.min_volatility()
            case PortfolioMethod.EFFICIENT_RISK:
                if self.target_volatility is None:
                    raise ValueError("target_volatility must be set for efficient_risk method")
                ef.efficient_risk(target_volatility=self.target_volatility)
            case _:
                raise ValueError(f"Unknown method: {self.method}")

        cleaned = ef.clean_weights()
        return {k: v for k, v in cleaned.items() if v > 1e-6}

    def _metadata(self, returns: pd.DataFrame) -> dict:
        meta = super()._metadata(returns)
        meta["method"] = self.method.value
        meta["risk_free_rate"] = self.risk_free_rate
        return meta
