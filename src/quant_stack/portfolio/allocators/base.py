"""Abstract base class and constraint model for portfolio allocators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_stack.core.schemas import PortfolioWeights


class AllocationConstraints(BaseModel):
    """Weight constraints applied after optimization."""

    model_config = ConfigDict(frozen=True)

    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    cash_buffer: float = Field(default=0.0, ge=0.0, lt=1.0)
    min_assets: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def max_gte_min(self) -> "AllocationConstraints":
        if self.max_weight < self.min_weight:
            raise ValueError(
                f"max_weight ({self.max_weight}) must be >= min_weight ({self.min_weight})"
            )
        return self


class BaseAllocator(ABC):
    """Common interface for all portfolio allocators.

    Subclasses implement ``_compute_raw_weights()``.  The base class applies
    constraints, normalises, handles fallback, and wraps in PortfolioWeights.
    """

    name: str = "base"

    def __init__(self, constraints: AllocationConstraints | None = None) -> None:
        self.constraints = constraints or AllocationConstraints()

    # ------------------------------------------------------------------
    # Public API

    def allocate(
        self,
        returns: pd.DataFrame,
        eligible: list[str] | None = None,
        rebalance_date: date | None = None,
    ) -> PortfolioWeights:
        """Compute constrained portfolio weights.

        Args:
            returns: Daily simple returns, DatetimeIndex × symbol columns.
            eligible: Subset of symbol names to consider.  ``None`` = all.
            rebalance_date: Date label for the resulting PortfolioWeights.

        Returns:
            PortfolioWeights with normalised, constrained weights.
        """
        universe = self._filter_universe(returns, eligible)
        if universe.empty:
            return self._equal_fallback(list(returns.columns), rebalance_date, reason="empty universe")

        try:
            raw = self._compute_raw_weights(universe)
        except Exception as exc:
            from loguru import logger
            logger.warning(
                f"{self.name}: optimization failed ({exc}); falling back to equal weight"
            )
            raw = self._equal_weight_dict(list(universe.columns))

        constrained = self._apply_constraints(raw)
        if not constrained:
            return self._equal_fallback(list(returns.columns), rebalance_date, reason="no assets after constraints")

        return PortfolioWeights(
            weights=constrained,
            method=self.name,
            rebalance_date=rebalance_date or date.today(),
            optimization_metadata=self._metadata(universe),
        )

    # ------------------------------------------------------------------
    # Abstract method

    @abstractmethod
    def _compute_raw_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        """Compute unconstrained weights.  Must return a dict that sums to ~1."""
        ...

    # ------------------------------------------------------------------
    # Helpers

    def _filter_universe(
        self, returns: pd.DataFrame, eligible: list[str] | None
    ) -> pd.DataFrame:
        if eligible is not None:
            cols = [c for c in eligible if c in returns.columns]
            returns = returns[cols]
        # Drop columns with all-NaN
        returns = returns.dropna(axis=1, how="all")
        # Keep only rows where all remaining columns have data
        return returns.dropna(how="any")

    def _apply_constraints(self, raw: dict[str, float]) -> dict[str, float]:
        c = self.constraints
        investable = 1.0 - c.cash_buffer

        # Normalise raw weights to investable budget
        total_raw = sum(raw.values())
        if total_raw <= 0:
            return {}
        weights = {k: v / total_raw * investable for k, v in raw.items()}

        # Iterative capping: redistribute excess from capped to uncapped
        for _ in range(len(weights) + 1):
            over = {k for k, v in weights.items() if v > c.max_weight + 1e-12}
            if not over:
                break
            excess = sum(weights[k] - c.max_weight for k in over)
            for k in over:
                weights[k] = c.max_weight
            free = {k for k, v in weights.items() if v < c.max_weight - 1e-12}
            if not free:
                break
            free_total = sum(weights[k] for k in free)
            if free_total <= 0:
                break
            for k in free:
                weights[k] += excess * weights[k] / free_total

        # Apply min_weight floor
        weights = {k: v for k, v in weights.items() if v >= c.min_weight}

        if len(weights) < c.min_assets:
            return {}

        total = sum(weights.values())
        if total <= 0:
            return {}

        # If cap prevents filling the full investable budget, put the gap in CASH
        remainder = investable - total
        cash = c.cash_buffer
        if remainder > 1e-9:
            cash += remainder
        if cash > 0:
            weights["CASH"] = cash

        return weights

    @staticmethod
    def _equal_weight_dict(symbols: list[str]) -> dict[str, float]:
        n = len(symbols)
        return {s: 1.0 / n for s in symbols} if n > 0 else {}

    def _equal_fallback(
        self,
        symbols: list[str],
        rebalance_date: date | None,
        reason: str = "",
    ) -> PortfolioWeights:
        from loguru import logger
        logger.warning(f"{self.name}: using equal-weight fallback ({reason})")
        weights = self._equal_weight_dict(symbols)
        return PortfolioWeights(
            weights=weights,
            method="equal_weight_fallback",
            rebalance_date=rebalance_date or date.today(),
            optimization_metadata={"fallback_reason": reason},
        )

    def _metadata(self, returns: pd.DataFrame) -> dict:
        return {
            "n_assets": len(returns.columns),
            "n_periods": len(returns),
        }
