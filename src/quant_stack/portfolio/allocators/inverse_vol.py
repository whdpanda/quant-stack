"""Inverse-volatility allocator — weight proportional to 1/σ."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_stack.portfolio.allocators.base import AllocationConstraints, BaseAllocator


class InverseVolatilityAllocator(BaseAllocator):
    """Weights assets in proportion to 1 / realised volatility.

    Uses the full returns series supplied to ``allocate()``.  Symbols with
    zero volatility receive zero weight (avoids division by zero).
    """

    name = "inverse_volatility"

    def __init__(
        self,
        constraints: AllocationConstraints | None = None,
        annualize: bool = True,
    ) -> None:
        super().__init__(constraints)
        self.annualize = annualize

    def _compute_raw_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        vol = returns.std(ddof=1)
        if self.annualize:
            vol = vol * np.sqrt(252)

        # Guard: replace zero vol with NaN so those assets are excluded
        vol = vol.replace(0.0, np.nan).dropna()
        if vol.empty:
            return self._equal_weight_dict(list(returns.columns))

        inv_vol = 1.0 / vol
        total = inv_vol.sum()
        weights = (inv_vol / total).to_dict()
        return weights
