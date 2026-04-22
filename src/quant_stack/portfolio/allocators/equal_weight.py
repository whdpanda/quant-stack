"""Equal-weight allocator — simplest possible baseline."""

from __future__ import annotations

import pandas as pd

from quant_stack.portfolio.allocators.base import AllocationConstraints, BaseAllocator


class EqualWeightAllocator(BaseAllocator):
    """Assigns 1/N weight to every eligible asset."""

    name = "equal_weight"

    def __init__(self, constraints: AllocationConstraints | None = None) -> None:
        super().__init__(constraints)

    def _compute_raw_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        return self._equal_weight_dict(list(returns.columns))
