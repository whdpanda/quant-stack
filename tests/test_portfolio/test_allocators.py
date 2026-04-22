"""Tests for portfolio allocators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_stack.core.schemas import PortfolioWeights
from quant_stack.portfolio.allocators import (
    AllocationConstraints,
    EqualWeightAllocator,
    InverseVolatilityAllocator,
)
from quant_stack.portfolio.allocators.base import BaseAllocator


# ── AllocationConstraints ─────────────────────────────────────────────────────

class TestAllocationConstraints:
    def test_defaults(self) -> None:
        c = AllocationConstraints()
        assert c.min_weight == 0.0
        assert c.max_weight == 1.0
        assert c.cash_buffer == 0.0
        assert c.min_assets == 1

    def test_max_gte_min_passes(self) -> None:
        AllocationConstraints(min_weight=0.05, max_weight=0.40)

    def test_max_lt_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_weight"):
            AllocationConstraints(min_weight=0.5, max_weight=0.3)

    def test_cash_buffer_lt_1(self) -> None:
        with pytest.raises(ValueError):
            AllocationConstraints(cash_buffer=1.0)

    def test_frozen(self) -> None:
        c = AllocationConstraints()
        with pytest.raises(Exception):
            c.min_weight = 0.1  # type: ignore[misc]


# ── EqualWeightAllocator ──────────────────────────────────────────────────────

class TestEqualWeightAllocator:
    def test_returns_portfolio_weights(self, returns_3) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_3)
        assert isinstance(pw, PortfolioWeights)

    def test_equal_weights(self, returns_3) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_3)
        weights = list(pw.weights.values())
        assert all(abs(w - weights[0]) < 1e-10 for w in weights)

    def test_weights_sum_to_one(self, returns_3) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_3)
        assert abs(sum(pw.weights.values()) - 1.0) < 1e-9

    def test_n_assets(self, returns_3) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_3)
        assert len(pw.weights) == 3

    def test_eligible_filter(self, returns_3) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_3, eligible=["SPY", "QQQ"])
        assert "IEF" not in pw.weights
        assert len(pw.weights) == 2
        assert abs(sum(pw.weights.values()) - 1.0) < 1e-9

    def test_method_name(self, returns_3) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_3)
        assert pw.method == "equal_weight"

    def test_cash_buffer_applied(self, returns_3) -> None:
        c = AllocationConstraints(cash_buffer=0.10)
        alloc = EqualWeightAllocator(constraints=c)
        pw = alloc.allocate(returns_3)
        assert "CASH" in pw.weights
        assert abs(pw.weights["CASH"] - 0.10) < 1e-9
        assert abs(sum(pw.weights.values()) - 1.0) < 1e-9

    def test_max_weight_capped(self, returns_3) -> None:
        c = AllocationConstraints(max_weight=0.2)
        alloc = EqualWeightAllocator(constraints=c)
        pw = alloc.allocate(returns_3)
        for sym, w in pw.weights.items():
            if sym != "CASH":
                assert w <= 0.2 + 1e-9

    def test_single_asset(self, returns_1) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_1)
        assert abs(sum(pw.weights.values()) - 1.0) < 1e-9


# ── InverseVolatilityAllocator ────────────────────────────────────────────────

class TestInverseVolatilityAllocator:
    def test_weights_sum_to_one(self, returns_3) -> None:
        alloc = InverseVolatilityAllocator()
        pw = alloc.allocate(returns_3)
        assert abs(sum(pw.weights.values()) - 1.0) < 1e-9

    def test_lower_vol_gets_higher_weight(self, returns_3) -> None:
        # IEF has lower vol in our fixture (scale=0.01 but different mean)
        # We construct a cleaner test: A has half the vol of B
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=300, freq="B")
        data = np.column_stack([
            rng.normal(0, 0.005, 300),   # A: low vol
            rng.normal(0, 0.020, 300),   # B: high vol
        ])
        rets = pd.DataFrame(data, index=idx, columns=["A", "B"])
        alloc = InverseVolatilityAllocator()
        pw = alloc.allocate(rets)
        assert pw.weights["A"] > pw.weights["B"]

    def test_flat_returns_fallback(self, flat_returns) -> None:
        # Zero-vol assets fall back to equal weight without crashing
        alloc = InverseVolatilityAllocator()
        pw = alloc.allocate(flat_returns)
        assert abs(sum(pw.weights.values()) - 1.0) < 1e-9

    def test_eligible_filter(self, returns_3) -> None:
        alloc = InverseVolatilityAllocator()
        pw = alloc.allocate(returns_3, eligible=["SPY", "IEF"])
        assert "QQQ" not in pw.weights

    def test_method_name(self, returns_3) -> None:
        alloc = InverseVolatilityAllocator()
        pw = alloc.allocate(returns_3)
        assert pw.method == "inverse_volatility"


# ── Fallback behaviour ────────────────────────────────────────────────────────

class TestFallbackBehaviour:
    def test_empty_eligible_triggers_fallback(self, returns_3) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_3, eligible=[])
        assert pw.method == "equal_weight_fallback"

    def test_nonexistent_eligible_triggers_fallback(self, returns_3) -> None:
        alloc = EqualWeightAllocator()
        pw = alloc.allocate(returns_3, eligible=["TSLA"])
        assert pw.method == "equal_weight_fallback"

    def test_broken_allocator_falls_back(self, returns_3) -> None:
        """A subclass that always raises should fall back to equal weight."""

        class BrokenAllocator(BaseAllocator):
            name = "broken"

            def _compute_raw_weights(self, returns: pd.DataFrame) -> dict[str, float]:
                raise RuntimeError("intentional failure")

        alloc = BrokenAllocator()
        pw = alloc.allocate(returns_3)
        assert abs(sum(pw.weights.values()) - 1.0) < 1e-9
        assert len(pw.weights) == 3
