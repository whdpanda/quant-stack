"""LEAN algorithm skeleton — Sector ETF Momentum Strategy.

Purpose
-------
This file is designed to run inside the LEAN engine (QuantConnect Python runtime).
It does NOT recompute the momentum model.  Instead it reads a pre-computed
target-weights JSON payload written by quant_stack.execution.adapters.LeanExecutionAdapter
and calls LEAN's ``SetHoldings()`` API at each scheduled rebalance date.

Design principle
----------------
Weight computation stays entirely in the quant-stack research/execution layers.
LEAN is only responsible for order routing.  This keeps the strategy logic in
one place and eliminates research/execution drift.

File handoff
────────────
    quant_stack execution layer
        └── LeanExecutionAdapter.execute(plan, dry_run=False)
                └── writes  lean_output/target_weights.json
                                    ↓
    LEAN runtime (this file)
        └── _load_weights()  reads  lean_output/target_weights.json
        └── _rebalance()     calls  SetHoldings(symbol, weight)

Payload format expected at lean_output/target_weights.json::

    {
        "strategy_name": "sector_momentum_210d_top3",
        "rebalance_date": "2025-12-30",
        "generated_at": "2026-04-25T22:25:05",
        "weights": {
            "QQQ": 0.333333,
            "XLI": 0.333333,
            "GDX": 0.333333
        },
        "all_target_weights": { ... },
        "metadata": { "risk_checks_passed": true, ... }
    }

Running inside LEAN
-------------------
    # Install LEAN CLI
    pip install lean

    # Backtest
    lean backtest lean/SectorMomentumAlgorithm.py

    # Live (requires broker config in lean/config.json)
    lean live lean/SectorMomentumAlgorithm.py

Outside LEAN this file cannot execute (AlgorithmImports is not available).
It is kept in the repository as the bridge specification and skeleton.
"""

from __future__ import annotations

import json
from pathlib import Path

# ── LEAN runtime guard ────────────────────────────────────────────────────────
# AlgorithmImports is injected by the LEAN Python runtime.
# Outside LEAN this import fails — this is expected and intentional.
try:
    from AlgorithmImports import *  # type: ignore[import]  # noqa: F401,F403
    _LEAN_AVAILABLE = True
except ImportError:
    _LEAN_AVAILABLE = False


# ── Strategy configuration ────────────────────────────────────────────────────
# Mirror of the formal strategy parameters.
# These values must match RISK_ON_UNIVERSE and STRATEGY_PARAMS in:
#   src/quant_stack/research/strategies/sector_momentum.py
#   experiments/sector_momentum_experiment.py

STRATEGY_CONFIG: dict = {
    # Risk-on universe (IYT universe — consolidated 2026-04-25)
    "universe": ["IYT", "QQQ", "XLE", "XLV", "XLF", "XLI", "VTV", "GDX", "XLP"],
    "top_n": 3,
    # Bi-monthly rebalance: fire on odd calendar months (Jan=1, Mar=3, May=5, ...)
    # Equivalent to pandas rebalance_freq="2ME" starting from January.
    "rebalance_months": {1, 3, 5, 7, 9, 11},
    # Risk gates applied at the LEAN layer (secondary gate; primary is in RiskConfig)
    "max_position_size": 0.40,   # matches RiskConfig.max_position_size
    "cash_buffer": 0.02,         # keep 2% cash; avoid over-allocating due to rounding
    # Path to the weight payload written by LeanExecutionAdapter
    "weights_file": "./lean_output/target_weights.json",
    # Logging
    "log_every_bar": False,
}


# ── Algorithm class ───────────────────────────────────────────────────────────

if _LEAN_AVAILABLE:

    class SectorMomentumAlgorithm(QCAlgorithm):  # type: ignore[name-defined]
        """Sector ETF momentum algorithm driven by quant-stack target weights.

        This algorithm is intentionally thin.  All strategy logic lives in:
          - quant_stack.research.strategies.sector_momentum
          - quant_stack.execution.service.RebalanceService
          - quant_stack.execution.adapters.LeanExecutionAdapter

        LEAN's role: read pre-computed weights → place SetHoldings orders.
        """

        def Initialize(self) -> None:
            self.SetStartDate(2010, 1, 1)
            self.SetEndDate(2025, 12, 31)
            self.SetCash(100_000)
            self.SetBrokerageModel(
                BrokerageName.InteractiveBrokersBrokerage,  # type: ignore[name-defined]
                AccountType.Margin,                         # type: ignore[name-defined]
            )

            # Add universe assets — equity ETFs, daily resolution
            for ticker in STRATEGY_CONFIG["universe"]:
                self.AddEquity(ticker, Resolution.Daily)   # type: ignore[name-defined]

            # Schedule rebalance check: every month-end, 1 minute after open
            # The handler itself filters for bi-monthly cadence.
            self.Schedule.On(
                self.DateRules.MonthEnd(),                 # type: ignore[name-defined]
                self.TimeRules.AfterMarketOpen(            # type: ignore[name-defined]
                    STRATEGY_CONFIG["universe"][0], 1
                ),
                self._rebalance_if_scheduled,
            )

            self._target_weights: dict[str, float] = {}
            self._weights_file = Path(STRATEGY_CONFIG["weights_file"])

        def _rebalance_if_scheduled(self) -> None:
            """Fire on every month-end; only rebalance in configured months."""
            if self.Time.month not in STRATEGY_CONFIG["rebalance_months"]:
                return
            self._load_weights()
            self._rebalance()

        def _load_weights(self) -> None:
            """Load the quant-stack target-weights payload from disk."""
            if not self._weights_file.exists():
                self.Log(
                    f"WARNING: weights file not found: {self._weights_file}"
                    " — holding current positions"
                )
                return

            with self._weights_file.open(encoding="utf-8") as f:
                payload = json.load(f)

            meta = payload.get("metadata", {})
            if not meta.get("risk_checks_passed", True):
                self.Log(
                    "WARNING: risk_checks_passed=False in payload"
                    " — skipping rebalance"
                )
                return

            self._target_weights = payload.get("weights", {})
            self.Log(
                f"Loaded weights for {payload.get('rebalance_date')}: "
                f"{self._target_weights}"
            )

        def _rebalance(self) -> None:
            """Execute rebalance to target weights with LEAN-side risk gates."""
            if not self._target_weights:
                self.Log("No target weights available — holding current positions")
                return

            max_size = STRATEGY_CONFIG["max_position_size"]
            cash_buf = STRATEGY_CONFIG["cash_buffer"]

            # Allocate to target symbols
            for symbol, weight in self._target_weights.items():
                # Secondary position-size gate (primary is in RiskConfig)
                clamped = min(weight, max_size - cash_buf)
                if clamped != weight:
                    self.Log(
                        f"  {symbol}: weight clamped"
                        f" {weight:.2%} → {clamped:.2%}"
                    )
                self.SetHoldings(symbol, clamped)     # type: ignore[attr-defined]

            # Liquidate any holding not in the current target
            for holding in self.Portfolio.Values:     # type: ignore[attr-defined]
                sym = holding.Symbol.Value
                if holding.Invested and sym not in self._target_weights:
                    self.Log(f"  Liquidating {sym} (not in target)")
                    self.Liquidate(holding.Symbol)    # type: ignore[attr-defined]

        def OnData(self, data: object) -> None:  # noqa: N802
            # All logic is schedule-driven; OnData is intentionally empty.
            pass

else:
    # ── Stub class for documentation / type-checking outside LEAN ─────────────
    # This stub is never executed.  It exists only so IDE tools and tests can
    # import this module without LEAN installed.

    class SectorMomentumAlgorithm:  # type: ignore[no-redef]
        """Stub — only the LEAN runtime version is functional.

        See module docstring for usage instructions.
        """

        STRATEGY_CONFIG = STRATEGY_CONFIG

        def __repr__(self) -> str:
            return (
                "SectorMomentumAlgorithm [LEAN not available — "
                "install LEAN to run this algorithm]"
            )
