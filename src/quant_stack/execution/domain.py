"""Execution layer domain models.

These models represent facts about the execution process.
They live entirely within the execution boundary:

  research layer                   execution layer
  ─────────────────────────────    ──────────────────────────────────────
  PortfolioWeights (weight recs)→  TargetWeights (handoff schema)
  ExperimentRecord (audit log)     PositionSnapshot (current state)
                                   PositionDiff (per-symbol delta)
                                   RebalanceDecision (what + why)
                                   OrderIntent (one order)
                                   OrderPlan (all orders this cycle)
                                   RiskCheckResult (gate outcome)
                                   ExecutionResult (adapter outcome)

Research → Execution boundary
──────────────────────────────
Call ``target_weights_from_portfolio_weights()`` to cross the boundary.
This is the ONLY sanctioned crossing point.  No signal DataFrames,
no vectorbt portfolios, and no BacktestResult objects may cross into
the execution layer — they carry look-ahead context from the research
pipeline that must not drive live orders.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from quant_stack.core.schemas import PortfolioWeights


# ── Enums ─────────────────────────────────────────────────────────────────────

class OrderSide(StrEnum):
    BUY  = "buy"
    SELL = "sell"


# ── Position & target schemas ─────────────────────────────────────────────────

class PositionSnapshot(BaseModel):
    """Current portfolio state at a point in time.

    ``positions`` maps symbol → fraction of NAV.  This is always the authoritative
    weight used by the strategy layer.  For amount-driven input (new format),
    weights are derived from ``position_metadata.market_value_usd / nav``; they
    are NOT loaded as raw percentages, so precision is preserved.

    ``cash_fraction + sum(positions.values())`` should ≈ 1.0.

    ``position_metadata`` carries per-symbol quantity/price/market-value from the
    input file.  It is optional and never used by the strategy layer — only by the
    display and artifact layers.  Defaults to {} for backward compatibility.

    ``source`` records where this snapshot came from:
      "manual"  — hand-constructed (e.g., for testing or first run)
      "paper"   — retrieved from PaperExecutionAdapter's internal state
      "broker"  — pulled from a live broker API (future)
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    nav: float = Field(..., gt=0, description="Total portfolio value (account currency).")
    positions: dict[str, float] = Field(default_factory=dict)
    cash_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "manual"
    position_metadata: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "Optional per-symbol metadata (quantity, last_price_usd, market_value_usd). "
            "Populated by the amount-driven input format; empty for legacy weight format. "
            "Never used by strategy or risk logic."
        ),
    )


class TargetWeights(BaseModel):
    """Target allocation produced by the research / strategy layer.

    This is the sole handoff schema from research to execution.
    ``weights`` sum to ≤ 1.0; any remainder is an implicit cash buffer.

    Do NOT construct this directly from strategy signals.  Always use
    ``target_weights_from_portfolio_weights()`` so the crossing point
    is explicit and auditable.
    """

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    rebalance_date: date
    weights: dict[str, float]
    generated_at: datetime = Field(default_factory=datetime.now)
    source_record_id: str = ""   # ExperimentRecord.experiment_id, if available


# ── Diff & decision schemas ───────────────────────────────────────────────────

class PositionDiff(BaseModel):
    """Per-symbol weight delta between target and current holdings."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    current_weight: float
    target_weight: float
    delta_weight: float   # = target - current; positive → buy, negative → sell

    @property
    def side(self) -> OrderSide:
        return OrderSide.BUY if self.delta_weight >= 0 else OrderSide.SELL


class RebalanceDecision(BaseModel):
    """Record of what changed and why in one rebalance cycle.

    ``all_diffs`` covers every symbol in target ∪ current.
    ``actionable`` is the filtered subset that exceeded ``min_trade_size``.
    Only actionable diffs become OrderIntents.
    """

    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    target: TargetWeights
    snapshot: PositionSnapshot
    all_diffs: list[PositionDiff]
    actionable: list[PositionDiff]


# ── Order schemas ─────────────────────────────────────────────────────────────

class OrderIntent(BaseModel):
    """A single order instruction derived from a PositionDiff.

    ``delta_value`` and ``target_value`` are informational (in account currency).
    The adapter translates these into broker-specific order types.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: OrderSide
    target_weight: float
    delta_weight: float
    nav: float

    @property
    def delta_value(self) -> float:
        """Absolute change in $ (always positive)."""
        return abs(self.delta_weight) * self.nav

    @property
    def target_value(self) -> float:
        """Target $ value of this position after rebalance."""
        return self.target_weight * self.nav


class OrderPlan(BaseModel):
    """Complete set of order intents for one rebalance cycle.

    ``approved`` starts False.  The RebalanceService sets it to True
    only after all risk checks pass and dry_run=False.
    """

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    decision: RebalanceDecision
    orders: list[OrderIntent]
    total_turnover: float   # sum(|delta_weight|) across all actionable diffs
    estimated_cost_bps: float = 0.0
    approved: bool = False


# ── Risk check schemas ────────────────────────────────────────────────────────

class RiskViolation(BaseModel):
    """One violated risk rule."""

    model_config = ConfigDict(frozen=True)

    rule: str
    value: float
    limit: float
    message: str


class RiskCheckResult(BaseModel):
    """Aggregate outcome of all risk checks against an OrderPlan."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    violations: list[RiskViolation] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.passed:
            return "All risk checks passed."
        return "; ".join(v.message for v in self.violations)


# ── Execution result ──────────────────────────────────────────────────────────

class ExecutionResult(BaseModel):
    """Outcome returned by an adapter after executing (or simulating) an OrderPlan.

    ``lean_payload`` is populated only by LeanExecutionAdapter; it carries
    the JSON structure that the LEAN algorithm skeleton reads from disk.
    """

    plan_id: str
    executed_at: datetime = Field(default_factory=datetime.now)
    adapter_mode: str   # "dry_run" | "paper" | "lean" | "blocked"
    orders_attempted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    estimated_cost: float = 0.0
    risk_check: RiskCheckResult | None = None
    log_entries: list[str] = Field(default_factory=list)
    lean_payload: dict[str, Any] = Field(default_factory=dict)
    success: bool = True


# ── Research → Execution boundary helper ─────────────────────────────────────

def target_weights_from_portfolio_weights(
    pw: PortfolioWeights,
    strategy_name: str,
    source_record_id: str = "",
) -> TargetWeights:
    """Convert a research-layer PortfolioWeights into an execution TargetWeights.

    This is the ONLY sanctioned crossing point from research to execution.
    The caller must supply ``strategy_name`` because PortfolioWeights carries
    no strategy label — it is a pure weight container.

    CASH entries are silently dropped: the execution layer treats any weight
    not allocated to a named symbol as implicit cash buffer.

    Args:
        pw:               PortfolioWeights from the research / portfolio layer.
        strategy_name:    Name of the strategy that produced these weights.
        source_record_id: ExperimentRecord.experiment_id for traceability.

    Returns:
        TargetWeights ready for RebalanceService.run().
    """
    rebalance_date = pw.rebalance_date or date.today()
    weights = {k: v for k, v in pw.weights.items() if k != "CASH"}
    return TargetWeights(
        strategy_name=strategy_name,
        rebalance_date=rebalance_date,
        weights=weights,
        source_record_id=source_record_id,
    )
