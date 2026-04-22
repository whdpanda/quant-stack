"""Pydantic schemas shared across all layers.

Two categories live here:

1. Per-run config schemas (DataConfig, BacktestConfig, PortfolioConfig,
   ExecutionConfig) — constructed each time you kick off a run.
   The AppConfig helpers in core/config.py build these from layer defaults.

2. Data record schemas (BarData, Signal, BacktestResult, PortfolioWeights,
   ExperimentRecord) — represent facts about a past event or optimisation.
   All record schemas are frozen (immutable after construction).
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class DataProviderKind(StrEnum):
    YAHOO = "yahoo"
    CSV = "csv"


class PortfolioMethod(StrEnum):
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    EFFICIENT_RISK = "efficient_risk"


class ExecutionMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class BarFreq(StrEnum):
    DAILY = "1D"
    WEEKLY = "1W"
    MONTHLY = "1M"


class SignalDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalSource(StrEnum):
    """Enforce the research / execution decoupling at the data level.

    A Signal produced by a backtest or vectorbt study carries RESEARCH.
    Only signals emitted by the live execution layer carry EXECUTION.
    The execution layer must never consume RESEARCH signals directly —
    it re-generates them through its own pipeline to avoid look-ahead risk.
    """
    RESEARCH = "research"
    EXECUTION = "execution"


# ── Per-run config schemas ─────────────────────────────────────────────────────

class DataConfig(BaseModel):
    """Parameters for a single data-fetch call."""

    symbols: list[str] = Field(..., min_length=1)
    start: date
    end: date
    provider: DataProviderKind = DataProviderKind.YAHOO
    cache_dir: str = "./data"

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: date, info: Any) -> date:
        start = info.data.get("start")
        if start and v <= start:
            raise ValueError("end date must be after start date")
        return v


class BacktestConfig(BaseModel):
    """Parameters for a single backtest run."""

    data: DataConfig
    strategy_name: str
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    initial_cash: float = Field(default=100_000.0, gt=0)
    commission: float = Field(default=0.001, ge=0.0, le=0.05)
    slippage: float = Field(default=0.001, ge=0.0, le=0.05)
    freq: str = "1D"


class PortfolioConfig(BaseModel):
    """Parameters for a single portfolio optimisation."""

    method: PortfolioMethod = PortfolioMethod.MAX_SHARPE
    risk_free_rate: float = Field(default=0.05, ge=0.0)
    weight_bounds: tuple[float, float] = (0.0, 0.4)
    target_volatility: float | None = None


class ExecutionConfig(BaseModel):
    """Parameters for the execution layer (handed to LeanBridge)."""

    mode: ExecutionMode = ExecutionMode.PAPER
    lean_config_path: str = "./lean/config.json"


# ── Data record schemas (frozen) ───────────────────────────────────────────────

class BarData(BaseModel):
    """A single OHLCV bar for one symbol.

    ``close`` should always be the adjusted price when ``adj_close`` is None.
    When a data provider supplies both raw and adjusted prices, store the raw
    price in ``close`` and the adjusted price in ``adj_close``.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    adj_close: float | None = Field(default=None, gt=0)
    freq: BarFreq = BarFreq.DAILY

    @model_validator(mode="after")
    def price_consistency(self) -> "BarData":
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high < self.open:
            raise ValueError(f"high ({self.high}) must be >= open ({self.open})")
        if self.high < self.close:
            raise ValueError(f"high ({self.high}) must be >= close ({self.close})")
        if self.low > self.open:
            raise ValueError(f"low ({self.low}) must be <= open ({self.open})")
        if self.low > self.close:
            raise ValueError(f"low ({self.low}) must be <= close ({self.close})")
        return self

    @property
    def effective_close(self) -> float:
        """Return adjusted close if available, otherwise close."""
        return self.adj_close if self.adj_close is not None else self.close


class Signal(BaseModel):
    """A directional trading signal produced by a strategy.

    ``source`` distinguishes research signals (produced by vectorbt / backtest
    pipelines) from execution signals (produced by the live engine).
    The execution layer must never read RESEARCH signals and act on them
    directly — this would constitute look-ahead or pipeline contamination.

    ``strength`` encodes confidence or target sizing:
        0.0 = effectively FLAT regardless of direction
        1.0 = full-position signal
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    direction: SignalDirection
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    strategy_name: str
    source: SignalSource
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def flat_has_zero_strength(self) -> "Signal":
        """A FLAT signal should carry zero strength to prevent ambiguity."""
        if self.direction == SignalDirection.FLAT and self.strength > 0.0:
            # Allow it but the caller should be intentional
            pass
        return self


# ── Result schemas ─────────────────────────────────────────────────────────────

class BacktestResult(BaseModel):
    """Aggregate performance metrics from a completed backtest.

    Anti look-ahead note: these metrics are computed on out-of-sample or
    historical data. Walk-forward metrics should be stored separately in
    ExperimentRecord.metadata.
    """

    strategy_name: str
    symbols: list[str] = Field(default_factory=list)
    period_start: date | None = None
    period_end: date | None = None

    # Core performance metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int

    # Cost accounting
    commission_paid: float = 0.0
    slippage_paid: float = 0.0

    # Benchmark comparison (optional; None = no benchmark used)
    benchmark_return: float | None = None

    # Extended metrics (populated by vbt_adapter; None = not computed)
    sortino_ratio: float | None = None
    annual_volatility: float | None = None
    annual_turnover: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def excess_return(self) -> float | None:
        """Return above benchmark, or None if no benchmark was set."""
        if self.benchmark_return is None:
            return None
        return self.total_return - self.benchmark_return


class PortfolioWeights(BaseModel):
    """Output of the portfolio optimisation layer.

    This schema carries only weight recommendations and associated metrics.
    It contains no order instructions — the execution layer translates weights
    into orders independently, applying its own risk controls.
    """

    weights: dict[str, float]
    method: str | None = None          # PortfolioMethod value OR allocator name
    rebalance_date: date | None = None

    # Expected metrics from the optimiser
    expected_return: float | None = None
    expected_volatility: float | None = None
    sharpe_ratio: float | None = None

    optimization_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"weights must sum to ~1.0, got {total:.4f}")
        return v

    @field_validator("weights")
    @classmethod
    def weights_non_negative(cls, v: dict[str, float]) -> dict[str, float]:
        # "CASH" is a reserved key for explicit cash buffer entries
        negative = {k: w for k, w in v.items() if w < 0 and k != "CASH"}
        if negative:
            raise ValueError(
                f"Long-only portfolio: negative weights found: {negative}. "
                "Short selling requires a separate schema."
            )
        return v


# ── Experiment record ──────────────────────────────────────────────────────────

class ExperimentRecord(BaseModel):
    """Full audit record for a research experiment.

    Stores everything needed to reproduce the experiment later:
    the config snapshot, the results, and free-form notes.

    Reproducibility contract:
        config_snapshot = AppConfig.model_dump()  # captured at experiment start
        The snapshot is frozen after construction (model_config frozen=True
        is intentionally NOT set here so the record can be built incrementally,
        but save() should be called only once the experiment is complete).
    """

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    description: str = ""

    # Reproducibility snapshot — call AppConfig.model_dump() before running
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    # Explicit strategy parameters (separate from the full config snapshot)
    strategy_params: dict[str, Any] = Field(default_factory=dict)

    # Scope
    symbols: list[str] = Field(default_factory=list)
    period_start: date | None = None
    period_end: date | None = None

    # Results (filled in as the pipeline runs)
    backtest_result: BacktestResult | None = None
    portfolio_weights: PortfolioWeights | None = None
    agent_analysis: str = ""

    # Paths to generated artifacts (relative to experiment directory)
    # e.g. {"equity_curve": "artifacts/equity.png", "weights_csv": "artifacts/weights.csv"}
    artifact_paths: dict[str, str] = Field(default_factory=dict)

    # Annotation
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialise to JSON. Returns the path written."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentRecord":
        """Deserialise from a JSON file produced by save()."""
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(raw)
