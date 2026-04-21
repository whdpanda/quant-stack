"""Unified application configuration.

All six layers (data / strategy / portfolio / backtest / reporting / execution)
are expressed as frozen Pydantic models that compose into AppConfig.

Loading order:
    default values in this file
    → values from a YAML file (via load_config)
    → caller-level overrides (construct a new AppConfig manually)

Env-var overrides are intentionally NOT handled here; the CLI or entry
script should read os.environ before calling load_config().
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_stack.core.schemas import (
    DataProviderKind,
    ExecutionMode,
    PortfolioMethod,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _positive(v: float, name: str) -> float:
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _fraction(v: float, name: str) -> float:
    if not (0.0 < v <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {v}")
    return v


# ── Layer configs ──────────────────────────────────────────────────────────────

class DataLayerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_dir: Path = Path("./data")
    default_provider: DataProviderKind = DataProviderKind.YAHOO
    default_start: date = date(2015, 1, 1)
    default_end: date = date(2024, 12, 31)
    adjust_prices: bool = True
    fill_method: Literal["ffill", "bfill", "none"] = "ffill"
    fill_limit: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def end_after_start(self) -> "DataLayerConfig":
        if self.default_end <= self.default_start:
            raise ValueError("default_end must be after default_start")
        return self


class StrategyLayerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_freq: str = "1D"
    universe: list[str] = Field(default_factory=list)
    lookback_days: int = Field(default=252, gt=0)


class PortfolioLayerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: PortfolioMethod = PortfolioMethod.MAX_SHARPE
    risk_free_rate: float = Field(default=0.05, ge=0.0)
    weight_bounds: tuple[float, float] = (0.0, 0.4)
    target_volatility: float | None = None
    rebalance_freq: Literal["daily", "weekly", "monthly", "quarterly"] = "monthly"

    @model_validator(mode="after")
    def bounds_valid(self) -> "PortfolioLayerConfig":
        lo, hi = self.weight_bounds
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"weight_bounds must satisfy 0 ≤ lo < hi ≤ 1, got {self.weight_bounds}"
            )
        return self

    @model_validator(mode="after")
    def target_vol_required_for_efficient_risk(self) -> "PortfolioLayerConfig":
        if self.method == PortfolioMethod.EFFICIENT_RISK and self.target_volatility is None:
            raise ValueError("target_volatility is required when method=efficient_risk")
        return self


class BacktestLayerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    initial_cash: float = Field(default=100_000.0, gt=0)
    commission: float = Field(default=0.001, ge=0.0, le=0.05)
    slippage: float = Field(default=0.001, ge=0.0, le=0.05)
    freq: str = "1D"


class ReportingLayerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path = Path("./reports")
    formats: list[Literal["markdown", "html", "json"]] = Field(default_factory=lambda: ["markdown"])
    include_plots: bool = False


class RiskConfig(BaseModel):
    """Hard risk controls enforced by the execution layer.

    These values are NOT guidelines — the LeanBridge (and any live executor)
    must halt or refuse to place orders when any limit is breached.
    Research and portfolio layers must NOT reference this config.
    """

    model_config = ConfigDict(frozen=True)

    max_position_size: float = Field(
        default=0.40,
        description="Maximum fraction of NAV for any single position.",
    )
    max_drawdown_halt: float = Field(
        default=0.15,
        description="Halt all new orders if portfolio drawdown exceeds this.",
    )
    daily_loss_limit: float = Field(
        default=0.03,
        description="Halt all new orders if intraday P&L loss exceeds this fraction of NAV.",
    )

    @field_validator("max_position_size", "max_drawdown_halt", "daily_loss_limit", mode="before")
    @classmethod
    def must_be_positive_fraction(cls, v: Any) -> float:
        v = float(v)
        if not (0.0 < v < 1.0):
            raise ValueError(f"Risk limit must be in (0, 1), got {v}")
        return v

    @model_validator(mode="after")
    def drawdown_exceeds_daily(self) -> "RiskConfig":
        if self.max_drawdown_halt <= self.daily_loss_limit:
            raise ValueError(
                "max_drawdown_halt should be larger than daily_loss_limit "
                f"(got {self.max_drawdown_halt} ≤ {self.daily_loss_limit})"
            )
        return self


class ExecutionLayerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: ExecutionMode = ExecutionMode.PAPER
    lean_config_path: Path = Path("./lean/config.json")
    risk: RiskConfig = Field(default_factory=RiskConfig)


# ── Root config ────────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    """Root application configuration.

    Compose all layer configs. Instantiate directly or via load_config().
    Once constructed, the object is frozen — create a new instance to change values.
    """

    model_config = ConfigDict(frozen=True)

    data: DataLayerConfig = Field(default_factory=DataLayerConfig)
    strategy: StrategyLayerConfig = Field(default_factory=StrategyLayerConfig)
    portfolio: PortfolioLayerConfig = Field(default_factory=PortfolioLayerConfig)
    backtest: BacktestLayerConfig = Field(default_factory=BacktestLayerConfig)
    reporting: ReportingLayerConfig = Field(default_factory=ReportingLayerConfig)
    execution: ExecutionLayerConfig = Field(default_factory=ExecutionLayerConfig)

    # ------------------------------------------------------------------
    # Convenience constructors for per-run schemas

    def to_data_config(
        self,
        symbols: list[str],
        start: date | None = None,
        end: date | None = None,
    ) -> "quant_stack.core.schemas.DataConfig":  # type: ignore[name-defined]
        """Build a DataConfig from layer defaults + caller overrides."""
        from quant_stack.core.schemas import DataConfig
        return DataConfig(
            symbols=symbols,
            start=start or self.data.default_start,
            end=end or self.data.default_end,
            provider=self.data.default_provider,
            cache_dir=str(self.data.cache_dir),
        )

    def to_backtest_config(
        self,
        symbols: list[str],
        strategy_name: str,
        start: date | None = None,
        end: date | None = None,
        **strategy_params: Any,
    ) -> "quant_stack.core.schemas.BacktestConfig":  # type: ignore[name-defined]
        """Build a BacktestConfig from layer defaults."""
        from quant_stack.core.schemas import BacktestConfig
        return BacktestConfig(
            data=self.to_data_config(symbols, start, end),
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            initial_cash=self.backtest.initial_cash,
            commission=self.backtest.commission,
            slippage=self.backtest.slippage,
            freq=self.backtest.freq,
        )

    def to_portfolio_config(self) -> "quant_stack.core.schemas.PortfolioConfig":  # type: ignore[name-defined]
        """Build a PortfolioConfig from layer defaults."""
        from quant_stack.core.schemas import PortfolioConfig
        return PortfolioConfig(
            method=self.portfolio.method,
            risk_free_rate=self.portfolio.risk_free_rate,
            weight_bounds=self.portfolio.weight_bounds,
            target_volatility=self.portfolio.target_volatility,
        )


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config/settings.yaml") -> AppConfig:
    """Load AppConfig from a YAML file.

    Missing sections fall back to their pydantic defaults.
    The loaded values are deep-merged into the default AppConfig structure,
    so a partial YAML is valid — only override what you need.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated, frozen AppConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        pydantic.ValidationError: If any value fails validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")

    with p.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig.model_validate(raw)
