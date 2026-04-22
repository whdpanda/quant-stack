"""SignalFrame — the unified output schema for all signal generators.

Design rationale
----------------
Signal generators produce bulk signals for a universe of symbols over a
time range. These are represented as DataFrames (not individual Signal
objects) because they are consumed by vectorized research tools.

The individual Signal schema (core/schemas.py) is for point-in-time
dispatch to the execution layer. SignalFrame provides a bridge:
    SignalFrame.to_long_df()   → tidy DataFrame (for analysis / storage)
    SignalFrame.latest()       → list[Signal] (for paper/live dispatch)

Decoupling guarantee
--------------------
SignalFrame.source is always SignalSource.RESEARCH. The execution layer
must NEVER read a RESEARCH SignalFrame and act on it directly. It must
re-derive signals through its own pipeline to ensure:
    1. No stale signal contamination
    2. Clean audit trail
    3. Ability to apply execution-side filters (risk limits, etc.)

Fields
------
signals : pd.DataFrame
    Float 0.0 / 1.0 (or fractional for partial positions).
    1.0 = long, 0.0 = flat. No short signals in this version.
    Index: DatetimeIndex, Columns: symbol strings.
    NaN = insufficient data (not flat — callers must handle NaN explicitly).

strength : pd.DataFrame
    Float [0, 1]. For binary signals, identical to `signals`.
    For ranked signals, encodes relative conviction.

ranks : pd.DataFrame | None
    Cross-sectional rank (1 = best) for each symbol on each date.
    Only populated by ranking-based signal generators.

eligible : pd.DataFrame | None
    Boolean eligibility mask applied before ranking/selection.
    e.g. "symbol passes trend filter" before momentum ranking.
    None = no eligibility filter applied.

strategy_name : str
    Human-readable identifier for this signal set.

source : SignalSource
    Always RESEARCH. Execution layer uses this to refuse direct consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from quant_stack.core.schemas import Signal, SignalDirection, SignalSource

if TYPE_CHECKING:
    pass


@dataclass
class SignalFrame:
    """Bulk signal output for a universe of symbols over a time range."""

    signals: pd.DataFrame
    strength: pd.DataFrame
    strategy_name: str
    ranks: pd.DataFrame | None = None
    eligible: pd.DataFrame | None = None
    source: SignalSource = field(default=SignalSource.RESEARCH)
    generated_at: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    # Validation

    def __post_init__(self) -> None:
        if self.signals.shape != self.strength.shape:
            raise ValueError(
                f"signals and strength must have the same shape: "
                f"{self.signals.shape} vs {self.strength.shape}"
            )
        if not self.signals.columns.equals(self.strength.columns):
            raise ValueError("signals and strength must have identical columns")

    # ------------------------------------------------------------------
    # Accessors

    @property
    def symbols(self) -> list[str]:
        return list(self.signals.columns)

    @property
    def n_dates(self) -> int:
        return len(self.signals)

    # ------------------------------------------------------------------
    # Conversion helpers

    def to_long_df(self) -> pd.DataFrame:
        """Return tidy long-format DataFrame compatible with Signal schema.

        Columns: date, symbol, direction, strength, strategy_name, source
        Rows with NaN strength are excluded.
        """
        df = self.strength.copy()
        df.index.name = "date"  # ensure the index is named before reset_index()
        melted = (
            df
            .reset_index()
            .melt(id_vars="date", var_name="symbol", value_name="strength")
            .dropna(subset=["strength"])
        )
        melted["direction"] = melted["strength"].apply(
            lambda s: SignalDirection.LONG.value if s > 0.0 else SignalDirection.FLAT.value
        )
        melted["strategy_name"] = self.strategy_name
        melted["source"] = self.source.value

        if self.ranks is not None:
            rank_df = self.ranks.copy()
            rank_df.index.name = "date"
            rank_melted = (
                rank_df
                .reset_index()
                .melt(id_vars="date", var_name="symbol", value_name="rank")
            )
            melted = melted.merge(rank_melted, on=["date", "symbol"], how="left")

        cols = ["date", "symbol", "direction", "strength", "strategy_name", "source"]
        if "rank" in melted.columns:
            cols.append("rank")
        return melted[cols].sort_values(["date", "symbol"]).reset_index(drop=True)

    def latest(self, date: pd.Timestamp | None = None) -> list[Signal]:
        """Return Signal objects for the most recent (or specified) date.

        This is the bridge to the execution-layer Signal dispatch path.
        Caller must inspect signal.source == RESEARCH and re-derive
        execution signals independently.

        Args:
            date: Specific date to extract. Defaults to last non-NaN row.

        Returns:
            list[Signal], one per symbol with non-NaN strength.
        """
        if date is None:
            idx = self.signals.last_valid_index()
            if idx is None:
                return []
        else:
            idx = date

        row_signals = self.signals.loc[idx]
        row_strength = self.strength.loc[idx]

        result: list[Signal] = []
        for symbol in self.symbols:
            s = row_strength.get(symbol)
            if pd.isna(s):
                continue
            direction = SignalDirection.LONG if float(s) > 0.0 else SignalDirection.FLAT
            result.append(
                Signal(
                    symbol=symbol,
                    timestamp=pd.Timestamp(idx).to_pydatetime(),
                    direction=direction,
                    strength=float(s),
                    strategy_name=self.strategy_name,
                    source=self.source,
                )
            )
        return result

    def __repr__(self) -> str:
        return (
            f"SignalFrame(strategy='{self.strategy_name}', "
            f"symbols={self.symbols}, dates={self.n_dates}, "
            f"source={self.source.value})"
        )
