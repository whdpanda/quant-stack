"""OHLCV data validation and cleaning pipeline.

Pipeline (applied in order by DataValidator.validate):
    1. check_required_columns   — fail fast if columns are missing
    2. coerce_types             — date → datetime64, OHLCV → float64
    3. add_adj_close            — set adj_close = close if column absent
    4. sort                     — (symbol, date) ascending
    5. handle_duplicates        — drop exact dups; raise on key conflicts
    6. fill_missing             — forward-fill gaps up to fill_limit bars
    7. check_ohlcv_consistency  — high >= low etc. (warn, don't raise)

Each step is a standalone static method so it can be tested independently.

Survivorship-bias note
----------------------
The validator does NOT drop symbols or rows. Filtering (e.g. removing
delisted tickers) must be an explicit caller decision, not a silent default.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from quant_stack.data.loaders.base import CANONICAL_COLUMNS, NUMERIC_COLUMNS, REQUIRED_COLUMNS


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class ValidationConfig:
    fill_method: Literal["ffill", "bfill", "none"] = "ffill"
    fill_limit: int = 5
    drop_exact_duplicates: bool = True
    check_ohlcv: bool = True
    ohlcv_violation_action: Literal["warn", "raise"] = "warn"
    # Symbols with a missing-value ratio above this threshold emit a warning
    missing_ratio_warn_threshold: float = 0.05


# ── Exceptions ─────────────────────────────────────────────────────────────────

class DataValidationError(ValueError):
    """Raised when the data cannot be used safely."""


# ── Validator ──────────────────────────────────────────────────────────────────

class DataValidator:
    """Apply the full validation and cleaning pipeline to a raw OHLCV DataFrame."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all validation steps and return a cleaned DataFrame.

        Returns a DataFrame with exactly CANONICAL_COLUMNS in that order.
        The index is reset to a RangeIndex.
        """
        df = df.copy()
        df = self.check_required_columns(df)
        df = self.coerce_types(df)
        df = self.add_adj_close(df)
        df = self.sort(df)
        df = self.handle_duplicates(df)
        df = self.fill_missing(df, self.config.fill_method, self.config.fill_limit)
        if self.config.check_ohlcv:
            df = self.check_ohlcv_consistency(df, self.config.ohlcv_violation_action)
        return df[CANONICAL_COLUMNS].reset_index(drop=True)

    # ── Step 1 ────────────────────────────────────────────────────────────────

    @staticmethod
    def check_required_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Raise DataValidationError if any required column is absent."""
        present = frozenset(df.columns)
        missing = REQUIRED_COLUMNS - present
        if missing:
            raise DataValidationError(
                f"Required columns missing: {sorted(missing)}. "
                f"Present: {sorted(present)}"
            )
        return df

    # ── Step 2 ────────────────────────────────────────────────────────────────

    @staticmethod
    def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to canonical dtypes.

        - date  → datetime64[ns]
        - symbol → str (object)
        - OHLCV → float64 (volume included; avoids int NaN issues)
        """
        df = df.copy()

        # date
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # symbol
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

        # numeric columns (only those present)
        num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ── Step 3 ────────────────────────────────────────────────────────────────

    @staticmethod
    def add_adj_close(df: pd.DataFrame) -> pd.DataFrame:
        """Add adj_close column if absent, defaulting to close."""
        df = df.copy()
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        return df

    # ── Step 4 ────────────────────────────────────────────────────────────────

    @staticmethod
    def sort(df: pd.DataFrame) -> pd.DataFrame:
        """Sort by (symbol, date) ascending."""
        return df.sort_values(["symbol", "date"], ascending=True, ignore_index=True)

    # ── Step 5 ────────────────────────────────────────────────────────────────

    @staticmethod
    def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows; raise on (date, symbol) key conflicts.

        A key conflict means two rows share the same (date, symbol) but have
        different prices — this indicates a data quality problem that cannot
        be resolved silently.
        """
        # Drop exact duplicates first
        df = df.drop_duplicates().reset_index(drop=True)

        # Check for key conflicts
        key_dupes = df.duplicated(subset=["date", "symbol"], keep=False)
        if key_dupes.any():
            conflicting = (
                df.loc[key_dupes, ["date", "symbol"]]
                .drop_duplicates()
                .head(5)
                .to_dict("records")
            )
            raise DataValidationError(
                f"Conflicting rows found for the same (date, symbol) key. "
                f"First conflicts: {conflicting}. "
                "Resolve conflicts in the source file before loading."
            )
        return df

    # ── Step 6 ────────────────────────────────────────────────────────────────

    @staticmethod
    def fill_missing(
        df: pd.DataFrame,
        method: Literal["ffill", "bfill", "none"] = "ffill",
        limit: int = 5,
    ) -> pd.DataFrame:
        """Fill missing OHLCV values per symbol.

        Forward-fill is applied per-symbol so a gap in SPY does not
        contaminate QQQ data. Only numeric OHLCV columns are filled;
        date and symbol are never interpolated.

        A warning is emitted for symbols where missing-value ratio
        exceeds 5% after filling, as this may indicate a structural
        data problem (e.g. survivorship bias from incomplete history).
        """
        if method == "none":
            return df

        num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
        filled_frames: list[pd.DataFrame] = []

        for symbol, group in df.groupby("symbol", sort=False):
            before_nulls = group[num_cols].isnull().sum().sum()
            if before_nulls > 0:
                filled = group.copy()
                filled[num_cols] = (
                    filled[num_cols]
                    .ffill(limit=limit)
                    if method == "ffill"
                    else filled[num_cols].bfill(limit=limit)
                )
                after_nulls = filled[num_cols].isnull().sum().sum()
                remaining_ratio = after_nulls / max(len(filled) * len(num_cols), 1)
                if remaining_ratio > 0.05:
                    warnings.warn(
                        f"Symbol '{symbol}': {after_nulls} NaN values remain after "
                        f"fill (ratio={remaining_ratio:.1%}). Check source data for gaps.",
                        stacklevel=4,
                    )
                filled_frames.append(filled)
            else:
                filled_frames.append(group)

        return pd.concat(filled_frames, ignore_index=True)

    # ── Step 7 ────────────────────────────────────────────────────────────────

    @staticmethod
    def check_ohlcv_consistency(
        df: pd.DataFrame,
        action: Literal["warn", "raise"] = "warn",
    ) -> pd.DataFrame:
        """Check OHLCV price relationships (high >= low, etc.).

        Violations indicate bad source data (e.g. split not applied,
        corrupted CSV). The action parameter controls whether to warn
        or raise; in production use "raise" once data quality is trusted.

        Look-ahead note: this check is purely retrospective — it does not
        use future prices to validate past prices.
        """
        violations: list[str] = []

        checks = [
            (df["high"] < df["low"],  "high < low"),
            (df["high"] < df["open"], "high < open"),
            (df["high"] < df["close"], "high < close"),
            (df["low"] > df["open"],  "low > open"),
            (df["low"] > df["close"], "low > close"),
            (df["volume"] < 0,        "volume < 0"),
            (df["open"] <= 0,         "open <= 0"),
            (df["close"] <= 0,        "close <= 0"),
        ]

        for mask, desc in checks:
            n = mask.sum()
            if n > 0:
                sample = df.loc[mask, ["date", "symbol"]].head(3).to_dict("records")
                violations.append(f"  {desc}: {n} rows — e.g. {sample}")

        if violations:
            msg = "OHLCV consistency violations:\n" + "\n".join(violations)
            if action == "raise":
                raise DataValidationError(msg)
            warnings.warn(msg, stacklevel=4)

        return df
