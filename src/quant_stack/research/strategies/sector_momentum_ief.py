"""Sector Momentum Strategy with IEF Defensive Fallback.

Logic
-----
1. Compute momentum for every asset in the risk-on universe
   (all columns in *close* except the fallback asset).
2. Apply absolute momentum filter: only ETFs with momentum > threshold
   are eligible for selection.
3. Rank eligible ETFs cross-sectionally; select the top N.
4. Each selected ETF receives weight = 1 / top_n.
5. Remaining slots go to the fallback asset (default IEF):
       fallback_weight = (top_n - n_selected) / top_n
6. Total weight always sums to 1.0.

Cash vs. IEF fallback
---------------------
Unlike the plain cash-fallback version, this class puts unused capacity
into IEF rather than leaving it uninvested.  IEF acts as a defensive
position during risk-off periods — it earns the duration premium rather
than sitting in zero-yield cash.

Warmup period
-------------
The first *momentum_window* rows have no valid momentum.  Those rows
are returned as NaN, which the vectorbt adapter translates to
"hold initial cash" (no orders placed).

Usage
-----
    from quant_stack.research.strategies.sector_momentum_ief import (
        SectorMomentumIefFallbackStrategy,
    )
    from quant_stack.research.vbt_adapter import VbtRunConfig, run_vbt_backtest

    strategy = SectorMomentumIefFallbackStrategy(
        momentum_window=210,
        top_n=3,
        abs_momentum_threshold=0.0,
        fallback_asset="IEF",
    )
    # close must contain both risk-on columns AND the fallback_asset column
    weights = strategy.compute_weights(close)
    result  = run_vbt_backtest(
                  close, weights,
                  config=VbtRunConfig(rebalance_freq="ME"),
                  benchmark_close=close["SPY"],
                  strategy_name=strategy.name,
              )
"""

from __future__ import annotations

import pandas as pd

from quant_stack.factors.momentum import momentum


class SectorMomentumIefFallbackStrategy:
    """Sector momentum with a configurable defensive fallback asset.

    Args:
        momentum_window: Lookback in trading days (default 210 ≈ 10 months).
        top_n: Maximum number of risk-on ETFs to hold (default 3).
        abs_momentum_threshold: Risk-on ETFs must have momentum strictly
            greater than this value to be eligible.  Default 0.0 (positive
            momentum required).
        fallback_asset: Column name of the defensive fallback asset.
            Must be present in the *close* DataFrame passed to
            ``compute_weights``.  Default "IEF".
    """

    def __init__(
        self,
        momentum_window: int = 210,
        top_n: int = 3,
        abs_momentum_threshold: float = 0.0,
        fallback_asset: str = "IEF",
    ) -> None:
        if momentum_window <= 0:
            raise ValueError(f"momentum_window must be > 0, got {momentum_window}")
        if top_n <= 0:
            raise ValueError(f"top_n must be > 0, got {top_n}")

        self.momentum_window = momentum_window
        self.top_n = top_n
        self.abs_momentum_threshold = abs_momentum_threshold
        self.fallback_asset = fallback_asset

        threshold_tag = str(abs_momentum_threshold).replace("-", "neg")
        self.name = (
            f"sector_momentum_{momentum_window}d_top{top_n}"
            f"_{fallback_asset.lower()}_fallback_{threshold_tag}"
        )

    # ------------------------------------------------------------------

    def compute_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        """Compute daily target-weight DataFrame for use with run_vbt_backtest.

        Args:
            close: Adjusted close prices.  DatetimeIndex × symbol columns.
                   Must include the fallback_asset column AND at least one
                   risk-on asset column (any other column).

        Returns:
            DataFrame with the same index as *close* and columns =
            [risk_on_cols..., fallback_asset].  Values are target weights
            that sum to exactly 1.0 per row.  NaN rows = warmup period
            (momentum_window bars from the start).

        Weight formula
        --------------
        For a given row:
          - n_selected  = number of risk-on ETFs that pass the filter AND
                          rank in the top N
          - Each selected ETF weight = 1 / top_n
          - Fallback weight           = (top_n - n_selected) / top_n
          - Sum = n_selected/top_n + (top_n - n_selected)/top_n = 1.0
        """
        if self.fallback_asset not in close.columns:
            raise ValueError(
                f"fallback_asset '{self.fallback_asset}' not found in close columns "
                f"{list(close.columns)}"
            )

        risk_on_cols = [c for c in close.columns if c != self.fallback_asset]
        if not risk_on_cols:
            raise ValueError("close must contain at least one risk-on asset column")

        close_ro = close[risk_on_cols]

        # ── Momentum for risk-on assets ───────────────────────────────
        mom = momentum(close_ro, self.momentum_window)

        # ── Absolute momentum eligibility ─────────────────────────────
        # NaN comparisons yield False → warmup rows treated as fully
        # ineligible, which is fine because we restore them to NaN below.
        eligible = mom > self.abs_momentum_threshold   # bool DataFrame

        # ── Cross-sectional ranking within eligible assets ────────────
        masked_mom = mom.where(eligible)               # NaN for ineligible
        ranks = masked_mom.rank(
            axis=1, ascending=False, method="first", na_option="keep"
        )

        # ── Binary selection: top N ───────────────────────────────────
        selected = (ranks <= self.top_n).fillna(False)  # bool, NaN rank → False

        # ── Build weight DataFrame ────────────────────────────────────
        n_selected = selected.sum(axis=1)             # 0 … top_n, integer per row

        # Each selected risk-on ETF gets 1/top_n
        risk_on_w = selected.astype(float).div(self.top_n)

        # Fallback asset fills the unused slots
        fallback_w = (self.top_n - n_selected) / self.top_n

        weights = risk_on_w.copy()
        weights[self.fallback_asset] = fallback_w

        # ── Restore NaN for warmup rows ───────────────────────────────
        # Warmup = rows where momentum is entirely NaN (factor not yet ready).
        warmup = mom.isna().all(axis=1)
        weights.loc[warmup] = float("nan")

        return weights[risk_on_cols + [self.fallback_asset]]

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"window={self.momentum_window}, top_n={self.top_n}, "
            f"threshold={self.abs_momentum_threshold}, "
            f"fallback={self.fallback_asset!r})"
        )
