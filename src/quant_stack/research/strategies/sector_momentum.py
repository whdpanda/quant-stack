"""Sector / Style ETF Momentum Strategy.

Logic (Quantpedia sector-rotation template)
-------------------------------------------
Universe  : N sector or style ETFs (caller-supplied via close columns)
Momentum  : price ROC over *momentum_window* trading days
            Default 252 ≈ 12 calendar months
Ranking   : cross-sectional; top *top_n* ETFs receive signal = 1.0
Holding   : equal-weight (strength uniform across longs)
Rebalance : bi-monthly (every 2 months) — enforced by the backtest adapter, NOT here
            (set VbtRunConfig.rebalance_freq = "2ME" in the caller)

Usage with vbt_adapter (recommended — includes look-ahead prevention
and bi-monthly rebalancing):

    from quant_stack.research.strategies.sector_momentum import SectorMomentumStrategy
    from quant_stack.research.vbt_adapter import (
        VbtRunConfig, run_vbt_backtest, signal_frame_to_weights,
    )
    from quant_stack.signals.base import SignalFrame

    strategy = SectorMomentumStrategy()
    signals  = strategy.generate_signals(close)     # daily 0/1 DataFrame
    sf       = SignalFrame(signals=signals, strength=signals.copy(),
                           strategy_name=strategy.name)
    weights  = signal_frame_to_weights(sf)
    result   = run_vbt_backtest(
                   close, weights,
                   config=VbtRunConfig(rebalance_freq="2ME"),
                   benchmark_close=close["SPY"],
                   strategy_name=strategy.name,
               )

Usage with run_backtest (daily rebalancing — higher turnover):

    from quant_stack.research.backtest import run_backtest
    result = run_backtest(strategy, close, BacktestConfig(...))
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

from quant_stack.factors.momentum import momentum
from quant_stack.research.base import Strategy
from quant_stack.signals.momentum import relative_momentum_ranking_signal


class WeightingScheme(StrEnum):
    EQUAL          = "equal"
    INVERSE_VOL    = "inverse_vol"
    MOMENTUM_SCORE = "momentum_score"

# ── Canonical risk-on universe ────────────────────────────────────────────────
# Single source of truth for all sector momentum experiments.
# IEF is deliberately excluded: it is a defensive fallback asset, not a
# risk-on ranking candidate.
RISK_ON_UNIVERSE: list[str] = [
    "VNQ",  # Real Estate
    "QQQ",  # Technology / Nasdaq
    "XLE",  # Energy
    "XLV",  # Health Care
    "XLF",  # Financials
    "XLI",  # Industrials
    "VTV",  # Vanguard Value ETF
    "SPY",  # Broad Market
    "XLP",  # Consumer Staples
]


class SectorMomentumStrategy(Strategy):
    """Rank ETFs by rolling price momentum; hold the top N equal-weight.

    Args:
        momentum_window: Lookback in trading days (default 252 ≈ 12 months).
        top_n: Number of ETFs to hold simultaneously (default 3).
    """

    name = "sector_momentum_12m"

    def __init__(self, momentum_window: int = 252, top_n: int = 3) -> None:
        super().__init__(momentum_window=momentum_window, top_n=top_n)
        if momentum_window <= 0:
            raise ValueError(f"momentum_window must be > 0, got {momentum_window}")
        if top_n <= 0:
            raise ValueError(f"top_n must be > 0, got {top_n}")
        self.momentum_window = momentum_window
        self.top_n = top_n
        self.name = f"sector_momentum_{momentum_window}d_top{top_n}"

    def compute_weights(
        self,
        close: pd.DataFrame,
        scheme: WeightingScheme = WeightingScheme.EQUAL,
        vol_window: int = 63,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (signals, strength) under the requested weighting scheme.

        signals  : 0/1/NaN daily DataFrame (identical to generate_signals output)
        strength : scheme-specific raw values; pass to signal_frame_to_weights()
                   for normalisation.

        Schemes
        -------
        equal          : strength = 1.0 for every selected asset → equal weight
        inverse_vol    : strength = 1 / rolling_std(returns, vol_window);
                         near-zero vol is clipped at 1e-8 to avoid division errors
        momentum_score : strength = raw ROC shifted to non-negative via row-wise
                         min-shift + ε.  Rationale: ROC can be negative in bear
                         markets; shifting preserves relative ordering while
                         ensuring all weights are positive after normalisation.
        """
        mom = momentum(close, self.momentum_window)
        sf_base = relative_momentum_ranking_signal(
            mom, top_n=self.top_n, strategy_name=self.name
        )
        signals = sf_base.signals  # 0/1/NaN

        if scheme == WeightingScheme.EQUAL:
            strength = signals.copy()

        elif scheme == WeightingScheme.INVERSE_VOL:
            vol = close.pct_change().rolling(vol_window).std()
            strength = (1.0 / vol.clip(lower=1e-8)).where(signals == 1.0, other=0.0)

        elif scheme == WeightingScheme.MOMENTUM_SCORE:
            raw = mom.where(signals == 1.0)           # NaN for non-selected
            row_min = raw.min(axis=1)
            shifted = raw.sub(row_min, axis=0) + 1e-6  # all selected ≥ ε > 0
            strength = shifted.where(signals == 1.0, other=0.0)

        else:
            raise ValueError(f"Unknown weighting scheme: {scheme!r}")

        return signals, strength

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-sectional momentum signals for every trading day.

        Returns a DataFrame of the same shape as *close*:
          - 1.0  : ETF is in the top-N by momentum (go long)
          - 0.0  : ETF is ranked outside top-N (stay flat)
          - NaN  : insufficient history (first *momentum_window* rows)

        Note: signals are recomputed daily. Monthly rebalancing is the
        responsibility of the calling adapter (VbtRunConfig.rebalance_freq).
        """
        mom = momentum(close, self.momentum_window)
        sf = relative_momentum_ranking_signal(
            mom,
            top_n=self.top_n,
            strategy_name=self.name,
        )
        return sf.signals
