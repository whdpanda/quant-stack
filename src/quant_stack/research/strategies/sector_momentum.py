"""Sector / Style ETF Momentum Strategy.

Logic (Quantpedia sector-rotation template)
-------------------------------------------
Universe  : N sector or style ETFs (caller-supplied via close columns)
Momentum  : price ROC over *momentum_window* trading days
            Default 252 ≈ 12 calendar months
Ranking   : cross-sectional; top *top_n* ETFs receive signal = 1.0
Holding   : equal-weight (strength uniform across longs)
Rebalance : monthly — enforced by the backtest adapter, NOT here
            (set VbtRunConfig.rebalance_freq = "ME" in the caller)

Usage with vbt_adapter (recommended — includes look-ahead prevention
and monthly rebalancing):

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
                   config=VbtRunConfig(rebalance_freq="ME"),
                   benchmark_close=close["SPY"],
                   strategy_name=strategy.name,
               )

Usage with run_backtest (daily rebalancing — higher turnover):

    from quant_stack.research.backtest import run_backtest
    result = run_backtest(strategy, close, BacktestConfig(...))
"""

from __future__ import annotations

import pandas as pd

from quant_stack.factors.momentum import momentum
from quant_stack.research.base import Strategy
from quant_stack.signals.momentum import relative_momentum_ranking_signal


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
