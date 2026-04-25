"""Sector / Style ETF Momentum Strategy.

Logic (Quantpedia sector-rotation template)
-------------------------------------------
Universe  : N sector or style ETFs (caller-supplied via close columns)
Momentum  : price ROC over *momentum_window* trading days
            Default 210 ≈ 10 calendar months
Ranking   : cross-sectional; top *top_n* ETFs receive signal = 1.0
Holding   : blend_70_30 (70% equal + 30% inverse-vol) by default
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
    BLEND_50_50    = "blend_50_50"   # 50% equal + 50% inverse_vol
    BLEND_70_30    = "blend_70_30"   # 70% equal + 30% inverse_vol


class HysteresisMode(StrEnum):
    NONE             = "no_hysteresis"
    EXIT_BUFFER_TOP4 = "exit_buffer_top4"
    EXIT_BUFFER_TOP5 = "exit_buffer_top5"
    ENTRY_MARGIN     = "entry_margin"

# ── Canonical risk-on universe ────────────────────────────────────────────────
# Single source of truth for all sector momentum experiments.
# IEF is deliberately excluded: it is a defensive fallback asset, not a
# risk-on ranking candidate.
#
# Universe history:
#   2026-04-24: GDX replaced SPY (18-candidate study; GDX superior on all 3 metrics)
#   2026-04-25: IYT replaced VNQ (10-candidate study; IYT: dSharpe=+0.063, dCAGR=+1.01%,
#               dMaxDD=-1.41% — robust win on all three criteria)
#
# IMPORTANT: asset order is significant.  relative_momentum_ranking_signal uses
# method="first" for tie-breaking, so column order determines outcomes in ties.
# IYT occupies index 0 (formerly VNQ's slot) — preserve this order in all studies.
RISK_ON_UNIVERSE: list[str] = [
    "IYT",  # Transportation (iShares) — consolidated 2026-04-25, replaces VNQ
    "QQQ",  # Technology / Nasdaq
    "XLE",  # Energy
    "XLV",  # Health Care
    "XLF",  # Financials
    "XLI",  # Industrials
    "VTV",  # Vanguard Value ETF
    "GDX",  # VanEck Gold Miners — consolidated 2026-04-24, replaces SPY
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

    def generate_signals_full(
        self, close: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (signals, ranks, momentum_scores) for advanced overlays.

        signals          : 0/1/NaN daily DataFrame
        ranks            : cross-sectional rank each day (1 = best); NaN during warmup
        momentum_scores  : raw price ROC values; NaN during warmup
        """
        mom = momentum(close, self.momentum_window)
        sf = relative_momentum_ranking_signal(
            mom, top_n=self.top_n, strategy_name=self.name
        )
        # ranks: ascending rank (1 = highest momentum)
        ranks = mom.rank(axis=1, ascending=False, method="min")
        ranks[mom.isna().all(axis=1)] = float("nan")
        return sf.signals, ranks, mom

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


# ── Hysteresis / turnover-buffer overlay ─────────────────────────────────────

def apply_hysteresis(
    signals: pd.DataFrame,
    ranks: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    mode: HysteresisMode,
    top_n: int = 3,
    exit_buffer: int = 4,
    entry_margin: float = 0.02,
) -> pd.DataFrame:
    """Apply a stateful hysteresis rule on top of raw signals.

    Iterates day by day maintaining a *held* set.  NaN (warmup) rows pass
    through unchanged.  On non-warmup rows the rule is applied and a new
    0/1 signal DataFrame is returned.

    Modes
    -----
    NONE          : return signals unchanged (baseline)
    EXIT_BUFFER_TOP4/5 :
        A held asset is only evicted when its rank > exit_buffer.
        Any vacancy (held < top_n) is filled by the highest-ranked
        asset not already held, as long as its rank <= top_n.
    ENTRY_MARGIN  :
        A new asset (rank <= top_n, not held) displaces the lowest-ranked
        held asset only if its momentum ROC exceeds that asset's ROC by
        at least *entry_margin* (absolute, e.g. 0.02 = 2 pp).
    """
    if mode == HysteresisMode.NONE:
        return signals.copy()

    cols = signals.columns
    n_rows = len(signals)
    out = signals.copy().astype(float)

    held: set[str] = set()

    for i in range(n_rows):
        row_sig = signals.iloc[i]

        # Warmup rows: all-NaN → pass through, held stays empty
        if row_sig.isna().all():
            held = set()
            continue

        row_rank = ranks.iloc[i]
        row_mom  = momentum_scores.iloc[i]

        if mode in (HysteresisMode.EXIT_BUFFER_TOP4, HysteresisMode.EXIT_BUFFER_TOP5):
            _apply_exit_buffer(held, row_rank, top_n, exit_buffer)
        elif mode == HysteresisMode.ENTRY_MARGIN:
            _apply_entry_margin(held, row_rank, row_mom, top_n, entry_margin)

        # Write new signals for this row
        new_row = pd.Series(0.0, index=cols)
        for sym in held:
            new_row[sym] = 1.0
        out.iloc[i] = new_row

    return out


def _apply_exit_buffer(
    held: set[str],
    row_rank: pd.Series,
    top_n: int,
    exit_buffer: int,
) -> None:
    """Mutate *held* in-place: evict only when rank > exit_buffer, fill vacancies."""
    # Evict held assets whose rank has fallen beyond the buffer
    to_evict = {sym for sym in held if row_rank.get(sym, float("inf")) > exit_buffer}
    held -= to_evict

    # Fill vacancies with best-ranked assets not already held (rank <= top_n)
    candidates = sorted(
        [sym for sym in row_rank.index if sym not in held and row_rank[sym] <= top_n],
        key=lambda s: row_rank[s],
    )
    while len(held) < top_n and candidates:
        held.add(candidates.pop(0))


def _apply_entry_margin(
    held: set[str],
    row_rank: pd.Series,
    row_mom: pd.Series,
    top_n: int,
    entry_margin: float,
) -> None:
    """Mutate *held* in-place: replace with margin rule, evict hard drops."""
    # Hard eviction: held asset is no longer in the raw top_n signal universe
    # (rank > top_n * 2 acts as a hard floor to avoid holding perpetual losers)
    hard_floor = top_n * 2
    to_evict = {sym for sym in held if row_rank.get(sym, float("inf")) > hard_floor}
    held -= to_evict

    # Identify new entrants: top_n ranked, not currently held
    entrants = sorted(
        [sym for sym in row_rank.index if sym not in held and row_rank[sym] <= top_n],
        key=lambda s: row_rank[s],
    )

    for entrant in entrants:
        if len(held) < top_n:
            held.add(entrant)
            continue
        # Find lowest-momentum held asset
        worst_held = min(held, key=lambda s: row_mom.get(s, float("-inf")))
        mom_entrant = row_mom.get(entrant, float("-inf"))
        mom_worst   = row_mom.get(worst_held, float("-inf"))
        if mom_entrant - mom_worst > entry_margin:
            held.discard(worst_held)
            held.add(entrant)


# ── Blended weighting helpers ─────────────────────────────────────────────────

def _blend_strength(
    signals: pd.DataFrame,
    inv_vol_strength: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Return pre-normalised blended weights (row sums ≈ 1 when invested).

    alpha  : fraction allocated to equal weight (0 < alpha < 1).
    1-alpha: fraction allocated to inverse-vol weight.

    Pre-normalising here means signal_frame_to_weights divides by ~1.0,
    leaving the blended weights effectively unchanged.

    Robustness:
    - Zero / near-zero volatility: clipped upstream to 1e-8 before this call.
    - No selected assets in a row: row sum = 0 → NaN → propagated as 0.0
      after the final .where mask; signal_frame_to_weights later treats that
      row as explicit cash (all-zero post-warmup).
    - Warmup rows (signals all-NaN): masked to 0.0; signal_frame_to_weights
      restores them to NaN via warmup_mask.
    """
    # Equal weights: 1/n per selected asset
    n_sel = (signals == 1.0).sum(axis=1).replace(0, float("nan"))
    eq_w  = signals.where(signals == 1.0, other=0.0).div(n_sel, axis=0)

    # Inverse-vol weights: row-normalise the raw inv-vol strengths
    ivol_sum = inv_vol_strength.sum(axis=1).replace(0.0, float("nan"))
    ivol_w   = inv_vol_strength.div(ivol_sum, axis=0)

    blended = alpha * eq_w + (1.0 - alpha) * ivol_w
    return blended.where(signals == 1.0, other=0.0)


def compute_strength(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    scheme: WeightingScheme,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Compute per-asset strength from *pre-filtered* signals.

    Unlike SectorMomentumStrategy.compute_weights(), this function accepts
    signals that have already been processed (e.g. entry_margin hysteresis)
    and only determines how to weight the selected assets — it does NOT
    recompute signals from scratch.

    Args:
        signals   : 0/1/NaN DataFrame (e.g. output of apply_hysteresis).
        close     : daily adjusted-close prices, same columns as signals.
        scheme    : weighting method (WeightingScheme enum).
        vol_window: rolling window for volatility; default 63 trading days
                    (≈ 3 calendar months), consistent with compute_weights().

    Returns:
        Strength DataFrame to pass into SignalFrame → signal_frame_to_weights().
        - EQUAL / INVERSE_VOL: raw unnormalised strengths (normalised later).
        - BLEND_*: pre-normalised row-wise (signal_frame_to_weights divides
          by ~1.0, leaving weights unchanged).

    Volatility robustness:
        Rolling std is clipped at 1e-8 to avoid division-by-zero.  vol_window
        warmup (63 bars) is always covered by the strategy's own momentum
        warmup (210 bars), so no NaN leaks into post-warmup strength values.
    """
    if scheme == WeightingScheme.EQUAL:
        return signals.copy()

    if scheme == WeightingScheme.INVERSE_VOL:
        vol = close.pct_change().rolling(vol_window).std()
        return (1.0 / vol.clip(lower=1e-8)).where(signals == 1.0, other=0.0)

    if scheme == WeightingScheme.MOMENTUM_SCORE:
        raise ValueError(
            "MOMENTUM_SCORE requires raw momentum scores not available here; "
            "use SectorMomentumStrategy.compute_weights(close, scheme) instead."
        )

    if scheme in (WeightingScheme.BLEND_50_50, WeightingScheme.BLEND_70_30):
        alpha = 0.5 if scheme == WeightingScheme.BLEND_50_50 else 0.7
        vol = close.pct_change().rolling(vol_window).std()
        raw_inv_vol = (1.0 / vol.clip(lower=1e-8)).where(signals == 1.0, other=0.0)
        return _blend_strength(signals, raw_inv_vol, alpha)

    raise ValueError(f"Unknown WeightingScheme: {scheme!r}")
