"""Research and strategy tools.

All tools are read-only with respect to disk state:
  - load_market_data_tool  : downloads prices → ctx.close_df
  - build_factors_tool     : computes ROC momentum → ctx.momentum_df
  - generate_signals_tool  : applies top-N ranking → ctx.signals_df / ctx.selected_symbols
  - allocate_portfolio_tool: computes blended weights → ctx.target_weights
  - run_research_backtest_tool: full vbt backtest → ExperimentRecord on disk
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

from quant_stack.agent.tools._context import ToolContext


# ── Tool implementations ──────────────────────────────────────────────────────

def load_market_data(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Download adjusted close prices and store in context."""
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as e:
        return {"status": "error", "error": f"Missing dependency: {e}. pip install yfinance"}

    symbols: list[str] = inputs.get("symbols", [])
    if not symbols:
        return {"status": "error", "error": "'symbols' is required"}

    lookback_days = int(inputs.get("lookback_days", 350))
    end = date.today()
    start = end - timedelta(days=lookback_days + 120)

    try:
        raw = yf.download(
            symbols, start=str(start), end=str(end),
            auto_adjust=True, progress=False,
        )
    except Exception as e:
        return {"status": "error", "error": f"yfinance download failed: {e}"}

    if raw is None or raw.empty:
        return {"status": "error", "error": "yfinance returned empty data. Check connection."}

    import pandas as pd
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])

    close = close.dropna(how="all")
    available = [s for s in symbols if s in close.columns]
    missing = [s for s in symbols if s not in close.columns]

    ctx.close_df = close[available]

    return {
        "status": "ok",
        "rows": len(close),
        "columns": available,
        "date_range": [str(close.index[0].date()), str(close.index[-1].date())],
        "missing_symbols": missing,
    }


def build_factors(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Compute rolling-price-ROC momentum for every symbol in context.

    Stores ctx.momentum_df (continuous ROC values) for use by generate_signals_tool.
    Does NOT apply top-N ranking — that is generate_signals_tool's job.
    """
    if ctx.close_df is None:
        return {"status": "error", "error": "No price data in context. Call load_market_data_tool first."}

    momentum_window = int(inputs.get("momentum_window", 210))

    try:
        from quant_stack.factors.momentum import momentum
        mom = momentum(ctx.close_df, window=momentum_window)
    except Exception as e:
        return {"status": "error", "error": f"Momentum computation failed: {e}"}

    valid_rows = mom.dropna(how="all")
    if valid_rows.empty:
        return {
            "status": "error",
            "error": (
                f"Insufficient history for {momentum_window}-day momentum. "
                f"Have {len(ctx.close_df)} rows, need at least {momentum_window}."
            ),
        }

    ctx.momentum_df = mom

    last_row = valid_rows.iloc[-1]
    signal_date = valid_rows.index[-1]
    ctx.signal_date = str(signal_date.date() if hasattr(signal_date, "date") else signal_date)

    scores = {
        sym: round(float(last_row[sym]), 4)
        for sym in last_row.index
        if not math.isnan(float(last_row[sym]))
    }

    return {
        "status": "ok",
        "signal_date": ctx.signal_date,
        "momentum_window": momentum_window,
        "momentum_scores": dict(sorted(scores.items(), key=lambda x: -x[1])),
    }


def generate_signals(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Apply cross-sectional top-N ranking to momentum scores.

    Reads ctx.momentum_df; writes ctx.signals_df and ctx.selected_symbols.
    """
    if ctx.momentum_df is None:
        return {"status": "error", "error": "No momentum data. Call build_factors_tool first."}

    top_n = int(inputs.get("top_n", 3))

    try:
        from quant_stack.signals.momentum import relative_momentum_ranking_signal
        sf = relative_momentum_ranking_signal(
            ctx.momentum_df,
            top_n=top_n,
            strategy_name=f"sector_momentum_tool_top{top_n}",
        )
    except Exception as e:
        return {"status": "error", "error": f"Signal generation failed: {e}"}

    ctx.signals_df = sf.signals

    last_valid = sf.signals.dropna(how="all")
    if last_valid.empty:
        return {"status": "error", "error": "No valid signal rows after warmup."}

    last_row = last_valid.iloc[-1]
    selected = sorted([sym for sym in last_row.index if last_row[sym] == 1.0])
    ctx.selected_symbols = selected

    excluded = sorted([sym for sym in last_row.index if last_row[sym] == 0.0])

    return {
        "status": "ok",
        "signal_date": ctx.signal_date,
        "top_n": top_n,
        "selected": selected,
        "excluded": excluded,
    }


def allocate_portfolio(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Compute blended target weights from signals.

    Reads ctx.signals_df + ctx.close_df; writes ctx.target_weights.
    """
    if ctx.signals_df is None:
        return {"status": "error", "error": "No signals. Call generate_signals_tool first."}
    if ctx.close_df is None:
        return {"status": "error", "error": "No price data in context."}

    weighting_str = inputs.get("weighting", "BLEND_70_30").upper()
    vol_window = int(inputs.get("vol_window", 63))

    try:
        from quant_stack.research.strategies.sector_momentum import (
            WeightingScheme,
            compute_strength,
        )
        from quant_stack.research.vbt_adapter import signal_frame_to_weights
        from quant_stack.signals.base import SignalFrame

        scheme_map = {
            "BLEND_70_30": WeightingScheme.BLEND_70_30,
            "BLEND_50_50": WeightingScheme.BLEND_50_50,
            "EQUAL": WeightingScheme.EQUAL,
            "INVERSE_VOL": WeightingScheme.INVERSE_VOL,
        }
        scheme = scheme_map.get(weighting_str)
        if scheme is None:
            return {"status": "error", "error": f"Unknown weighting: {weighting_str}. Choose from {list(scheme_map)}"}

        strength = compute_strength(ctx.signals_df, ctx.close_df, scheme, vol_window=vol_window)
        sf = SignalFrame(
            signals=ctx.signals_df,
            strength=strength,
            strategy_name=f"sector_momentum_tool_{weighting_str.lower()}",
        )
        weights_df = signal_frame_to_weights(sf)

    except Exception as e:
        return {"status": "error", "error": f"Weight computation failed: {e}"}

    valid_rows = weights_df.dropna(how="all")
    if valid_rows.empty:
        return {"status": "error", "error": "No valid weight rows produced."}

    latest_row = valid_rows.iloc[-1]
    import pandas as pd
    weights = {
        sym: round(float(latest_row[sym]), 4)
        for sym in latest_row.index
        if not pd.isna(latest_row[sym]) and float(latest_row[sym]) > 0.001
    }

    ctx.target_weights = weights

    return {
        "status": "ok",
        "signal_date": ctx.signal_date,
        "weighting": weighting_str,
        "weights": dict(sorted(weights.items(), key=lambda x: -x[1])),
        "sum_check": round(sum(weights.values()), 4),
    }


def run_research_backtest(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Run a full vbt backtest with the sector momentum strategy and save the record.

    Requires vectorbt. Uses ctx.close_df if available; otherwise fails with an
    informative message (call load_market_data_tool first).
    """
    if ctx.close_df is None:
        return {"status": "error", "error": "No price data. Call load_market_data_tool first."}

    momentum_window = int(inputs.get("momentum_window", 210))
    top_n = int(inputs.get("top_n", 3))
    weighting_str = inputs.get("weighting", "BLEND_70_30").upper()
    cost_bps = float(inputs.get("cost_bps", 20.0))
    label = inputs.get("label", f"sector_momentum_{momentum_window}d_top{top_n}")

    try:
        from quant_stack.research.strategies.sector_momentum import (
            SectorMomentumStrategy,
            WeightingScheme,
            compute_strength,
        )
        from quant_stack.research.vbt_adapter import VbtRunConfig, run_vbt_backtest, signal_frame_to_weights
        from quant_stack.signals.base import SignalFrame
    except ImportError as e:
        return {"status": "error", "error": f"Missing dependency: {e}. pip install vectorbt"}

    try:
        strategy = SectorMomentumStrategy(momentum_window=momentum_window, top_n=top_n)
        signals = strategy.generate_signals(ctx.close_df)

        scheme_map = {
            "BLEND_70_30": WeightingScheme.BLEND_70_30,
            "BLEND_50_50": WeightingScheme.BLEND_50_50,
            "EQUAL": WeightingScheme.EQUAL,
            "INVERSE_VOL": WeightingScheme.INVERSE_VOL,
        }
        scheme = scheme_map.get(weighting_str, WeightingScheme.BLEND_70_30)
        strength = compute_strength(signals, ctx.close_df, scheme)
        sf = SignalFrame(signals=signals, strength=strength, strategy_name=strategy.name)
        weights_df = signal_frame_to_weights(sf)

        half_bps = cost_bps / 2 / 10_000
        config = VbtRunConfig(commission=half_bps, slippage=half_bps)

        import pandas as pd
        benchmark = None
        for bm in ("SPY", "QQQ"):
            if bm in ctx.close_df.columns:
                benchmark = ctx.close_df[bm]
                break

        result = run_vbt_backtest(
            close=ctx.close_df,
            weights=weights_df,
            config=config,
            benchmark_close=benchmark,
            strategy_name=strategy.name,
        )
    except Exception as e:
        return {"status": "error", "error": f"Backtest failed: {e}"}

    # Save ExperimentRecord
    try:
        from quant_stack.agent.experiment_tracker import save_record
        from quant_stack.core.schemas import ExperimentRecord

        record = ExperimentRecord(
            description=label,
            symbols=list(ctx.close_df.columns),
            period_start=ctx.close_df.index[0].date(),
            period_end=ctx.close_df.index[-1].date(),
            strategy_params={
                "momentum_window": momentum_window,
                "top_n": top_n,
                "weighting": weighting_str,
                "cost_bps": cost_bps,
            },
            backtest_result=result,
            tags=["sector_momentum", "agent_tool"],
        )
        record_path = save_record(record, base_dir="experiments/records")
        ctx.last_record_path = str(record_path)
    except Exception as e:
        record_path = None

    return {
        "status": "ok",
        "label": label,
        "cagr": round(result.cagr, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "max_drawdown": round(result.max_drawdown, 4),
        "total_return": round(result.total_return, 4),
        "n_trades": result.n_trades,
        "record_path": str(record_path) if record_path else None,
    }


# ── Anthropic tool schemas ────────────────────────────────────────────────────

RESEARCH_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "load_market_data_tool",
        "description": (
            "Download adjusted daily close prices from Yahoo Finance for a list of symbols. "
            "Stores price data in session context for subsequent tool calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ticker symbols to download (e.g. [\"IBB\", \"QQQ\", \"XLE\"])",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Approximate trading-day history to fetch (default 350)",
                },
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "build_factors_tool",
        "description": (
            "Compute rolling price-ROC momentum for every symbol in context. "
            "Requires load_market_data_tool to have been called first. "
            "Stores continuous momentum scores; does NOT apply top-N ranking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "momentum_window": {
                    "type": "integer",
                    "description": "Lookback in trading days (default 210 ≈ 10 months)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_signals_tool",
        "description": (
            "Apply cross-sectional top-N ranking to momentum scores. "
            "Requires build_factors_tool to have been called first. "
            "Returns which symbols are selected (signal=1) vs excluded (signal=0)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of symbols to hold simultaneously (default 3)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "allocate_portfolio_tool",
        "description": (
            "Compute target portfolio weights from signals using a blending scheme. "
            "Requires generate_signals_tool to have been called first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "weighting": {
                    "type": "string",
                    "enum": ["BLEND_70_30", "BLEND_50_50", "EQUAL", "INVERSE_VOL"],
                    "description": "Weighting scheme (default BLEND_70_30)",
                },
                "vol_window": {
                    "type": "integer",
                    "description": "Rolling window for volatility estimation (default 63)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "run_research_backtest_tool",
        "description": (
            "Run a full vectorbt backtest with the sector momentum strategy. "
            "Requires load_market_data_tool to have been called first. "
            "Saves an ExperimentRecord to disk and returns performance metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "momentum_window": {"type": "integer", "description": "Lookback days (default 210)"},
                "top_n": {"type": "integer", "description": "Number of holdings (default 3)"},
                "weighting": {
                    "type": "string",
                    "enum": ["BLEND_70_30", "BLEND_50_50", "EQUAL", "INVERSE_VOL"],
                },
                "cost_bps": {"type": "number", "description": "Round-trip cost in basis points (default 20)"},
                "label": {"type": "string", "description": "Human-readable experiment label"},
            },
            "required": [],
        },
    },
]
