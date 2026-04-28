"""Shadow Run — dry-run / shadow execution of the formal sector momentum strategy.

This is the official entrypoint for human-supervised shadow execution.
No orders are submitted. All outputs are for manual review only.

Usage
-----
    # First run (all-cash):
    python experiments/shadow_run.py

    # With an existing portfolio:
    python experiments/shadow_run.py --positions data/current_positions.json

    # Override NAV:
    python experiments/shadow_run.py --positions data/current_positions.json --nav 150000

Inputs
------
    data/current_positions.json  — your current holdings
                                   (copy/edit template before running)

Outputs
-------
    shadow_artifacts/{run_id}/
        current_positions_snapshot.json  — serialised input snapshot
        target_weights_snapshot.json     — strategy output
        rebalance_plan.json              — full order plan with all diffs
        risk_check_result.json           — all risk gate results
        shadow_execution_summary.md      — human review document  ← read this
        execution_log.jsonl              — structured audit log

    shadow_artifacts/latest/             — same files, always the most recent run

    execution_artifacts/                 — legacy per-adapter artifacts from
                                           RebalanceService (also written)

What this script does NOT do
-----------------------------
    - Does not connect to any broker
    - Does not submit orders
    - Does not modify strategy parameters
    - Does not implement live trading
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Imports from quant_stack ───────────────────────────────────────────────────
from quant_stack.core.config import AppConfig, load_config
from quant_stack.core.schemas import PortfolioWeights
from quant_stack.execution.adapters import DryRunExecutionAdapter
from quant_stack.execution.domain import PositionSnapshot, target_weights_from_portfolio_weights
from quant_stack.execution.positions import load_positions_json
from quant_stack.execution.service import RebalanceService
from quant_stack.execution.shadow import ShadowExecutionService
from quant_stack.research.strategies.sector_momentum import (
    RISK_ON_UNIVERSE,
    SectorMomentumStrategy,
    WeightingScheme,
    compute_strength,
)
from quant_stack.research.vbt_adapter import signal_frame_to_weights
from quant_stack.signals.base import SignalFrame

# ── Strategy constants (must match formal strategy; do NOT change) ─────────────
STRATEGY_NAME = "sector_momentum_210d_top3"
MOMENTUM_WINDOW = 210
TOP_N = 3
VOL_WINDOW = 63
WEIGHTING_METHOD_DISPLAY = "BLEND_70_30 (70% equal + 30% inverse-vol)"
UNIVERSE_TYPE_DISPLAY = "Sector / industry / thematic ETFs"

# ── Paths ──────────────────────────────────────────────────────────────────────
POSITIONS_DEFAULT = Path("data/current_positions.json")
SHADOW_DIR = Path("shadow_artifacts")
EXECUTION_ARTIFACTS_DIR = Path("execution_artifacts")


def _download_fresh_prices(symbols: list[str], lookback_days: int = 350) -> "pd.DataFrame":
    """Download recent adjusted close prices from Yahoo Finance."""
    import pandas as pd

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for live signal generation.\n"
            "Install it with: pip install yfinance"
        )

    end = date.today()
    # extra buffer for weekends/holidays/missing sessions
    start = end - timedelta(days=lookback_days + 120)

    logger.info(f"Downloading prices for {symbols} ({start} to {end})")
    raw = yf.download(
        symbols,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance returned empty data. Check internet connection and symbol list."
        )

    # yfinance 1.x returns MultiIndex columns by default: (type, ticker).
    # raw["Close"] selects by first level and returns a ticker-column DataFrame.
    # For a flat-column result (single ticker or older yfinance), "Close" is a direct key.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    elif "Close" in raw.columns:
        close = raw[["Close"]].droplevel(0, axis=1) if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]
    else:
        close = raw

    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])

    # Drop all-NaN rows; reorder to canonical universe order
    close = close.dropna(how="all")
    available = [s for s in symbols if s in close.columns]
    missing = [s for s in symbols if s not in close.columns]
    if missing:
        logger.warning(f"Symbols not returned by yfinance: {missing}")
    return close[available]


def _generate_target_weights(close: "pd.DataFrame") -> PortfolioWeights:
    """Run the formal strategy on fresh prices; return current target weights.

    Uses the same parameters as the formal backtest:
        momentum_window=210, top_n=3, weighting=BLEND_70_30

    The signal is computed on the latest available price bar.  This is the
    current market view — what the strategy recommends as of today.

    NOTE: Actual execution should only happen on scheduled bi-monthly rebalance
    dates.  This function does NOT enforce the rebalance schedule — it returns
    the current signal regardless of calendar position.
    """
    import pandas as pd

    strategy = SectorMomentumStrategy(momentum_window=MOMENTUM_WINDOW, top_n=TOP_N)
    signals = strategy.generate_signals(close)

    # BLEND_70_30 = 70% equal-weight + 30% inverse-vol
    strength = compute_strength(signals, close, WeightingScheme.BLEND_70_30)

    sf = SignalFrame(signals=signals, strength=strength, strategy_name=strategy.name)
    weights_df = signal_frame_to_weights(sf)

    # Last non-NaN row = current signal
    valid_rows = weights_df.dropna(how="all")
    if valid_rows.empty:
        raise RuntimeError(
            f"Insufficient price history for {MOMENTUM_WINDOW}-day momentum. "
            f"Got {len(close)} rows. Need at least {MOMENTUM_WINDOW} trading days."
        )

    latest_row = valid_rows.iloc[-1]
    signal_date = valid_rows.index[-1]
    if hasattr(signal_date, "date"):
        signal_date = signal_date.date()

    weights = {
        sym: float(w)
        for sym, w in latest_row.items()
        if pd.notna(w) and w > 0.001
    }

    logger.info(f"Signal date: {signal_date}")
    logger.info(f"Target weights: {weights}")

    return PortfolioWeights(
        weights=weights,
        method="blend_70_30",
        rebalance_date=signal_date,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shadow run — dry-run execution of sector momentum strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=POSITIONS_DEFAULT,
        help=f"Path to current_positions.json (default: {POSITIONS_DEFAULT})",
    )
    parser.add_argument(
        "--nav",
        type=float,
        default=None,
        help="Override NAV from positions file (e.g. --nav 150000)",
    )
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────────────
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")

    print("=" * 70)
    print("  Shadow Run — Sector Momentum Strategy (dry-run only)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Load config ────────────────────────────────────────────────────────────
    try:
        app_cfg = load_config("config/settings.yaml")
    except FileNotFoundError:
        app_cfg = AppConfig()
    risk_cfg = app_cfg.execution.risk

    # ── Step 1: Load current positions ─────────────────────────────────────────
    print(f"\n[1/6] Loading current positions from: {args.positions}")

    if not args.positions.exists():
        print(f"\n  ERROR: File not found: {args.positions}")
        print()
        print("  For a first run (all-cash), create this file:")
        print("  {")
        print('    "nav": 100000.0,')
        print('    "positions": {},')
        print('    "cash_fraction": 1.0,')
        print('    "source": "manual"')
        print("  }")
        print()
        print("  For an existing portfolio:")
        print("  {")
        print('    "nav": 100000.0,')
        print('    "positions": {"QQQ": 0.3333, "XLI": 0.3333, "GDX": 0.3334},')
        print('    "cash_fraction": 0.0001,')
        print('    "source": "manual"')
        print("  }")
        sys.exit(1)

    snapshot = load_positions_json(args.positions)
    if args.nav is not None:
        snapshot = PositionSnapshot(
            timestamp=snapshot.timestamp,
            nav=args.nav,
            positions=snapshot.positions,
            cash_fraction=snapshot.cash_fraction,
            source=snapshot.source,
        )

    print(f"  NAV      : ${snapshot.nav:,.2f}")
    print(f"  Source   : {snapshot.source}")
    if snapshot.positions:
        print("  Holdings :")
        for sym, w in sorted(snapshot.positions.items(), key=lambda x: -x[1]):
            print(f"    {sym:6s}  {w:.2%}  (${w * snapshot.nav:,.0f})")
    else:
        print("  Holdings : NONE — all-cash (first rebalance)")
    print(f"  Cash     : {snapshot.cash_fraction:.2%}  (${snapshot.cash_fraction * snapshot.nav:,.0f})")

    # ── Step 2: Generate current target weights ─────────────────────────────────
    print(f"\n[2/6] Generating target weights from fresh market data...")
    print(f"  Universe         : {RISK_ON_UNIVERSE}")
    print(f"  Momentum window  : {MOMENTUM_WINDOW} days (~10 months)")
    print(f"  Top-N            : {TOP_N}")
    print(f"  Weighting        : BLEND_70_30 (70% equal + 30% inverse-vol)")

    try:
        close = _download_fresh_prices(RISK_ON_UNIVERSE, lookback_days=MOMENTUM_WINDOW + 100)
    except Exception as exc:
        print(f"\n  ERROR: Could not download price data: {exc}")
        print("  Check your internet connection and try again.")
        sys.exit(1)

    try:
        portfolio_weights = _generate_target_weights(close)
    except RuntimeError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    target = target_weights_from_portfolio_weights(
        portfolio_weights,
        strategy_name=STRATEGY_NAME,
        source_record_id="",
    )

    # Extract reference prices from the last market close row (display only)
    latest_row = close.iloc[-1]
    latest_prices = {
        sym: float(latest_row[sym])
        for sym in RISK_ON_UNIVERSE
        if sym in latest_row.index and not __import__("math").isnan(float(latest_row[sym]))
    }

    print(f"\n  Signal date : {target.rebalance_date}")
    print("  Target weights:")
    for sym, w in sorted(target.weights.items(), key=lambda x: -x[1]):
        print(f"    {sym:6s}  {w:.2%}  (${w * snapshot.nav:,.0f})")

    # ── Step 3: Build and run shadow execution ──────────────────────────────────
    print(f"\n[3/6] Running dry-run shadow execution...")

    adapter = DryRunExecutionAdapter()
    service = RebalanceService(
        adapter=adapter,
        risk=risk_cfg,
        dry_run=True,
        kill_switch=False,
        stale_signal_days=5,
        min_trade_size=0.005,
        max_turnover=1.5,
        max_orders=20,
        cost_bps=20.0,
        artifacts_dir=str(EXECUTION_ARTIFACTS_DIR),
    )
    shadow_svc = ShadowExecutionService(service=service, shadow_dir=SHADOW_DIR)
    shadow_result = shadow_svc.run(
        target,
        snapshot,
        weighting_method=WEIGHTING_METHOD_DISPLAY,
        universe=RISK_ON_UNIVERSE,
        universe_type=UNIVERSE_TYPE_DISPLAY,
        latest_prices=latest_prices,
    )

    plan = shadow_result.plan
    result = shadow_result.result

    # ── Step 4: Print rebalance plan ───────────────────────────────────────────
    print(f"\n[4/6] Rebalance plan:")
    if plan.orders:
        header = f"  {'Symbol':6s}  {'Side':4s}  {'Current':>8s}  {'Target':>8s}  {'Delta':>8s}  {'Delta $':>10s}"
        sep    = f"  {'─'*6}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}"
        print(header)
        print(sep)
        for o in sorted(plan.orders, key=lambda x: x.symbol):
            cur = snapshot.positions.get(o.symbol, 0.0)
            delta_usd = o.delta_weight * snapshot.nav
            print(
                f"  {o.symbol:6s}  {str(o.side).upper():4s}"
                f"  {cur:>8.2%}  {o.target_weight:>8.2%}"
                f"  {o.delta_weight:>+8.2%}  {delta_usd:>+10,.0f}"
            )
        nav = snapshot.nav
        est_cost = plan.total_turnover * 20.0 / 10_000 * nav
        print(f"\n  Total turnover   : {plan.total_turnover:.2%}")
        print(f"  Estimated cost   : ~${est_cost:,.0f}  (20 bps × NAV)")
    else:
        print("  No orders required — portfolio already at target allocation.")

    # ── Step 5: Print risk checks ──────────────────────────────────────────────
    print(f"\n[5/6] Risk checks:")
    risk_passed = result.risk_check.passed if result.risk_check else True
    overall_label = "PASSED" if risk_passed else "FAILED"
    print(f"  Overall: {overall_label}")
    if result.risk_check and not result.risk_check.passed:
        for v in result.risk_check.violations:
            print(f"  VIOLATION: {v.message}")

    # Soft warnings from log
    for entry in result.log_entries:
        if "[STALE" in entry or "[LOW CASH]" in entry or "[RECONCILIATION]" in entry:
            print(f"  WARNING: {entry.strip()}")

    # ── Step 6: Artifacts summary ──────────────────────────────────────────────
    print(f"\n[6/6] Artifacts written:")
    print(f"  Run directory : {shadow_result.run_dir}")
    print(f"  Latest alias  : {shadow_result.latest_dir}")
    print()
    for name, path in shadow_result.artifacts.items():
        print(f"    {path.name}")

    # ── Final recommendation ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if not result.success:
        print("  STATUS: BLOCKED — resolve risk violations before executing")
    elif shadow_result.needs_rebalance:
        print("  STATUS: REBALANCE RECOMMENDED")
        print("  Review shadow_execution_summary.md, then execute orders manually.")
    else:
        print("  STATUS: NO REBALANCE NEEDED")
    print()
    print("  Human review file:")
    print(f"    {shadow_result.artifacts['shadow_execution_summary']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
