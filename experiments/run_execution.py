"""Execution layer demo — research output → rebalance plan → adapter outputs.

This script demonstrates the complete research-to-execution handoff:

  1. Load the latest formal-strategy baseline from the experiment registry
  2. Extract PortfolioWeights → convert to TargetWeights (crossing the
     research/execution boundary explicitly)
  3. Build a PositionSnapshot representing the current portfolio state
     (two scenarios: all-cash first run, and partially-invested rebalance)
  4. Run RebalanceService with DryRunExecutionAdapter    (always safe)
  5. Run RebalanceService with PaperExecutionAdapter     (simulates fills)
  6. Run RebalanceService with LeanExecutionAdapter      (writes LEAN payload)
  7. Print all artifacts: order plan, risk check, execution log, LEAN payload

Run
---
    python experiments/run_execution.py

Outputs
-------
  execution_artifacts/
      *_order_plan.json         — full order plan with deltas
      *_execution_log.txt       — per-adapter log
  lean_output/
      target_weights.json       — LEAN algorithm payload

Requires
--------
    pip install 'quant-stack[research]'   pip install yfinance
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

from loguru import logger

from quant_stack.core.config import AppConfig, RiskConfig, load_config
from quant_stack.core.schemas import PortfolioWeights
from quant_stack.execution.adapters import (
    DryRunExecutionAdapter,
    LeanExecutionAdapter,
    PaperExecutionAdapter,
)
from quant_stack.execution.domain import (
    PositionSnapshot,
    TargetWeights,
    target_weights_from_portfolio_weights,
)
from quant_stack.execution.service import RebalanceService
from quant_stack.tracking import ExperimentTracker

# ── Silence loguru debug noise for this script ────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}")

# ── Config ─────────────────────────────────────────────────────────────────────

try:
    app_cfg = load_config("config/settings.yaml")
except FileNotFoundError:
    app_cfg = AppConfig()

RISK_CFG = app_cfg.execution.risk

# ── 1. Load latest baseline experiment ────────────────────────────────────────

print("=" * 68)
print("  Execution Layer Demo — Sector Momentum → Rebalance Plan")
print("=" * 68)

tracker = ExperimentTracker("./experiments")
baseline_entries = tracker.list_experiments(tag="baseline", limit=5)
sector_entries   = [e for e in baseline_entries if "sector-rotation" in e.get("tags", [])]

portfolio_weights: PortfolioWeights | None = None
strategy_name = "sector_momentum_210d_top3"
source_record_id = ""

if sector_entries:
    latest = sector_entries[0]
    exp_dir = tracker.base_dir / latest["exp_dir"]
    record_path = exp_dir / "record.json"
    if record_path.exists():
        try:
            record = tracker.load(latest["experiment_id"])
            portfolio_weights = record.portfolio_weights
            if record.backtest_result:
                strategy_name = record.backtest_result.strategy_name
            source_record_id = record.experiment_id
            print(f"\n[SOURCE] Loaded experiment: {latest['exp_dir']}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[SOURCE] Could not load record: {exc}")

# Fallback: use known last-rebalance weights from the formal strategy
if portfolio_weights is None:
    print("\n[SOURCE] No saved record found — using known formal-strategy weights")
    portfolio_weights = PortfolioWeights(
        weights={"QQQ": 0.3333, "XLI": 0.3333, "GDX": 0.3334},
        method="blend_70_30",
        rebalance_date=None,   # will default to today
    )

# ── 2. Convert to execution-layer TargetWeights ────────────────────────────────

target = target_weights_from_portfolio_weights(
    portfolio_weights,
    strategy_name=strategy_name,
    source_record_id=source_record_id,
)

print(f"\n[TARGET WEIGHTS]  {target.strategy_name}  rebal={target.rebalance_date}")
for sym, w in sorted(target.weights.items(), key=lambda x: -x[1]):
    print(f"  {sym:6s}  {w:.2%}")

# ── 3a. Scenario A — all-cash (first rebalance) ────────────────────────────────

snapshot_cash = PositionSnapshot(
    timestamp=datetime.now(),
    nav=100_000.0,
    positions={},          # no existing positions
    cash_fraction=1.0,
    source="manual",
)

# ── 3b. Scenario B — partially invested (routine rebalance) ────────────────────
# Simulate holding the *previous* allocation: QQQ + XLV + XLI equally.
snapshot_invested = PositionSnapshot(
    timestamp=datetime.now(),
    nav=100_000.0,
    positions={"QQQ": 0.333, "XLV": 0.333, "XLI": 0.334},
    cash_fraction=0.0,
    source="paper",
)

# ── 4. DryRun adapter ──────────────────────────────────────────────────────────

print("\n" + "─" * 68)
print("  ADAPTER 1: DryRunExecutionAdapter (all-cash scenario)")
print("─" * 68)

dry_adapter = DryRunExecutionAdapter()
svc_dry = RebalanceService(
    adapter=dry_adapter,
    risk=RISK_CFG,
    dry_run=True,       # always safe — nothing executes
    artifacts_dir="./execution_artifacts",
    cost_bps=20.0,
)
plan_dry, result_dry = svc_dry.run(target, snapshot_cash)

print(f"\n  orders   : {len(plan_dry.orders)}")
print(f"  turnover : {plan_dry.total_turnover:.2%}")
print(f"  risk ok  : {result_dry.risk_check.passed if result_dry.risk_check else 'N/A'}")
print(f"  success  : {result_dry.success}")

# ── 5. Paper adapter ───────────────────────────────────────────────────────────

print("\n" + "─" * 68)
print("  ADAPTER 2: PaperExecutionAdapter (partial rebalance scenario)")
print("─" * 68)

paper_adapter = PaperExecutionAdapter()
svc_paper = RebalanceService(
    adapter=paper_adapter,
    risk=RISK_CFG,
    dry_run=False,      # paper adapter simulates and persists internal positions
    artifacts_dir="./execution_artifacts",
    cost_bps=20.0,
)
plan_paper, result_paper = svc_paper.run(target, snapshot_invested)

print(f"\n  current  : {dict(snapshot_invested.positions)}")
print(f"  target   : {target.weights}")
print(f"  orders   : {len(plan_paper.orders)} filled={result_paper.orders_filled}"
      f" rejected={result_paper.orders_rejected}")
print(f"  paper positions after: {paper_adapter.positions}")
print(f"  est cost : ~${result_paper.estimated_cost:,.0f}")

# ── 6. Lean adapter ────────────────────────────────────────────────────────────

print("\n" + "─" * 68)
print("  ADAPTER 3: LeanExecutionAdapter (writes lean_output/target_weights.json)")
print("─" * 68)

lean_adapter = LeanExecutionAdapter(output_dir="./lean_output")
svc_lean = RebalanceService(
    adapter=lean_adapter,
    risk=RISK_CFG,
    dry_run=False,      # writes payload to disk
    artifacts_dir="./execution_artifacts",
    cost_bps=20.0,
)
plan_lean, result_lean = svc_lean.run(target, snapshot_cash)

print("\n  LEAN payload written to: lean_output/target_weights.json")
print("  Payload contents:")
print(json.dumps(result_lean.lean_payload, indent=4))

# ── 7. Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("  EXECUTION SUMMARY")
print("=" * 68)

rows = [
    ("Adapter",         "DryRun",                "Paper",               "Lean"),
    ("Mode",            result_dry.adapter_mode, result_paper.adapter_mode, result_lean.adapter_mode),
    ("Orders",          str(len(plan_dry.orders)), str(result_paper.orders_filled), str(result_lean.orders_filled)),
    ("Risk passed",     str(result_dry.risk_check.passed if result_dry.risk_check else "—"),
                        str(result_paper.risk_check.passed if result_paper.risk_check else "—"),
                        str(result_lean.risk_check.passed if result_lean.risk_check else "—")),
    ("Est cost ($)",    f"${result_dry.estimated_cost:,.0f}",
                        f"${result_paper.estimated_cost:,.0f}",
                        f"${result_lean.estimated_cost:,.0f}"),
    ("Success",         str(result_dry.success), str(result_paper.success), str(result_lean.success)),
]
col_w = 22
for row in rows:
    print("  " + "".join(f"{c:<{col_w}}" for c in row))

print("\n  Artifacts directory : ./execution_artifacts/")
print("  LEAN payload        : ./lean_output/target_weights.json")
print("\n  LEAN algorithm      : lean/SectorMomentumAlgorithm.py")
print("  (reads lean_output/target_weights.json at each rebalance)")
