# quant-stack

Quantitative research and execution-planning stack for a sector ETF momentum strategy.

Current focus: research + shadow execution with human-in-the-loop review. This is not an automated trading system.

## Current Status

| Layer | Status |
|-------|--------|
| Formal strategy (backtest) | Stable — IBB universe, blend_70_30, 2ME rebalance |
| Shadow execution | Working — daily monitoring + human review artifact set |
| Execution planning | Working — order plan, risk checks, cost estimates, whole-share constraint |
| Human-in-the-loop | Working — execution summary, deviation rules, fill log template |
| AI research assistant | Working — post-backtest commentary via Claude API |
| Agent tools for shadow execution | In progress — `feature/agent-tools` branch, not yet on main |
| Live / automated trading | Not implemented |
| Broker integration | Not integrated — LEAN skeleton exists as a future integration point |

## Current Formal Strategy

**Sector ETF Momentum — IBB universe (as of 2026-04-28)**

| Parameter | Value |
|-----------|-------|
| Universe | IBB QQQ XLE XLV XLF XLI VTV GDX XLP |
| Momentum window | 210 days (~10 months) |
| Selection | Top-3 by cross-sectional momentum rank |
| Hysteresis | entry_margin = 0.02 — new asset must beat displaced asset by ≥ 2pp ROC |
| Weighting | blend_70_30 — 70% equal-weight + 30% inverse-vol (vol window = 63 days) |
| Rebalance | Bi-monthly (2ME) — last business day of Jan · Mar · May · Jul · Sep · Nov |
| Cost assumption | 20 bps total per side (10 bps commission + 10 bps slippage) |
| Benchmark | SPY buy-and-hold |

**Confirmed baseline (2010–2025, IBB):** Sharpe = 1.117 · CAGR = 14.17% · MaxDD = 20.51%

Universe history:
- 2026-04-24: GDX replaced SPY
- 2026-04-25: IYT replaced VNQ (dSharpe +0.063 · dCAGR +1.01% · dMaxDD −1.41%)
- 2026-04-28: IBB replaced IYT (IYT not tradeable on Rakuten Securities; IBB is the best available substitute)

## Architecture

Three independent layers with explicit boundary contracts:

```
┌─────────────────────────────────────────────────────────────────┐
│  Research layer                                                  │
│  SectorMomentumStrategy · VBT backtest · ExperimentTracker       │
│  In:  historical OHLCV prices                                    │
│  Out: BacktestResult, PortfolioWeights                           │
├─────────────────────────────────────────────────────────────────┤
│  Execution layer                                                 │
│  RebalanceService · ShadowExecutionService                       │
│  DryRunAdapter · PaperAdapter · LeanAdapter (skeleton)           │
│  In:  TargetWeights + PositionSnapshot                           │
│  Out: OrderPlan, risk check results, shadow artifacts            │
├─────────────────────────────────────────────────────────────────┤
│  Agent layer                                                     │
│  Researcher · Reporter · Orchestrator                            │
│  In:  BacktestResult                                             │
│  Out: markdown analysis (Claude API)                             │
│  Scope: research commentary only — no access to execution state  │
└─────────────────────────────────────────────────────────────────┘
```

**Research → Execution boundary:** Only `PortfolioWeights` may cross, via `target_weights_from_portfolio_weights()`. No signal DataFrames, no `BacktestResult` objects may enter the execution layer — they carry look-ahead context from the research pipeline that must not drive live orders.

## Entry Points

| Script | Role | Run when |
|--------|------|----------|
| `experiments/shadow_run.py` | **Primary daily tool** | Daily signal monitoring and human execution review |
| `experiments/sector_momentum_experiment.py` | **Formal backtest** | Validating the strategy or updating the baseline |
| `experiments/run_execution.py` | Dev / demo | Testing the execution layer with all three adapters |
| `experiments/sector_momentum_cost_study.py` | Analysis | Cost sensitivity (0 / 10 / 20 / 30 bps) |
| `experiments/portfolio_validation.py` | Analysis | SPY / strategy blend analysis (70/30, 50/50, etc.) |

## shadow_run Workflow

`shadow_run.py` is the primary operational entry point. It runs the full strategy pipeline against fresh prices and produces a human review package. No orders are ever submitted automatically.

```bash
# First run — all-cash portfolio:
python experiments/shadow_run.py

# With an existing portfolio:
python experiments/shadow_run.py --positions data/current_positions.json

# Override NAV:
python experiments/shadow_run.py --positions data/current_positions.json --nav 150000
```

**What it does on each run:**

1. Downloads recent adjusted close prices from Yahoo Finance (~430 calendar days)
2. Computes the current signal: 210-day momentum → cross-sectional ranking → entry_margin hysteresis → blend_70_30 weights
3. Reads current positions from `data/current_positions.json`
4. Runs a full dry-run execution cycle — order plan, risk checks, cost estimate
5. Determines whether today is a scheduled rebalance day (last business day of a 2ME month)
6. Writes the full artifact set to `shadow_artifacts/{run_id}/`
7. Prints a STATUS line: MONITORING / REBALANCE RECOMMENDED / BLOCKED

### Current positions file

`data/current_positions.json` (gitignored — personal portfolio data):

```json
{
  "nav_usd": 5000.00,
  "cash_usd": 100.00,
  "source": "manual",
  "as_of": "2026-05-06T09:00:00",
  "positions": {
    "IBB": { "quantity": 10, "last_price_usd": 170.16, "market_value_usd": 1701.60 },
    "XLE": { "quantity": 30, "last_price_usd": 59.41,  "market_value_usd": 1782.30 },
    "GDX": { "quantity": 16, "last_price_usd": 85.66,  "market_value_usd": 1370.56 }
  }
}
```

For an all-cash first run, set `"positions": {}` and `"cash_usd"` to your full NAV.

### Rebalance schedule

Scheduled rebalance windows: last business day of Jan · Mar · May · Jul · Sep · Nov.

The script runs regardless of date. On non-scheduled days it reports STATUS: MONITORING and shows the next window. On scheduled days with executable orders it reports STATUS: REBALANCE RECOMMENDED.

### shadow_artifacts output

```
shadow_artifacts/{run_id}/
  shadow_execution_summary.md         ← read this first — full human review document
  rebalance_plan.json                 — order plan: deltas, notionals, turnover, cost
  current_positions_snapshot.json     — input portfolio state
  target_weights_snapshot.json        — strategy output (signal date, weights)
  risk_check_result.json              — per-check pass/fail detail
  manual_execution_log_template.json  — fill in after placing orders at broker
  execution_log.jsonl                 — structured audit log (every step)

shadow_artifacts/latest/              — symlink to most recent run (always refreshed)
```

### Human execution checklist

1. Run `shadow_run.py`, open `shadow_execution_summary.md`
2. **Section A** — confirm the signal date and whether it is a scheduled rebalance day
3. **Section E** — confirm all risk checks passed (no FAIL)
4. **Section D** — review order plan (symbol, side, delta, notional)
5. **Section D.2** — read the real-time price at your broker, compute `floor(Suggested Notional / real-time price)` shares
6. **Section G.2** — apply price deviation rule (PROCEED / REVIEW / DEFER) before placing each order
7. Place orders manually at your broker
8. Record actual fills in `manual_execution_log_template.json`

### Risk checks (enforced on every run)

| Check | What it gates |
|-------|---------------|
| `kill_switch` | Hard stop — blocks all execution when True |
| `stale_signal` | Warns if market data is > 5 calendar days old |
| `position_reconciliation` | Verifies positions + cash sums to ~100% |
| `cash_sufficiency` | Buy notional (net of sell proceeds) must not exceed available cash |
| `duplicate_guard` | Prevents re-running the same signal twice in one session |
| `min_order_threshold` | Filters diffs < 0.5% to reduce noise trading |
| `max_position_size` | Blocks any target position > 40% of NAV |
| `max_turnover` | Blocks plans with total turnover > 150% |
| `max_order_count` | Blocks plans with > 20 orders |

## Execution Adapters

| Adapter | Behavior |
|---------|----------|
| `DryRunExecutionAdapter` | Logs order intents, never modifies state. Used by `shadow_run.py`. |
| `PaperExecutionAdapter` | Simulates fills in memory, tracks paper positions. Not connected to any broker API. |
| `LeanExecutionAdapter` | Writes a JSON payload to `lean_output/target_weights.json`. File handoff only — no live LEAN connection is configured. |

The LEAN algorithm skeleton (`lean/SectorMomentumAlgorithm.py`) reads the payload and calls `SetHoldings()`. It is an integration skeleton for future broker connectivity, not a connected live system.

## Agent Layer

The `agent/` module on `main` contains three research-oriented components:

| Component | What it does |
|-----------|-------------|
| `Researcher` | Sends `BacktestResult` to Claude API, returns markdown commentary |
| `Reporter` | Generates structured markdown reports from backtest + agent analysis |
| `Orchestrator` | Coordinates the full research pipeline: data → backtest → agent → report |

**Scope:** Research commentary only. The agent layer has no access to execution state, does not read or write order plans, and cannot trigger order execution.

**Agent tools for shadow execution** (`feature/agent-tools` branch, not yet merged to main): programmatic interface for querying the current shadow run, parsing the execution summary, and assisting with human review. These are in active development.

## Directory Layout

```
src/quant_stack/
├── core/
│   ├── schemas.py          # Pydantic models: BacktestResult, PortfolioWeights, ExperimentRecord
│   ├── config.py           # AppConfig, RiskConfig, load_config()
│   └── exceptions.py       # typed exceptions per layer
├── data/
│   └── providers/
│       ├── yahoo.py        # yfinance + parquet cache
│       └── csv.py          # local CSV / Parquet
├── research/
│   ├── vbt_adapter.py      # vectorbt runner: look-ahead prevention, 2ME rebalancing, cost model
│   └── strategies/
│       └── sector_momentum.py   # SectorMomentumStrategy, RISK_ON_UNIVERSE, WeightingScheme
├── execution/
│   ├── domain.py           # TargetWeights, PositionSnapshot, OrderPlan, ExecutionResult
│   ├── service.py          # RebalanceService — builds order plan, runs risk checks
│   ├── adapters.py         # DryRun / Paper / Lean adapters
│   ├── shadow.py           # ShadowExecutionService — dry-run + full artifact set
│   └── positions.py        # load_positions_json()
├── agent/
│   ├── researcher.py       # Researcher — Claude API backtest commentary
│   ├── reporter.py         # Reporter — markdown report generator
│   └── orchestrator.py     # Orchestrator — research pipeline coordinator
├── signals/
│   └── momentum.py         # relative_momentum_ranking_signal (method="first")
├── factors/
│   └── momentum.py         # price ROC factor
├── portfolio/
│   └── optimizer.py        # PyPortfolioOpt wrapper (optional extra)
└── tracking/
    ├── tracker.py          # ExperimentTracker (registry.json + per-experiment dirs)
    └── report.py           # Markdown report generator

experiments/
├── shadow_run.py                    # Primary daily tool — shadow execution
├── sector_momentum_experiment.py    # Formal strategy backtest
├── run_execution.py                 # Dev demo — all three adapters
├── sector_momentum_cost_study.py    # Cost sensitivity analysis
└── portfolio_validation.py          # Portfolio-level validation

shadow_artifacts/       # Shadow run outputs (gitignored)
data/
└── current_positions.json           # Current holdings input (gitignored)
lean/
└── SectorMomentumAlgorithm.py       # LEAN integration skeleton
```

## What This Project Does NOT Do

- **No automatic order submission.** All orders require human review and manual placement.
- **No broker connection.** There is no live API integration to any brokerage.
- **No live trading.** The LEAN adapter skeleton writes a JSON file; it is not connected to a running LEAN instance.
- **No autonomous agent.** The agent layer provides research commentary; it does not make or execute trading decisions.
- **No real-time data.** Prices are fetched from Yahoo Finance and are always at least one trading session old.
- **No intraday signals.** This is a low-frequency strategy; signal computation is end-of-day only.

## Execution Consistency Notes

- **Look-ahead prevention:** Weights computed from close prices at T are executed at T+1 business day. This is enforced by a 1 BDay shift in `vbt_adapter._prepare_orders`.
- **Rebalance timing in shadow_run:** Signal computation does not enforce the 2ME schedule — the signal is computed on the latest available price bar regardless of date. The schedule gate lives in `ShadowExecutionService` (`_rebalance_schedule_info`) and controls the STATUS recommendation only.
- **Rebalance timing in backtest:** `rebalance_freq="2ME"` is passed to VBT, which enforces actual portfolio changes only on scheduled dates in the simulation.
- **Research → Execution boundary:** Call `target_weights_from_portfolio_weights()` to cross. No signal DataFrames, no `BacktestResult` objects may enter the execution layer.
- **Cost model:** `commission` and `slippage` in `VbtRunConfig` are one-way fractions. Formal strategy uses 10 bps each (20 bps total round-trip).
- **Tie-breaking:** `relative_momentum_ranking_signal` uses `method="first"` — universe column order in `RISK_ON_UNIVERSE` is canonical and must not be reordered.

## Install

```bash
# Core + research (required for backtests and shadow_run)
pip install -e ".[research]"

# Optional: portfolio optimization
pip install -e ".[research,portfolio]"

# Full dev stack
pip install -e ".[all]"

# Set ANTHROPIC_API_KEY if using the agent layer
cp .env.example .env
```

```bash
# Run tests
pytest
```

## Optional Dependencies

| Extra | Installs | Enables |
|-------|----------|---------|
| `research` | vectorbt, matplotlib, seaborn | `run_vbt_backtest()`, strategy backtests |
| `portfolio` | PyPortfolioOpt, cvxpy | `portfolio/optimizer.py` |
| `polars` | polars, pyarrow | (future) Polars data layer |
| `dev` | pytest, ruff, mypy | testing and linting |
| `all` | everything above | full stack |

## Roadmap

Only items that have been explicitly decided:

- **Merge `feature/agent-tools` to main:** Shadow execution agent tools — programmatic interface to query shadow run state, parse execution summary, assist human review.

No timeline or plan for:
- Automated order submission
- Broker API integration
- Real-time or intraday data
