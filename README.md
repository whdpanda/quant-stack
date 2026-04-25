# quant-stack

Agent-Assisted Quant Research & Deployment Stack for low-frequency sector ETF momentum strategies.

## Current Formal Strategy

**Sector ETF Momentum — IYT universe (as of 2026-04-25)**

| Parameter | Value |
|-----------|-------|
| Universe | IYT QQQ XLE XLV XLF XLI VTV GDX XLP |
| Momentum window | 210 days (~10 months) |
| Selection | Top-3 by cross-sectional momentum rank |
| Weighting | blend_70_30 (70% equal + 30% inverse-vol) |
| Rebalance | Bi-monthly (2ME), executed T+1 business day |
| Hysteresis | entry_margin = 0.02 (new asset must beat displaced by ≥ 2pp ROC) |
| Cost | 20 bps total per side (10 bps commission + 10 bps slippage) |
| Benchmark | SPY buy-and-hold |

**Confirmed baseline (2010–2025):** Sharpe=1.162 · CAGR=14.13% · MaxDD=20.12%

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Agent Layer   researcher · reporter · orchestrator  │  ← Claude API
├─────────────────────────────────────────────────────┤
│  Portfolio     PyPortfolioOpt (mean-variance)         │
├─────────────────────────────────────────────────────┤
│  Research      vectorbt (strategy backtesting)        │
├─────────────────────────────────────────────────────┤
│  Data          pandas  (Polars interface pre-built)   │  ← yfinance / CSV
└─────────────────────────────────────────────────────┘
```

## Quick start

```bash
# 1. Install core + research + portfolio extras
pip install -e ".[research,portfolio]"

# 2. Copy environment template
cp .env.example .env
# Add ANTHROPIC_API_KEY if you want agent analysis

# 3. Run the formal strategy
python experiments/sector_momentum_experiment.py

# 4. Run cost sensitivity analysis
python experiments/sector_momentum_cost_study.py

# 5. Run portfolio-level validation (SPY/strategy blends)
python experiments/portfolio_validation.py

# 6. Run tests
pytest
```

## Experiment scripts

| Script | Purpose |
|--------|---------|
| `sector_momentum_experiment.py` | **Formal strategy entry point** — run this to reproduce the baseline |
| `sector_momentum_cost_study.py` | Cost sensitivity (0/10/20/30 bps) — confirms strategy robustness to costs |
| `portfolio_validation.py` | Portfolio-level validation — SPY/strategy blend analysis (70/30, 50/50, etc.) |

## Directory layout

```
src/quant_stack/
├── core/
│   ├── schemas.py      # Pydantic models shared across layers
│   ├── config.py       # AppConfig + layer configs, load_config()
│   └── exceptions.py   # typed exceptions per layer
├── data/
│   └── providers/
│       ├── yahoo.py    # yfinance + parquet cache
│       └── csv.py      # local CSV / Parquet
├── research/
│   ├── vbt_adapter.py  # vectorbt runner (look-ahead prevention, cost model)
│   └── strategies/
│       └── sector_momentum.py  # formal strategy + RISK_ON_UNIVERSE
├── signals/
│   └── momentum.py     # relative_momentum_ranking_signal (method="first")
├── factors/
│   └── momentum.py     # price ROC factor
├── portfolio/
│   └── optimizer.py    # PyPortfolioOpt wrapper
└── tracking/
    ├── tracker.py      # ExperimentTracker (registry.json + per-experiment dirs)
    └── report.py       # Markdown report generator
```

## Execution consistency notes

The research layer and any future execution layer share these timing contracts:

- **Look-ahead prevention**: Weights computed from close prices at T are executed at T+1 business day (1 BDay shift in `vbt_adapter._prepare_orders`).
- **Rebalance timing**: `rebalance_freq="2ME"` → last trading day of every odd month → execute on the next business day's open. Execution must trigger weight computation at month-end close, not intraday.
- **Signal generation boundary**: `SignalSource.RESEARCH` signals from backtests must never be consumed directly by execution. The execution layer must re-generate signals through its own pipeline at runtime.
- **Cost model**: Both `commission` and `slippage` in `VbtRunConfig` are one-way fractions. Formal strategy uses 10 bps each (20 bps total). Any live execution layer must measure actual costs against this assumption.
- **Tie-breaking**: `relative_momentum_ranking_signal` uses `method="first"` — earlier DataFrame columns win ties. Universe column order in `RISK_ON_UNIVERSE` is canonical and must be preserved.

## Adding a new strategy

```python
# src/quant_stack/research/strategies/my_strategy.py
from quant_stack.research.base import Strategy
import pandas as pd

class MyStrategy(Strategy):
    name = "my_strategy"

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        # Return DataFrame of same shape; 1.0 = hold long, 0.0 = flat, NaN = warmup
        ...
```

## Polars migration path

`DataProvider.fetch()` returns `pd.DataFrame` today. When migrating:
1. Add `DataProvider.fetch_polars()` returning `pl.DataFrame`.
2. Implement in each provider independently.
3. `transforms.py` gains a `pl`-backed parallel.
4. No changes needed in research / portfolio layers until they consume data directly.

## Optional dependencies

| Extra | Installs | Enables |
|-------|----------|---------|
| `research` | vectorbt, matplotlib | `run_backtest()`, `vbt_adapter` |
| `portfolio` | PyPortfolioOpt, cvxpy | `optimize_portfolio()` |
| `polars` | polars, pyarrow | future Polars data layer |
| `dev` | pytest, ruff, mypy | testing & linting |
| `all` | everything above | full stack |
