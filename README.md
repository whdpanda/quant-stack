# quant-stack

Agent-Assisted Quant Research & Deployment Stack for low-frequency equity/ETF strategies.

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
├─────────────────────────────────────────────────────┤
│  Execution     LEAN bridge stub (paper / live)        │  ← future
└─────────────────────────────────────────────────────┘
```

## Quick start

```bash
# 1. Install core + research + portfolio extras
pip install -e ".[research,portfolio]"

# 2. Copy environment template
cp .env.example .env
# Add ANTHROPIC_API_KEY if you want agent analysis

# 3. Run the CLI
quant info                                       # check layer availability
quant backtest --symbols SPY QQQ --start 2020-01-01 --end 2023-12-31
quant optimise --symbols SPY QQQ IEF GLD --start 2018-01-01

# 4. Run the end-to-end experiment script
python experiments/example_backtest.py

# 5. Run tests
pytest
```

## Directory layout

```
src/quant_stack/
├── cli.py              # `quant` CLI (typer)
├── core/
│   ├── schemas.py      # Pydantic models shared across layers
│   ├── logging.py      # loguru setup from config/logging.yaml
│   └── exceptions.py   # typed exceptions per layer
├── data/
│   ├── base.py         # DataProvider ABC
│   └── providers/
│       ├── yahoo.py    # yfinance + parquet cache
│       └── csv.py      # local CSV / Parquet
├── research/
│   ├── base.py         # Strategy ABC
│   ├── backtest.py     # vectorbt runner
│   └── strategies/
│       └── sma_cross.py
├── portfolio/
│   └── optimizer.py    # PyPortfolioOpt wrapper
├── execution/
│   ├── base.py         # Executor ABC
│   └── lean_bridge.py  # LEAN stub
└── agent/
    ├── researcher.py   # Claude-powered backtest analysis
    ├── reporter.py     # Markdown report generation
    └── orchestrator.py # End-to-end pipeline coordinator
```

## Adding a new strategy

```python
# src/quant_stack/research/strategies/my_strategy.py
from quant_stack.research.base import Strategy
import pandas as pd

class MyStrategy(Strategy):
    name = "my_strategy"

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        # Return DataFrame of same shape; True/1.0 = hold long
        ...
```

## Polars migration path

`DataProvider.fetch()` returns `pd.DataFrame` today. When migrating:
1. Add `DataProvider.fetch_polars()` returning `pl.DataFrame`.
2. Implement in each provider independently.
3. `transforms.py` gains a `pl`-backed parallel.
4. No changes needed in research / portfolio layers until they consume data directly.

## Connecting LEAN

1. Install LEAN: `docker pull quantconnect/lean`
2. Populate `lean/config.json` with broker credentials.
3. Implement `LeanBridge._send_order()` using LEAN's REST API.
4. Set `execution.mode: live` in `config/settings.yaml`.

## Optional dependencies

| Extra | Installs | Enables |
|-------|----------|---------|
| `research` | vectorbt, matplotlib | `run_backtest()` |
| `portfolio` | PyPortfolioOpt, cvxpy | `optimize_portfolio()` |
| `polars` | polars, pyarrow | future Polars data layer |
| `dev` | pytest, ruff, mypy | testing & linting |
| `all` | everything above | full stack |
