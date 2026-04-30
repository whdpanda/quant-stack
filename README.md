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
┌──────────────────────────────────────────────────────────────┐
│  AI Agent Layer   ShadowAgent (multi-LLM) + 11 agent tools   │  ← Anthropic / OpenAI / DeepSeek / Groq / Ollama
├──────────────────────────────────────────────────────────────┤
│  Shadow Execution  ShadowExecutionService (dry-run only)      │  ← risk checks · artifacts · audit trail
├──────────────────────────────────────────────────────────────┤
│  Agent Tools       researcher · reporter · orchestrator       │  ← Claude API
├──────────────────────────────────────────────────────────────┤
│  Portfolio         PyPortfolioOpt (mean-variance)             │
├──────────────────────────────────────────────────────────────┤
│  Research          vectorbt (strategy backtesting)            │
├──────────────────────────────────────────────────────────────┤
│  Data              pandas  (Polars interface pre-built)       │  ← yfinance / CSV
└──────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
# 1. Install core + research + portfolio extras
pip install -e ".[research,portfolio]"

# 2. Copy environment template
cp .env.example .env
# Add ANTHROPIC_API_KEY (or whichever provider you use)

# 3. Run the formal strategy backtest
python experiments/sector_momentum_experiment.py

# 4. Run shadow execution demo (direct tool chain, no LLM)
python experiments/agent_shadow_demo.py

# 5. Chat with the shadow AI agent (interactive)
python experiments/agent_shadow_chat.py

# 6. Run tests
pytest
```

## Shadow Execution

Shadow execution is the bridge between research and live trading. It re-runs the full signal → allocation → risk-check pipeline on live data, produces a rebalance plan, and writes a structured audit trail — but **never submits orders**. A human always makes the final trading decision.

### Running a shadow execution

```bash
# Direct tool chain (no LLM) — fastest way to produce artifacts
python experiments/agent_shadow_demo.py
python experiments/agent_shadow_demo.py --positions data/current_positions.json
python experiments/agent_shadow_demo.py --nav 150000

# AI agent (natural language interface)
python experiments/agent_shadow_chat.py --query "读取 latest shadow run，告诉我今天是否适合执行"
python experiments/agent_shadow_chat.py                  # interactive multi-turn
python experiments/agent_shadow_chat.py --provider deepseek --verbose
```

### Artifacts written per run

Each run writes to `shadow_artifacts/<run_id>/` and symlinks to `shadow_artifacts/latest/`:

| File | Contents |
|------|----------|
| `shadow_execution_summary.md` | Human-readable audit summary (Sections A–G) |
| `execution_template.md` | Blank order template for manual execution |
| `target_weights.json` | Strategy weights (formal, full-NAV basis) |
| `rebalance_plan.json` | Per-symbol order list |
| `risk_check_result.json` | Per-check PASS / WARN / FAIL results |
| `positions_snapshot.json` | Holdings at time of run |
| `metadata.json` | Run ID, timestamp, strategy params |

### Two-layer cash semantics

The strategy plans at full NAV (Layer 1). `cash_sufficiency = WARN` is **expected and normal** whenever the portfolio already holds positions — it means the formal plan cannot be 100% funded at face value, not that execution is blocked.

The shadow summary always produces two practical figures (Layer 2):
- **Section D.1 — Tradeable NAV**: liquidation value available for rebalancing
- **Section D.2 — Suggested Notional**: recommended dollar amount per ETF, scaled to Tradeable NAV

Execute against Section D.2, not against full NAV.

## AI Agent

`ShadowAgent` wraps the 11 shadow-run tools with an LLM tool-use loop. The provider is pluggable.

### Supported LLM providers

| Provider | Default model | Env var |
|----------|--------------|---------|
| `anthropic` (default) | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `ollama` | `qwen2.5:14b` | — (local, no key) |
| `openai-compatible` | any | — (pass `--model` + `base_url`) |

```bash
# Switch provider
python experiments/agent_shadow_chat.py --provider openai --model gpt-4o-mini
python experiments/agent_shadow_chat.py --provider groq
python experiments/agent_shadow_chat.py --provider ollama --model llama3.2:3b

# Use in Python
from quant_stack.agent import ShadowAgent, create_backend

agent = ShadowAgent()                                          # Anthropic (default)
agent = ShadowAgent(provider="deepseek")
agent = ShadowAgent(backend=create_backend("groq", model="llama-3.3-70b-versatile"))

print(agent.run("帮我解释今天 shadow run 为什么建议买这些 ETF"))
agent.run("读取 risk_check_result 并用中文解释每一项检查")    # multi-turn
agent.reset()                                                  # fresh session
```

### Safety boundaries (hard — never bypassed)

- Only the 11 approved shadow-run tools can be dispatched (whitelist enforced at dispatch layer)
- 10 broker/order tools are permanently blocked (`submit_order_tool`, `place_order_tool`, etc.)
- `summarize_shadow_run_tool` always runs with `dry_run=True` regardless of inputs
- Agent cannot modify strategy parameters, universe, or weighting scheme
- Human always makes the final trading decision

### 11 agent tools

| Tool | Purpose |
|------|---------|
| `load_market_data_tool` | Download OHLCV prices for the universe |
| `build_factors_tool` | Compute 210-day momentum scores |
| `generate_signals_tool` | Select top-N ETFs by momentum rank |
| `allocate_portfolio_tool` | Compute BLEND_70_30 weights |
| `run_research_backtest_tool` | Run a historical backtest |
| `generate_report_tool` | Write a Markdown experiment report |
| `compare_experiments_tool` | Compare two experiment runs |
| `load_current_positions_tool` | Read current holdings from JSON |
| `build_rebalance_plan_tool` | Preview rebalance orders (no file writes) |
| `summarize_shadow_run_tool` | Full shadow run → 7 artifact files |
| `review_execution_artifacts_tool` | Read any artifact from a shadow run |

## Experiment scripts

| Script | Purpose |
|--------|---------|
| `sector_momentum_experiment.py` | **Formal strategy entry point** — reproduces the baseline |
| `sector_momentum_cost_study.py` | Cost sensitivity (0 / 10 / 20 / 30 bps) |
| `portfolio_validation.py` | SPY / strategy blend analysis (70/30, 50/50, etc.) |
| `agent_shadow_demo.py` | Shadow execution demo — direct 8-step tool chain (no LLM) |
| `agent_shadow_chat.py` | Shadow AI agent CLI — single-shot or interactive multi-turn |

## Directory layout

```
src/quant_stack/
├── core/
│   ├── schemas.py          # Pydantic models shared across layers
│   ├── config.py           # AppConfig + layer configs, load_config()
│   └── exceptions.py       # typed exceptions per layer
├── data/
│   └── providers/
│       ├── yahoo.py        # yfinance + parquet cache
│       └── csv.py          # local CSV / Parquet
├── research/
│   ├── vbt_adapter.py      # vectorbt runner (look-ahead prevention, cost model)
│   └── strategies/
│       └── sector_momentum.py  # formal strategy + RISK_ON_UNIVERSE
├── signals/
│   └── momentum.py         # relative_momentum_ranking_signal (method="first")
├── factors/
│   └── momentum.py         # price ROC factor
├── portfolio/
│   └── optimizer.py        # PyPortfolioOpt wrapper
├── execution/
│   └── shadow_run.py       # ShadowExecutionService (dry-run execution pipeline)
├── tracking/
│   ├── tracker.py          # ExperimentTracker (registry.json + per-experiment dirs)
│   └── report.py           # Markdown report generator
└── agent/
    ├── providers.py        # LLMBackend ABC + AnthropicBackend + OpenAICompatibleBackend
    ├── shadow_agent.py     # ShadowAgent + ShadowAgentContext + safety guardrails
    ├── orchestrator.py     # Orchestrator
    ├── researcher.py       # Researcher
    ├── reporter.py         # Reporter
    └── tools/
        ├── _context.py     # ToolContext (shared in-memory state across tool calls)
        ├── research_tools.py   # 5 research tools + schemas
        ├── report_tools.py     # 2 report/experiment tools + schemas
        └── shadow_tools.py     # 4 shadow/audit tools + schemas
```

## Execution consistency notes

- **Look-ahead prevention**: Weights computed from close prices at T are executed at T+1 business day (1 BDay shift in `vbt_adapter._prepare_orders`).
- **Rebalance timing**: `rebalance_freq="2ME"` → last trading day of every odd month → execute on the next business day's open.
- **Signal generation boundary**: Research signals from backtests must never be consumed directly by execution. The execution layer re-generates signals at runtime through its own pipeline.
- **Cost model**: `commission` and `slippage` in `VbtRunConfig` are one-way fractions. Formal strategy uses 10 bps each (20 bps total).
- **Tie-breaking**: `relative_momentum_ranking_signal` uses `method="first"` — earlier DataFrame columns win ties. Universe column order in `RISK_ON_UNIVERSE` is canonical.

## Optional dependencies

| Extra | Installs | Enables |
|-------|----------|---------|
| `research` | vectorbt, matplotlib | `run_backtest()`, `vbt_adapter` |
| `portfolio` | PyPortfolioOpt, cvxpy | `optimize_portfolio()` |
| `polars` | polars, pyarrow | future Polars data layer |
| `dev` | pytest, ruff, mypy | testing & linting |
| `all` | everything above | full stack |

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

## Adding a new LLM provider

```python
# src/quant_stack/agent/providers.py
from quant_stack.agent.providers import LLMBackend, LLMResponse

class MyBackend(LLMBackend):
    def add_user_message(self, content: str) -> None: ...
    def add_tool_results(self, results): ...
    def chat(self, *, system, tools, max_tokens) -> LLMResponse: ...
    def reset_history(self) -> None: ...

    @property
    def provider_name(self) -> str:
        return "my-provider"

# Use it directly
from quant_stack.agent import ShadowAgent
agent = ShadowAgent(backend=MyBackend())
```
