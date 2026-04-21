"""End-to-end example: data → strategy → backtest → portfolio → report.

Run with:
    python experiments/example_backtest.py

Requires: pip install 'quant-stack[research,portfolio]'
"""

from __future__ import annotations

from datetime import date

from quant_stack.core.logging import setup_logging
from quant_stack.core.schemas import BacktestConfig, DataConfig, PortfolioConfig
from quant_stack.data.providers.yahoo import YahooProvider
from quant_stack.data.transforms import simple_returns
from quant_stack.research.backtest import run_backtest
from quant_stack.research.strategies.sma_cross import SmaCrossStrategy

setup_logging()

# ── 1. Data ────────────────────────────────────────────────────────────────────
data_cfg = DataConfig(
    symbols=["SPY", "QQQ", "IEF"],
    start=date(2018, 1, 1),
    end=date(2023, 12, 31),
)
provider = YahooProvider()
close = provider.fetch_close(data_cfg)

# ── 2. Strategy + Backtest ─────────────────────────────────────────────────────
strategy = SmaCrossStrategy(fast_window=20, slow_window=50)
bt_cfg = BacktestConfig(data=data_cfg, strategy_name="sma_cross_20_50")
result = run_backtest(strategy, close, bt_cfg)

print(f"\n=== Backtest: {result.strategy_name} ===")
print(f"  Total Return : {result.total_return:.2%}")
print(f"  CAGR         : {result.cagr:.2%}")
print(f"  Sharpe       : {result.sharpe_ratio:.3f}")
print(f"  Max Drawdown : {result.max_drawdown:.2%}")
print(f"  Trades       : {result.n_trades}")

# ── 3. Portfolio Optimisation ──────────────────────────────────────────────────
from quant_stack.portfolio.optimizer import optimize_portfolio

returns = simple_returns(close)
port_cfg = PortfolioConfig()
weights = optimize_portfolio(returns, port_cfg)

print("\n=== Optimal Weights ===")
for sym, w in sorted(weights.weights.items(), key=lambda x: -x[1]):
    print(f"  {sym:6s} {w:.2%}")

# ── 4. Report ─────────────────────────────────────────────────────────────────
from quant_stack.agent.reporter import Reporter

reporter = Reporter()
path = reporter.generate(backtest=result, weights=weights, title="SMA Cross — SPY/QQQ/IEF")
print(f"\nReport saved → {path}")
