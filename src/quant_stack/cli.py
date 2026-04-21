"""CLI entry point — `quant <command>`.

Usage examples:
    quant backtest --symbols SPY QQQ --start 2020-01-01 --end 2023-12-31
    quant optimise --symbols SPY QQQ IEF GLD --start 2018-01-01 --end 2023-12-31
    quant info
"""

from __future__ import annotations

from datetime import date

import typer
from rich.console import Console
from rich.table import Table

from quant_stack import __version__
from quant_stack.core.logging import setup_logging

app = typer.Typer(name="quant", help="Agent-Assisted Quant Research Stack", add_completion=False)
console = Console()


@app.callback()
def _setup() -> None:
    setup_logging()


@app.command()
def info() -> None:
    """Print version and layer availability."""
    table = Table(title=f"quant-stack v{__version__}", show_lines=True)
    table.add_column("Layer")
    table.add_column("Status")

    def _check(mod: str) -> str:
        try:
            __import__(mod)
            return "[green]available[/green]"
        except ImportError:
            return "[yellow]not installed[/yellow]"

    table.add_row("vectorbt (research)", _check("vectorbt"))
    table.add_row("PyPortfolioOpt (portfolio)", _check("pypfopt"))
    table.add_row("anthropic (agent)", _check("anthropic"))
    table.add_row("polars (data)", _check("polars"))
    console.print(table)


@app.command()
def backtest(
    symbols: str = typer.Option(..., "--symbols", "-s", help="Comma-separated tickers, e.g. SPY,QQQ"),
    start: str = typer.Option("2020-01-01", "--start", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(date.today().isoformat(), "--end", help="End date YYYY-MM-DD"),
    fast: int = typer.Option(20, "--fast", help="Fast SMA window"),
    slow: int = typer.Option(50, "--slow", help="Slow SMA window"),
    cash: float = typer.Option(100_000.0, "--cash", help="Initial cash"),
    no_agent: bool = typer.Option(False, "--no-agent", help="Skip agent analysis"),
) -> None:
    """Run an SMA crossover backtest and optionally get AI commentary."""
    from quant_stack.core.schemas import BacktestConfig, DataConfig
    from quant_stack.data.providers.yahoo import YahooProvider
    from quant_stack.research.backtest import run_backtest
    from quant_stack.research.strategies.sma_cross import SmaCrossStrategy

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    cfg = BacktestConfig(
        data=DataConfig(
            symbols=symbol_list,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
        ),
        strategy_name="sma_cross",
        initial_cash=cash,
    )

    provider = YahooProvider()
    close = provider.fetch_close(cfg.data)

    strategy = SmaCrossStrategy(fast_window=fast, slow_window=slow)
    result = run_backtest(strategy, close, cfg)

    console.print(f"\n[bold]Backtest: {result.strategy_name}[/bold]")
    console.print(f"  Total Return : {result.total_return:.2%}")
    console.print(f"  CAGR         : {result.cagr:.2%}")
    console.print(f"  Sharpe       : {result.sharpe_ratio:.3f}")
    console.print(f"  Max Drawdown : {result.max_drawdown:.2%}")
    console.print(f"  Trades       : {result.n_trades}")

    if not no_agent:
        try:
            from quant_stack.agent.researcher import Researcher
            console.print("\n[dim]Asking agent for analysis…[/dim]")
            analysis = Researcher().analyse_backtest(result)
            console.print(f"\n[bold]Agent Analysis[/bold]\n{analysis}")
        except Exception as exc:
            console.print(f"[yellow]Agent unavailable: {exc}[/yellow]")


@app.command()
def optimise(
    symbols: str = typer.Option(..., "--symbols", "-s", help="Comma-separated tickers, e.g. SPY,QQQ,IEF"),
    start: str = typer.Option("2018-01-01", "--start"),
    end: str = typer.Option(date.today().isoformat(), "--end"),
    method: str = typer.Option("max_sharpe", "--method"),
) -> None:
    """Compute optimal portfolio weights with PyPortfolioOpt."""
    from quant_stack.core.schemas import DataConfig, PortfolioConfig, PortfolioMethod
    from quant_stack.data.providers.yahoo import YahooProvider
    from quant_stack.data.transforms import simple_returns
    from quant_stack.portfolio.optimizer import optimize_portfolio

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    data_cfg = DataConfig(
        symbols=symbol_list,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
    )
    provider = YahooProvider()
    close = provider.fetch_close(data_cfg)
    returns = simple_returns(close)

    port_cfg = PortfolioConfig(method=PortfolioMethod(method))
    weights = optimize_portfolio(returns, port_cfg)

    console.print("\n[bold]Optimal Weights[/bold]")
    for sym, w in sorted(weights.weights.items(), key=lambda x: -x[1]):
        console.print(f"  {sym:8s} {w:.2%}")
    if weights.sharpe_ratio:
        console.print(f"\n  Sharpe: {weights.sharpe_ratio:.3f}")


if __name__ == "__main__":
    app()
