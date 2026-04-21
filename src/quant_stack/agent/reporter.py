"""Report generator — produces a markdown research report from experiment results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from quant_stack.core.schemas import BacktestResult, PortfolioWeights


class Reporter:
    """Compile backtest + portfolio results into a markdown report file."""

    def __init__(self, reports_dir: str = "./reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        backtest: BacktestResult,
        weights: PortfolioWeights | None = None,
        analysis: str = "",
        title: str = "",
    ) -> Path:
        """Write a markdown report and return its path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = backtest.strategy_name.replace(" ", "_")
        path = self.reports_dir / f"{ts}_{slug}.md"

        lines: list[str] = [
            f"# {title or f'Backtest Report — {backtest.strategy_name}'}",
            f"\n_Generated: {datetime.now().isoformat(timespec='seconds')}_\n",
            "## Performance Summary\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Strategy | `{backtest.strategy_name}` |",
            f"| Total Return | {backtest.total_return:.2%} |",
            f"| CAGR | {backtest.cagr:.2%} |",
            f"| Sharpe Ratio | {backtest.sharpe_ratio:.3f} |",
            f"| Max Drawdown | {backtest.max_drawdown:.2%} |",
            f"| Trades | {backtest.n_trades} |",
        ]

        if weights:
            lines += [
                "\n## Portfolio Allocation\n",
                "| Symbol | Weight |",
                "|--------|--------|",
                *[f"| {sym} | {w:.2%} |" for sym, w in sorted(weights.weights.items())],
            ]
            if weights.expected_return is not None:
                lines += [
                    f"\nExpected Return: **{weights.expected_return:.2%}**  ",
                    f"Expected Volatility: **{weights.expected_volatility:.2%}**  ",
                    f"Sharpe: **{weights.sharpe_ratio:.3f}**",
                ]

        if analysis:
            lines += ["\n## Agent Analysis\n", analysis]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Report written → {path}")
        return path
