"""ReportGenerator — produces a Markdown research report from an ExperimentRecord."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from quant_stack.core.schemas import BacktestResult, ExperimentRecord, PortfolioWeights


class ReportGenerator:
    """Convert an ExperimentRecord into a structured Markdown report.

    Usage::

        md = ReportGenerator().generate(record)         # str
        ReportGenerator().write(record, Path("out.md")) # writes file
    """

    def generate(self, record: ExperimentRecord) -> str:
        """Return the full report as a Markdown string."""
        sections = [
            self._header(record),
            self._data_scope(record),
            self._strategy_params(record),
            self._performance(record),
            self._portfolio_weights(record),
            self._artifacts(record),
            self._notes(record),
            self._footer(),
        ]
        return "\n\n".join(s for s in sections if s)

    def write(self, record: ExperimentRecord, path: Path) -> Path:
        """Write the report to *path* and return it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate(record), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Section builders (each returns a str or "" to be skipped)

    def _header(self, record: ExperimentRecord) -> str:
        r = record.backtest_result
        name = r.strategy_name if r else "Experiment"
        lines = [
            f"# {name} - Backtest Report",
            "",
            f"**Experiment ID**: `{record.experiment_id}`  ",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        ]
        if record.description:
            lines.append(f"**Description**: {record.description}  ")
        if record.tags:
            lines.append(f"**Tags**: {', '.join(f'`{t}`' for t in record.tags)}")
        return "\n".join(lines)

    def _data_scope(self, record: ExperimentRecord) -> str:
        if not record.symbols and not record.period_start:
            return ""
        lines = ["---", "", "## Data Scope", "", "| Field | Value |", "|-------|-------|"]
        if record.symbols:
            lines.append(f"| Universe | {', '.join(record.symbols)} |")
        if record.period_start and record.period_end:
            n_days = (record.period_end - record.period_start).days
            n_years = n_days / 365.25
            lines.append(f"| Period | {record.period_start} to {record.period_end} |")
            lines.append(f"| Duration | {n_years:.1f} years ({n_days:,} days) |")
        elif record.period_start:
            lines.append(f"| Period Start | {record.period_start} |")
        return "\n".join(lines)

    def _strategy_params(self, record: ExperimentRecord) -> str:
        params = record.strategy_params
        if not params:
            return ""
        lines = [
            "---", "", "## Strategy Parameters", "",
            "| Parameter | Value |", "|-----------|-------|",
        ]
        for k, v in params.items():
            lines.append(f"| `{k}` | `{v}` |")
        return "\n".join(lines)

    def _performance(self, record: ExperimentRecord) -> str:
        r = record.backtest_result
        if r is None:
            return ""

        has_bm = r.benchmark_return is not None

        def _row(label: str, strat, bm=None, fmt: str = ".2%") -> str:
            sv = f"{strat:{fmt}}" if strat is not None else "N/A"
            if has_bm:
                bv = f"{bm:{fmt}}" if bm is not None else "N/A"
                return f"| {label} | {sv} | {bv} |"
            return f"| {label} | {sv} |"

        sep = "|--------|----------|" + ("---------:|" if has_bm else "")
        hdr = "| Metric | Strategy |" + (" Benchmark |" if has_bm else "")

        lines = ["---", "", "## Performance Metrics", "", hdr, sep]
        lines.append(_row("Total Return",     r.total_return,      r.benchmark_return))
        lines.append(_row("CAGR",             r.cagr))
        if r.annual_volatility is not None:
            lines.append(_row("Annual Volatility", r.annual_volatility))
        lines.append(_row("Sharpe Ratio",     r.sharpe_ratio,      fmt=".3f"))
        if r.sortino_ratio is not None:
            lines.append(_row("Sortino Ratio", r.sortino_ratio,    fmt=".3f"))
        lines.append(_row("Max Drawdown",     r.max_drawdown))
        if r.annual_turnover is not None:
            lines.append(_row("Annual Turnover", r.annual_turnover))
        lines.append(
            f"| Trades | {r.n_trades:,} |" + (" N/A |" if has_bm else "")
        )
        if r.commission_paid:
            lines.append(
                f"| Commission Paid | ${r.commission_paid:,.0f} |" + (" N/A |" if has_bm else "")
            )

        if has_bm and r.excess_return is not None:
            sign = "+" if r.excess_return >= 0 else ""
            lines += ["", f"> **Excess Return vs Benchmark**: {sign}{r.excess_return:.2%}"]

        return "\n".join(lines)

    def _portfolio_weights(self, record: ExperimentRecord) -> str:
        pw = record.portfolio_weights
        if pw is None or not pw.weights:
            return ""
        lines = ["---", "", "## Portfolio Weights (Last Rebalance)", ""]
        if pw.method:
            lines.append(f"Method: `{pw.method}`  ")
        if pw.rebalance_date:
            lines.append(f"Rebalance Date: `{pw.rebalance_date}`  ")
        lines += ["", "| Symbol | Weight |", "|--------|--------|"]
        for sym, w in sorted(pw.weights.items(), key=lambda x: -x[1]):
            lines.append(f"| {sym} | {w:.2%} |")
        if pw.expected_return is not None and pw.expected_volatility is not None:
            lines.append(
                f"\nExpected Return: **{pw.expected_return:.2%}** | "
                f"Expected Volatility: **{pw.expected_volatility:.2%}**"
            )
        return "\n".join(lines)

    def _artifacts(self, record: ExperimentRecord) -> str:
        if not record.artifact_paths:
            return ""
        lines = ["---", "", "## Artifacts", ""]
        for name, path in record.artifact_paths.items():
            lines.append(f"- **{name}**: `{path}`")
        return "\n".join(lines)

    def _notes(self, record: ExperimentRecord) -> str:
        # agent_analysis takes priority; fall back to notes
        text = record.agent_analysis or record.notes
        if not text:
            return ""
        heading = "## Agent Analysis" if record.agent_analysis else "## Notes"
        return "\n".join(["---", "", heading, "", text])

    def _footer(self) -> str:
        return (
            "---\n\n"
            f"_Generated by quant-stack ExperimentTracker"
            f" ({datetime.now().strftime('%Y-%m-%d')})_"
        )
