"""Report generator — produces a markdown research report from experiment results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from quant_stack.core.schemas import BacktestResult, ExperimentRecord, PortfolioWeights


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
        logger.info(f"Report written -> {path}")
        return path

    # ------------------------------------------------------------------

    def generate_from_record(
        self,
        record: ExperimentRecord,
        title: str = "",
    ) -> Path:
        """Write a markdown report from an ExperimentRecord and return its path.

        All sections are optional — missing data is silently skipped so that
        partial records (e.g. no backtest_result yet) still produce valid output.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = (
            record.backtest_result.strategy_name
            if record.backtest_result
            else "experiment"
        )
        slug = strategy_name.replace(" ", "_")
        path = self.reports_dir / f"{ts}_{slug}.md"

        sections: list[str] = []

        # ── Title ──────────────────────────────────────────────────────
        heading = title or f"Research Report: {strategy_name}"
        sections.append(
            f"# {heading}\n\n"
            f"_Generated: {datetime.now().isoformat(timespec='seconds')}_"
        )

        # ── Experiment Metadata ────────────────────────────────────────
        meta_lines = [
            "## Experiment Metadata\n",
            "| Field | Value |",
            "|-------|-------|",
            f"| Experiment ID | `{record.experiment_id}` |",
            f"| Created At | {record.created_at.isoformat(timespec='seconds')} |",
        ]
        if record.description:
            meta_lines.append(f"| Description | {record.description} |")
        if record.symbols:
            meta_lines.append(f"| Symbols | {', '.join(record.symbols)} |")
        if record.period_start and record.period_end:
            meta_lines.append(
                f"| Period | {record.period_start} to {record.period_end} |"
            )
        elif record.period_start:
            meta_lines.append(f"| Period Start | {record.period_start} |")
        if record.tags:
            meta_lines.append(
                f"| Tags | {', '.join(f'`{t}`' for t in record.tags)} |"
            )
        if record.notes:
            meta_lines += ["", f"> {record.notes}"]
        sections.append("\n".join(meta_lines))

        # ── Strategy Parameters ────────────────────────────────────────
        if record.strategy_params:
            param_lines = [
                "## Strategy Parameters\n",
                "| Parameter | Value |",
                "|-----------|-------|",
            ]
            for k, v in record.strategy_params.items():
                param_lines.append(f"| `{k}` | `{v}` |")
            sections.append("\n".join(param_lines))

        # ── Configuration Snapshot (data + backtest sections only) ─────
        snap = record.config_snapshot
        if snap:
            snap_lines = ["## Configuration Snapshot\n"]
            for section_key, section_label in [("data", "Data Config"), ("backtest", "Backtest Config")]:
                sub = snap.get(section_key)
                if not sub or not isinstance(sub, dict):
                    continue
                snap_lines += [
                    f"### {section_label}\n",
                    "| Field | Value |",
                    "|-------|-------|",
                ]
                for k, v in sub.items():
                    snap_lines.append(f"| `{k}` | `{v}` |")
                snap_lines.append("")
            if len(snap_lines) > 1:  # at least one sub-section was written
                sections.append("\n".join(snap_lines))

        # ── Backtest Result ────────────────────────────────────────────
        r = record.backtest_result
        if r:
            result_lines = [
                "## Backtest Result\n",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Strategy | `{r.strategy_name}` |",
                f"| Total Return | {r.total_return:.2%} |",
                f"| CAGR | {r.cagr:.2%} |",
                f"| Sharpe Ratio | {r.sharpe_ratio:.3f} |",
                f"| Max Drawdown | {r.max_drawdown:.2%} |",
                f"| Trades | {r.n_trades:,} |",
            ]
            if r.sortino_ratio is not None:
                result_lines.append(f"| Sortino Ratio | {r.sortino_ratio:.3f} |")
            if r.annual_volatility is not None:
                result_lines.append(f"| Annual Volatility | {r.annual_volatility:.2%} |")
            if r.annual_turnover is not None:
                result_lines.append(f"| Annual Turnover | {r.annual_turnover:.2%} |")
            if r.commission_paid:
                result_lines.append(f"| Commission Paid | ${r.commission_paid:,.0f} |")
            if r.benchmark_return is not None:
                result_lines.append(f"| Benchmark Return | {r.benchmark_return:.2%} |")
            if r.excess_return is not None:
                sign = "+" if r.excess_return >= 0 else ""
                result_lines.append(f"| Excess Return | {sign}{r.excess_return:.2%} |")
            sections.append("\n".join(result_lines))

        # ── Portfolio Allocation ───────────────────────────────────────
        pw = record.portfolio_weights
        if pw and pw.weights:
            alloc_lines = [
                "## Portfolio Allocation\n",
                "| Symbol | Weight |",
                "|--------|--------|",
                *[
                    f"| {sym} | {w:.2%} |"
                    for sym, w in sorted(pw.weights.items(), key=lambda x: -x[1])
                ],
            ]
            if pw.method:
                alloc_lines.insert(1, f"Method: `{pw.method}`  \n")
            if pw.expected_return is not None:
                alloc_lines += [
                    "",
                    f"Expected Return: **{pw.expected_return:.2%}**  ",
                    f"Expected Volatility: **{pw.expected_volatility:.2%}**  ",
                    f"Sharpe: **{pw.sharpe_ratio:.3f}**",
                ]
            sections.append("\n".join(alloc_lines))

        # ── Agent Analysis ─────────────────────────────────────────────
        if record.agent_analysis:
            sections.append(f"## Agent Analysis\n\n{record.agent_analysis}")

        # ── Artifacts ─────────────────────────────────────────────────
        if record.artifact_paths:
            artifact_lines = ["## Artifacts\n"]
            for name, p in record.artifact_paths.items():
                artifact_lines.append(f"- **{name}**: `{p}`")
            sections.append("\n".join(artifact_lines))

        # ── Footer ────────────────────────────────────────────────────
        sections.append(
            f"_Report generated by quant-stack Reporter "
            f"({datetime.now().strftime('%Y-%m-%d')})_"
        )

        path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
        logger.info(f"Report written -> {path}")
        return path
