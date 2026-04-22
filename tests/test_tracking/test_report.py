"""Tests for ReportGenerator."""

from __future__ import annotations

import pytest

from quant_stack.core.schemas import BacktestResult, ExperimentRecord
from quant_stack.tracking.report import ReportGenerator


class TestReportGenerator:
    def test_generate_returns_string(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_header_contains_strategy_name(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "test_strategy" in md

    def test_header_contains_experiment_id(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert full_record.experiment_id in md

    def test_header_contains_description(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "Unit test experiment" in md

    def test_header_contains_tags(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "`test`" in md
        assert "`momentum`" in md

    def test_data_scope_section(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "## Data Scope" in md
        assert "SPY" in md
        assert "2020-01-02" in md

    def test_strategy_params_section(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "## Strategy Parameters" in md
        assert "`top_n`" in md
        assert "`2`" in md

    def test_performance_section(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "## Performance Metrics" in md
        assert "42.00%" in md   # total_return
        assert "10.00%" in md   # cagr
        assert "1.250" in md    # sharpe

    def test_benchmark_column_when_present(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "Benchmark" in md
        assert "12.00%" in md   # benchmark_return
        assert "Excess Return" in md

    def test_no_benchmark_column_when_absent(self, minimal_result) -> None:
        record = ExperimentRecord(backtest_result=minimal_result)
        record.backtest_result = BacktestResult(
            strategy_name="no_bm",
            total_return=0.1,
            cagr=0.05,
            sharpe_ratio=0.8,
            max_drawdown=0.1,
            n_trades=10,
        )
        md = ReportGenerator().generate(record)
        assert "Benchmark" not in md
        assert "Excess Return" not in md

    def test_extended_metrics_included(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "Sortino Ratio" in md
        assert "Annual Volatility" in md
        assert "Annual Turnover" in md

    def test_portfolio_weights_section(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "## Portfolio Weights" in md
        assert "60.00%" in md
        assert "40.00%" in md

    def test_artifacts_section(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "## Artifacts" in md
        assert "weights_csv" in md
        assert "artifacts/weights.csv" in md

    def test_notes_section(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "## Notes" in md
        assert "This is a test note." in md

    def test_agent_analysis_overrides_notes(self, full_record) -> None:
        full_record.agent_analysis = "Agent says: very nice strategy."
        md = ReportGenerator().generate(full_record)
        assert "## Agent Analysis" in md
        assert "Agent says" in md
        # Notes should not appear separately when agent_analysis is set
        assert "## Notes" not in md

    def test_write_creates_file(self, full_record, tmp_path) -> None:
        out = tmp_path / "report.md"
        ReportGenerator().write(full_record, out)
        assert out.exists()
        assert out.stat().st_size > 0
        assert "test_strategy" in out.read_text(encoding="utf-8")

    def test_empty_record_does_not_crash(self) -> None:
        # A record with no backtest_result at all should still produce output
        record = ExperimentRecord(description="bare minimum")
        md = ReportGenerator().generate(record)
        assert isinstance(md, str)

    def test_footer_present(self, full_record) -> None:
        md = ReportGenerator().generate(full_record)
        assert "quant-stack ExperimentTracker" in md
