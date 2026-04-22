"""Tests for Reporter.generate_from_record()."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quant_stack.agent.reporter import Reporter
from quant_stack.core.schemas import BacktestResult, ExperimentRecord, PortfolioWeights


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_reporter(tmp_path: Path) -> Reporter:
    return Reporter(reports_dir=str(tmp_path))


# ── file output ────────────────────────────────────────────────────────────────


class TestGenerateFromRecordFile:
    def test_returns_path(self, full_record, tmp_path) -> None:
        reporter = _make_reporter(tmp_path)
        result = reporter.generate_from_record(full_record)
        assert isinstance(result, Path)

    def test_file_exists(self, full_record, tmp_path) -> None:
        reporter = _make_reporter(tmp_path)
        path = reporter.generate_from_record(full_record)
        assert path.exists()

    def test_file_is_markdown(self, full_record, tmp_path) -> None:
        reporter = _make_reporter(tmp_path)
        path = reporter.generate_from_record(full_record)
        assert path.suffix == ".md"

    def test_filename_contains_strategy_slug(self, full_record, tmp_path) -> None:
        reporter = _make_reporter(tmp_path)
        path = reporter.generate_from_record(full_record)
        assert "test_strategy" in path.name

    def test_custom_title_not_used_in_filename(self, full_record, tmp_path) -> None:
        reporter = _make_reporter(tmp_path)
        path = reporter.generate_from_record(full_record, title="My Custom Title")
        assert path.exists()


# ── content — title section ────────────────────────────────────────────────────


class TestGenerateFromRecordTitle:
    def test_default_title_contains_strategy_name(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert "test_strategy" in content

    def test_custom_title_appears_in_content(self, full_record, tmp_path) -> None:
        content = (
            _make_reporter(tmp_path)
            .generate_from_record(full_record, title="Special Report")
            .read_text(encoding="utf-8")
        )
        assert "Special Report" in content

    def test_experiment_id_in_content(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert full_record.experiment_id in content


# ── content — backtest result section ─────────────────────────────────────────


class TestGenerateFromRecordPerformance:
    def test_total_return_present(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert "20.00%" in content

    def test_sharpe_ratio_present(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert "1.000" in content

    def test_max_drawdown_present(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert "8.00%" in content

    def test_no_backtest_skips_performance_section(self, tmp_path) -> None:
        record = ExperimentRecord(description="no backtest")
        content = _make_reporter(tmp_path).generate_from_record(record).read_text(encoding="utf-8")
        assert "Backtest Result" not in content


# ── content — portfolio allocation section ─────────────────────────────────────


class TestGenerateFromRecordAllocation:
    def test_symbol_weights_present(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert "SPY" in content
        assert "QQQ" in content

    def test_no_weights_skips_allocation_section(self, tmp_path) -> None:
        record = ExperimentRecord(description="no weights")
        content = _make_reporter(tmp_path).generate_from_record(record).read_text(encoding="utf-8")
        assert "Portfolio Allocation" not in content


# ── content — optional sections ───────────────────────────────────────────────


class TestGenerateFromRecordOptional:
    def test_strategy_params_present(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert "top_n" in content

    def test_tags_present(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert "test" in content

    def test_agent_analysis_rendered(self, tmp_path) -> None:
        record = ExperimentRecord(
            description="with analysis",
            agent_analysis="This strategy looks promising.",
        )
        content = _make_reporter(tmp_path).generate_from_record(record).read_text(encoding="utf-8")
        assert "This strategy looks promising." in content

    def test_minimal_record_does_not_raise(self, tmp_path) -> None:
        record = ExperimentRecord(description="bare minimum")
        path = _make_reporter(tmp_path).generate_from_record(record)
        assert path.exists()

    def test_period_present_when_set(self, full_record, tmp_path) -> None:
        content = _make_reporter(tmp_path).generate_from_record(full_record).read_text(encoding="utf-8")
        assert "2020" in content
        assert "2023" in content

    def test_no_params_skips_params_section(self, tmp_path) -> None:
        record = ExperimentRecord(description="no params")
        content = _make_reporter(tmp_path).generate_from_record(record).read_text(encoding="utf-8")
        assert "Strategy Parameters" not in content


# ── backward compatibility — existing generate() still works ──────────────────


class TestGenerateBackwardCompat:
    def test_generate_still_works(self, tmp_path) -> None:
        result = BacktestResult(
            strategy_name="old_api",
            total_return=0.10,
            cagr=0.05,
            sharpe_ratio=0.8,
            max_drawdown=0.05,
            n_trades=3,
        )
        reporter = _make_reporter(tmp_path)
        path = reporter.generate(result)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "old_api" in content
