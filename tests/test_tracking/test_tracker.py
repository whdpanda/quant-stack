"""Tests for ExperimentTracker."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from quant_stack.core.schemas import ExperimentRecord
from quant_stack.tracking.tracker import ExperimentTracker


class TestExperimentTrackerSave:
    def test_creates_experiment_directory(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        exp_dir = tracker.save(full_record)
        assert exp_dir.exists()
        assert exp_dir.is_dir()

    def test_writes_record_json(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        exp_dir = tracker.save(full_record)
        assert (exp_dir / "record.json").exists()

    def test_writes_report_md(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        exp_dir = tracker.save(full_record)
        assert (exp_dir / "report.md").exists()

    def test_no_report_when_disabled(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        exp_dir = tracker.save(full_record, generate_report=False)
        assert not (exp_dir / "report.md").exists()

    def test_report_registered_in_artifact_paths(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        assert "report" in full_record.artifact_paths

    def test_dir_name_contains_strategy(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        exp_dir = tracker.save(full_record)
        assert "test_strategy" in exp_dir.name

    def test_dir_name_contains_short_id(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        exp_dir = tracker.save(full_record)
        short_id = full_record.experiment_id[:8]
        assert short_id in exp_dir.name

    def test_returns_path(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        result = tracker.save(full_record)
        assert isinstance(result, Path)


class TestExperimentTrackerRegistry:
    def test_registry_created(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        assert (tmp_path / "registry.json").exists()

    def test_registry_contains_entry(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        entries = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
        assert len(entries) == 1
        assert entries[0]["experiment_id"] == full_record.experiment_id

    def test_registry_contains_metrics(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        entries = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
        m = entries[0]["metrics"]
        assert abs(m["total_return"] - 0.42) < 1e-6
        assert abs(m["sharpe_ratio"] - 1.25) < 1e-6

    def test_multiple_saves_append_to_registry(self, full_record, minimal_result, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        # First experiment
        tracker.save(full_record)
        # Second distinct experiment
        record2 = ExperimentRecord(
            description="second",
            backtest_result=minimal_result,
        )
        tracker.save(record2)
        entries = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
        assert len(entries) == 2

    def test_resave_updates_not_duplicates(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        tracker.save(full_record)   # same ID, save again
        entries = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
        assert len(entries) == 1    # deduped by experiment_id

    def test_registry_has_correct_strategy_name(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        entries = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
        assert entries[0]["strategy_name"] == "test_strategy"

    def test_registry_has_tags(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        entries = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
        assert "test" in entries[0]["tags"]


class TestExperimentTrackerLoad:
    def test_load_roundtrip(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        loaded = tracker.load(full_record.experiment_id)
        assert loaded.experiment_id == full_record.experiment_id
        assert loaded.description == full_record.description

    def test_load_backtest_result(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        loaded = tracker.load(full_record.experiment_id)
        assert loaded.backtest_result is not None
        assert abs(loaded.backtest_result.total_return - 0.42) < 1e-9

    def test_load_portfolio_weights(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        loaded = tracker.load(full_record.experiment_id)
        assert loaded.portfolio_weights is not None
        assert abs(loaded.portfolio_weights.weights["SPY"] - 0.60) < 1e-9

    def test_load_strategy_params(self, full_record, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.save(full_record)
        loaded = tracker.load(full_record.experiment_id)
        assert loaded.strategy_params["top_n"] == 2

    def test_load_unknown_id_raises(self, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            tracker.load("nonexistent-id")


class TestListExperiments:
    def _save_n(self, n: int, tmp_path: Path) -> tuple[ExperimentTracker, list[ExperimentRecord]]:
        import time
        from quant_stack.core.schemas import BacktestResult
        tracker = ExperimentTracker(tmp_path)
        records = []
        for i in range(n):
            r = BacktestResult(
                strategy_name=f"strategy_{i}",
                total_return=i * 0.1,
                cagr=0.05,
                sharpe_ratio=1.0 + i * 0.1,
                max_drawdown=0.1,
                n_trades=10,
            )
            rec = ExperimentRecord(
                backtest_result=r,
                tags=["tag_a"] if i % 2 == 0 else ["tag_b"],
            )
            tracker.save(rec, generate_report=False)
            records.append(rec)
        return tracker, records

    def test_list_returns_all(self, tmp_path) -> None:
        tracker, records = self._save_n(3, tmp_path)
        entries = tracker.list_experiments()
        assert len(entries) == 3

    def test_list_newest_first(self, tmp_path) -> None:
        tracker, records = self._save_n(3, tmp_path)
        entries = tracker.list_experiments()
        # created_at strings sort lexicographically; newest should be first
        created = [e["created_at"] for e in entries]
        assert created == sorted(created, reverse=True)

    def test_filter_by_strategy_name(self, tmp_path) -> None:
        tracker, records = self._save_n(3, tmp_path)
        entries = tracker.list_experiments(strategy_name="strategy_1")
        assert len(entries) == 1
        assert entries[0]["strategy_name"] == "strategy_1"

    def test_filter_by_tag(self, tmp_path) -> None:
        tracker, records = self._save_n(4, tmp_path)
        entries = tracker.list_experiments(tag="tag_a")
        # indices 0 and 2 have tag_a
        assert len(entries) == 2
        assert all("tag_a" in e["tags"] for e in entries)

    def test_limit(self, tmp_path) -> None:
        tracker, records = self._save_n(5, tmp_path)
        entries = tracker.list_experiments(limit=2)
        assert len(entries) == 2

    def test_empty_registry_returns_empty_list(self, tmp_path) -> None:
        tracker = ExperimentTracker(tmp_path)
        assert tracker.list_experiments() == []
