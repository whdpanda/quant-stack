"""Tests for quant_stack.agent.experiment_tracker module-level functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_stack.agent.experiment_tracker import load_record, load_records, save_record
from quant_stack.core.schemas import BacktestResult, ExperimentRecord


# ── save_record ────────────────────────────────────────────────────────────────


class TestSaveRecord:
    def test_returns_path(self, full_record, tmp_path) -> None:
        result = save_record(full_record, base_dir=tmp_path)
        assert isinstance(result, Path)

    def test_file_exists(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        assert path.exists()

    def test_file_is_json(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        assert path.suffix == ".json"

    def test_file_contains_valid_json(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "experiment_id" in data

    def test_filename_contains_strategy_name(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        assert "test_strategy" in path.name

    def test_filename_contains_symbols(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        assert "spy" in path.name
        assert "qqq" in path.name

    def test_creates_base_dir_if_missing(self, full_record, tmp_path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        save_record(full_record, base_dir=nested)
        assert nested.exists()

    def test_no_symbols_uses_fallback_name(self, tmp_path) -> None:
        record = ExperimentRecord(description="no symbols")
        path = save_record(record, base_dir=tmp_path)
        assert "no_symbols" in path.name

    def test_no_backtest_uses_unknown_strategy(self, tmp_path) -> None:
        record = ExperimentRecord(description="no backtest")
        path = save_record(record, base_dir=tmp_path)
        assert "unknown_strategy" in path.name

    def test_multiple_saves_create_separate_files(self, full_record, tmp_path) -> None:
        import time

        path1 = save_record(full_record, base_dir=tmp_path)
        time.sleep(0.01)
        # Create a new record (new experiment_id) so filenames don't clash
        from quant_stack.core.schemas import ExperimentRecord as ER
        record2 = ER(description="second")
        path2 = save_record(record2, base_dir=tmp_path)
        assert path1 != path2


# ── load_record ────────────────────────────────────────────────────────────────


class TestLoadRecord:
    def test_roundtrip_experiment_id(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        loaded = load_record(path)
        assert loaded.experiment_id == full_record.experiment_id

    def test_roundtrip_description(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        loaded = load_record(path)
        assert loaded.description == full_record.description

    def test_roundtrip_backtest_result(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        loaded = load_record(path)
        assert loaded.backtest_result is not None
        assert abs(loaded.backtest_result.sharpe_ratio - 1.0) < 1e-9

    def test_roundtrip_portfolio_weights(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        loaded = load_record(path)
        assert loaded.portfolio_weights is not None
        assert abs(loaded.portfolio_weights.weights["SPY"] - 0.60) < 1e-9

    def test_roundtrip_strategy_params(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        loaded = load_record(path)
        assert loaded.strategy_params["top_n"] == 2

    def test_roundtrip_tags(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        loaded = load_record(path)
        assert "test" in loaded.tags

    def test_missing_file_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_record(tmp_path / "nonexistent.json")

    def test_accepts_string_path(self, full_record, tmp_path) -> None:
        path = save_record(full_record, base_dir=tmp_path)
        loaded = load_record(str(path))
        assert loaded.experiment_id == full_record.experiment_id


# ── load_records ───────────────────────────────────────────────────────────────


class TestLoadRecords:
    def test_empty_dir_returns_empty_list(self, tmp_path) -> None:
        result = load_records(base_dir=tmp_path)
        assert result == []

    def test_missing_dir_returns_empty_list(self, tmp_path) -> None:
        result = load_records(base_dir=tmp_path / "nonexistent")
        assert result == []

    def test_returns_all_records(self, full_record, tmp_path) -> None:
        save_record(full_record, base_dir=tmp_path)
        record2 = ExperimentRecord(description="second")
        save_record(record2, base_dir=tmp_path)
        records = load_records(base_dir=tmp_path)
        assert len(records) == 2

    def test_returns_experiment_record_instances(self, full_record, tmp_path) -> None:
        save_record(full_record, base_dir=tmp_path)
        records = load_records(base_dir=tmp_path)
        assert all(isinstance(r, ExperimentRecord) for r in records)

    def test_skips_invalid_json_file(self, full_record, tmp_path) -> None:
        save_record(full_record, base_dir=tmp_path)
        (tmp_path / "corrupt.json").write_text("not-json", encoding="utf-8")
        records = load_records(base_dir=tmp_path)
        assert len(records) == 1  # corrupt file skipped, valid one loaded

    def test_ignores_non_json_files(self, full_record, tmp_path) -> None:
        save_record(full_record, base_dir=tmp_path)
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        records = load_records(base_dir=tmp_path)
        assert len(records) == 1

    def test_preserves_backtest_result_after_bulk_load(self, full_record, tmp_path) -> None:
        save_record(full_record, base_dir=tmp_path)
        records = load_records(base_dir=tmp_path)
        assert records[0].backtest_result is not None
        assert records[0].backtest_result.strategy_name == "test_strategy"
