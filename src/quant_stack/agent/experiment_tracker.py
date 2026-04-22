from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger

from quant_stack.core.schemas import ExperimentRecord


DEFAULT_RECORDS_DIR = Path("experiments/records")


def save_record(
    record: ExperimentRecord,
    base_dir: str | Path = DEFAULT_RECORDS_DIR,
) -> Path:
    """Save one experiment record as JSON and return the written path."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    ts = record.created_at.strftime("%Y%m%dT%H%M%S")
    symbol_part = "_".join(s.lower() for s in record.symbols) if record.symbols else "no_symbols"

    strategy_name = (
        record.backtest_result.strategy_name
        if record.backtest_result is not None
        else "unknown_strategy"
    )
    strategy_part = strategy_name.lower().replace(" ", "_")

    filename = f"{ts}_{symbol_part}_{strategy_part}.json"
    path = base_path / filename

    record.save(path)
    logger.info(f"Experiment record saved -> {path}")
    return path


def load_record(path: str | Path) -> ExperimentRecord:
    """Load one experiment record from JSON."""
    record_path = Path(path)
    if not record_path.exists():
        raise FileNotFoundError(f"Record file not found: {record_path}")

    record = ExperimentRecord.load(record_path)
    logger.info(f"Experiment record loaded <- {record_path}")
    return record


def load_records(base_dir: str | Path = DEFAULT_RECORDS_DIR) -> List[ExperimentRecord]:
    """Load all experiment records from a directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Records directory does not exist: {base_path}")
        return []

    records: list[ExperimentRecord] = []
    for path in sorted(base_path.glob("*.json")):
        try:
            records.append(ExperimentRecord.load(path))
        except Exception as exc:
            logger.warning(f"Skipping invalid record file {path}: {exc}")

    logger.info(f"Loaded {len(records)} experiment record(s) from {base_path}")
    return records