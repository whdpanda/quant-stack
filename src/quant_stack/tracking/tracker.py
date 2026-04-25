"""ExperimentTracker — local-filesystem experiment registry.

Directory layout::

    {base_dir}/
        registry.json                           # flat index, all experiments
        20260422_143512_strategy_abc12345/      # one dir per experiment
            record.json                         # full ExperimentRecord (JSON)
            report.md                           # auto-generated Markdown report

The registry.json is a JSON array of lightweight summary dicts.  It is the
only file you need to browse experiments; the full record is in record.json.

Usage::

    tracker = ExperimentTracker()

    record = ExperimentRecord(
        description="trend filter + 63d momentum, top-2, monthly rebalancing",
        strategy_params={"top_n": 2, "momentum_window": 63, "rebalance_freq": "ME"},
        symbols=["SPY", "QQQ", "IWM"],
        period_start=date(2018, 1, 2),
        period_end=date(2023, 10, 2),
        backtest_result=result,
        tags=["momentum", "trend", "monthly"],
        notes="Outperforms in trending regimes; underperforms in choppy markets.",
    )

    exp_dir = tracker.save(record)
    print(f"Saved -> {exp_dir}")

    # Browse all experiments
    for e in tracker.list_experiments():
        print(e["strategy_name"], e["metrics"].get("sharpe_ratio"))

    # Re-load a specific record
    record2 = tracker.load(record.experiment_id)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from quant_stack.core.schemas import ExperimentRecord
from quant_stack.tracking.report import ReportGenerator


class ExperimentTracker:
    """Persist experiments to the local filesystem and maintain a searchable index."""

    _REGISTRY = "registry.json"

    def __init__(self, base_dir: str | Path = "./experiments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API

    def save(
        self,
        record: ExperimentRecord,
        generate_report: bool = True,
    ) -> Path:
        """Persist *record* to disk and return its experiment directory.

        Creates::
            {exp_dir}/record.json   — full serialised record
            {exp_dir}/report.md     — Markdown report (if generate_report=True)

        Also appends a summary entry to registry.json.

        Args:
            record: A completed ExperimentRecord.  Fill in backtest_result /
                    portfolio_weights / notes before calling this.
            generate_report: Write report.md alongside record.json.

        Returns:
            Path to the experiment directory.
        """
        exp_dir = self._exp_dir(record)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Persist record
        record.save(exp_dir / "record.json")

        # 2. Generate report and register it as an artifact
        if generate_report:
            report_path = exp_dir / "report.md"
            ReportGenerator().write(record, report_path)
            # Register in artifact_paths so the record reflects it
            record.artifact_paths.setdefault("report", "report.md")
            # Re-save to capture the updated artifact_paths
            record.save(exp_dir / "record.json")

        # 3. Update registry
        self._append_registry(record, exp_dir)

        logger.info(f"Experiment saved -> {exp_dir.resolve()}")
        return exp_dir

    def load(self, experiment_id: str) -> ExperimentRecord:
        """Load a previously saved record by experiment ID.

        Raises:
            KeyError: If the experiment ID is not found in the registry.
        """
        for entry in self._read_registry():
            if entry.get("experiment_id") == experiment_id:
                path = self.base_dir / entry["exp_dir"] / "record.json"
                return ExperimentRecord.load(path)
        raise KeyError(f"Experiment not found in registry: {experiment_id!r}")

    def list_experiments(
        self,
        strategy_name: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return registry summary entries, newest first.

        Args:
            strategy_name: Filter to a specific strategy name.
            tag: Filter to experiments that carry this tag.
            limit: Maximum number of results to return.
        """
        entries = sorted(
            self._read_registry(),
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )
        if strategy_name:
            entries = [e for e in entries if e.get("strategy_name") == strategy_name]
        if tag:
            entries = [e for e in entries if tag in e.get("tags", [])]
        if limit is not None:
            entries = entries[:limit]
        return entries

    # ------------------------------------------------------------------
    # Private helpers

    @property
    def _registry_path(self) -> Path:
        return self.base_dir / self._REGISTRY

    def _exp_dir(self, record: ExperimentRecord) -> Path:
        ts = record.created_at.strftime("%Y%m%d_%H%M%S")
        if record.backtest_result:
            raw_name = record.backtest_result.strategy_name
        else:
            raw_name = "experiment"
        safe = re.sub(r"[^\w]", "_", raw_name)[:30]
        short_id = record.experiment_id[:8]
        return self.base_dir / f"{ts}_{safe}_{short_id}"

    def _read_registry(self) -> list[dict]:
        if not self._registry_path.exists():
            return []
        try:
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _append_registry(self, record: ExperimentRecord, exp_dir: Path) -> None:
        entries = self._read_registry()
        # Avoid duplicates on re-save
        entries = [e for e in entries if e.get("experiment_id") != record.experiment_id]
        entries.append(self._summary(record, exp_dir))
        self._registry_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _summary(record: ExperimentRecord, exp_dir: Path) -> dict:
        r = record.backtest_result
        metrics: dict = {}
        if r:
            for key, val in {
                "total_return": r.total_return,
                "cagr": r.cagr,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "max_drawdown": r.max_drawdown,
                "annual_volatility": r.annual_volatility,
                "annual_turnover": r.annual_turnover,
                "benchmark_return": r.benchmark_return,
                "commission_paid": r.commission_paid,
                "n_trades": float(r.n_trades),
            }.items():
                if val is not None:
                    metrics[key] = round(val, 6)

        return {
            "experiment_id": record.experiment_id,
            "created_at": record.created_at.isoformat(timespec="seconds"),
            "strategy_name": r.strategy_name if r else "",
            "description": record.description,
            "tags": record.tags,
            "symbols": record.symbols,
            "period_start": str(record.period_start) if record.period_start else None,
            "period_end": str(record.period_end) if record.period_end else None,
            "exp_dir": exp_dir.name,
            "metrics": metrics,
        }
