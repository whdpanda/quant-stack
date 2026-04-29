"""Report and experiment comparison tools.

  - generate_report_tool    : ExperimentRecord → markdown report via Reporter
  - compare_experiments_tool: compare metrics across multiple ExperimentRecords
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_stack.agent.tools._context import ToolContext


# ── Tool implementations ──────────────────────────────────────────────────────

def generate_report(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Load an ExperimentRecord and write a markdown report.

    The report is written next to the record file.
    """
    record_path_str = inputs.get("record_path") or ctx.last_record_path
    if not record_path_str:
        return {
            "status": "error",
            "error": (
                "'record_path' is required. Pass the path returned by "
                "run_research_backtest_tool, or set ctx.last_record_path."
            ),
        }

    record_path = Path(record_path_str)
    if not record_path.exists():
        return {"status": "error", "error": f"Record not found: {record_path}"}

    try:
        from quant_stack.agent.experiment_tracker import load_record
        from quant_stack.agent.reporter import Reporter

        record = load_record(record_path)
        title = inputs.get("title", "")
        reporter = Reporter(reports_dir=str(record_path.parent))
        report_path = reporter.generate_from_record(record, title=title)
    except Exception as e:
        return {"status": "error", "error": f"Report generation failed: {e}"}

    return {
        "status": "ok",
        "record_path": str(record_path),
        "report_path": str(report_path),
    }


def compare_experiments(inputs: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Load multiple ExperimentRecords and compare performance metrics.

    Higher is better for cagr and sharpe_ratio; lower is better for max_drawdown.
    """
    record_paths: list[str] = inputs.get("record_paths", [])
    if len(record_paths) < 2:
        return {"status": "error", "error": "'record_paths' must contain at least 2 paths"}

    metrics: list[str] = inputs.get("metrics", ["cagr", "sharpe_ratio", "max_drawdown"])

    try:
        from quant_stack.agent.experiment_tracker import load_record
    except ImportError as e:
        return {"status": "error", "error": str(e)}

    rows: list[dict[str, Any]] = []
    for path_str in record_paths:
        path = Path(path_str)
        try:
            record = load_record(path)
        except Exception as e:
            rows.append({"label": str(path), "error": str(e)})
            continue

        label = record.description or record.experiment_id[:8]
        row: dict[str, Any] = {"label": label, "record_path": str(path)}
        if record.backtest_result:
            r = record.backtest_result
            for metric in metrics:
                val = getattr(r, metric, None)
                if val is not None:
                    row[metric] = round(float(val), 4)
        rows.append(row)

    # Determine winner per metric
    higher_better = {"cagr", "sharpe_ratio", "total_return", "sortino_ratio"}
    winners: dict[str, str] = {}
    for metric in metrics:
        best_label: str | None = None
        best_val: float | None = None
        prefer_high = metric in higher_better
        for row in rows:
            val = row.get(metric)
            if val is None:
                continue
            if best_val is None:
                best_val, best_label = val, row["label"]
            elif prefer_high and val > best_val:
                best_val, best_label = val, row["label"]
            elif not prefer_high and val < best_val:
                best_val, best_label = val, row["label"]
        if best_label:
            winners[metric] = best_label

    return {
        "status": "ok",
        "metrics": metrics,
        "table": rows,
        "winners": winners,
    }


# ── Anthropic tool schemas ────────────────────────────────────────────────────

REPORT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "generate_report_tool",
        "description": (
            "Load an ExperimentRecord JSON file and generate a markdown research report. "
            "The report is written to the same directory as the record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record_path": {
                    "type": "string",
                    "description": "Path to an ExperimentRecord JSON file",
                },
                "title": {
                    "type": "string",
                    "description": "Optional report title override",
                },
            },
            "required": [],
        },
    },
    {
        "name": "compare_experiments_tool",
        "description": (
            "Load two or more ExperimentRecord files and compare performance metrics. "
            "Returns a table and declares a winner per metric."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ExperimentRecord JSON paths to compare (min 2)",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compare (default: cagr, sharpe_ratio, max_drawdown)",
                },
            },
            "required": ["record_paths"],
        },
    },
]
