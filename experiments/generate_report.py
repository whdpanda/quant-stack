from __future__ import annotations

from pathlib import Path

from quant_stack.agent.experiment_tracker import load_records
from quant_stack.agent.reporter import Reporter


def main() -> None:
    records = load_records("experiments/records")
    if not records:
        raise RuntimeError("No experiment records found in experiments/records")

    reporter = Reporter(reports_dir="reports")

    latest_record = records[-1]
    report_path = reporter.generate_from_record(latest_record)

    print(f"Generated report: {report_path}")


if __name__ == "__main__":
    main()