"""Load current portfolio positions from local files (no broker integration).

Supports two formats:

JSON (preferred):
  {
    "nav": 100000.0,
    "positions": {"QQQ": 0.3333, "XLI": 0.3333, "GDX": 0.3334},
    "cash_fraction": 0.0001,
    "source": "manual",
    "as_of": "2026-04-27T09:00:00"   (optional; defaults to now)
  }

CSV:
  symbol,weight
  QQQ,0.3333
  XLI,0.3333
  GDX,0.3334
  (pass nav separately; cash_fraction inferred as 1 - sum(weights))

Validation
----------
- Symbols are uppercased automatically.
- Weights must be positive floats.
- cash_fraction: if omitted in JSON, inferred as max(0, 1 - sum(weights)).
- The sum of positions + cash_fraction may slightly exceed 1.0 due to rounding;
  RebalanceService.run() checks for gross violations (> 1.05).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from quant_stack.execution.domain import PositionSnapshot


def load_positions_json(path: str | Path) -> PositionSnapshot:
    """Load a PositionSnapshot from a JSON file.

    Args:
        path: Path to a JSON file with the schema described in this module.

    Returns:
        PositionSnapshot ready for RebalanceService.run().

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError / ValueError: If required fields are missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Positions file not found: {path}\n"
            "Create it from the template at data/current_positions.json"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    nav = float(data["nav"])
    raw_positions = data.get("positions", {})
    positions: dict[str, float] = {
        k.upper(): float(v)
        for k, v in raw_positions.items()
        if not k.startswith("_") and float(v) > 0
    }

    if "cash_fraction" in data:
        cash_fraction = float(data["cash_fraction"])
    else:
        cash_fraction = max(0.0, 1.0 - sum(positions.values()))

    as_of_str = data.get("as_of")
    timestamp = datetime.fromisoformat(as_of_str) if as_of_str else datetime.now()
    source = data.get("source", "manual")

    return PositionSnapshot(
        timestamp=timestamp,
        nav=nav,
        positions=positions,
        cash_fraction=cash_fraction,
        source=source,
    )


def load_positions_csv(path: str | Path, nav: float) -> PositionSnapshot:
    """Load a PositionSnapshot from a two-column CSV (symbol, weight).

    Args:
        path: Path to CSV with columns [symbol, weight].
        nav: Total portfolio value in account currency.

    Returns:
        PositionSnapshot. cash_fraction = max(0, 1 - sum(weights)).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Positions CSV not found: {path}")

    positions: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            symbol = row["symbol"].strip().upper()
            if symbol in ("CASH", "") or symbol.startswith("_"):
                continue
            w = float(row["weight"])
            if w > 0:
                positions[symbol] = w

    cash_fraction = max(0.0, 1.0 - sum(positions.values()))
    return PositionSnapshot(
        timestamp=datetime.now(),
        nav=nav,
        positions=positions,
        cash_fraction=cash_fraction,
        source="manual",
    )
