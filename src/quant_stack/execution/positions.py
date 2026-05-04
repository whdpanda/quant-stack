"""Load current portfolio positions from local files (no broker integration).

Two input schemas are supported — the loader detects automatically:

──────────────────────────────────────────────────────────────────────────────
NEW FORMAT (amount-driven, preferred)
──────────────────────────────────────────────────────────────────────────────
{
  "nav_usd":   4954.69,
  "cash_usd":  210.59,
  "source":    "manual",
  "as_of":     "2026-05-02T16:00:00",
  "positions": {
    "GDX": {
      "quantity":         15,
      "last_price_usd":   87.08,
      "market_value_usd": 1306.20
    },
    "IBB": { "quantity": 10, "last_price_usd": 167.27, "market_value_usd": 1672.70 },
    "XLE": { "quantity": 30, "last_price_usd":  58.84, "market_value_usd": 1765.20 }
  }
}

Source-of-truth hierarchy (new format):
  1. market_value_usd  →  weight = market_value_usd / nav_usd
  2. quantity × last_price_usd  →  cross-checked against market_value_usd (soft warn)
  3. weight field (if present)  →  treated as a human-entered label, cross-checked only

Validation (new format):
  • sum(market_value_usd) + cash_usd must be within $1.00 of nav_usd
  • If quantity and last_price_usd are both present:
      quantity × last_price_usd should be within $0.50 of market_value_usd

──────────────────────────────────────────────────────────────────────────────
LEGACY FORMAT (weight-fraction, backward-compatible)
──────────────────────────────────────────────────────────────────────────────
{
  "nav":           4954.69,
  "positions":     {"GDX": 0.26363, "IBB": 0.33760, "XLE": 0.35627},
  "cash_fraction": 0.042503,
  "source":        "manual",
  "as_of":         "2026-05-02T16:00:00"
}

──────────────────────────────────────────────────────────────────────────────
Detection logic:
  • "nav_usd" key  →  new format
  • "cash_usd" key  →  new format
  • positions values are dicts (not floats)  →  new format
  • everything else  →  legacy format

When both formats appear in the same file, new-format fields take precedence.

CSV:
  symbol,weight
  QQQ,0.3333
  (pass nav separately; cash_fraction inferred as 1 - sum(weights))
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from quant_stack.execution.domain import PositionSnapshot


# ── Format detection ──────────────────────────────────────────────────────────

def _detect_format(data: dict) -> str:
    """Return 'new' for amount-driven format, 'legacy' for weight-fraction format."""
    if "nav_usd" in data or "cash_usd" in data:
        return "new"
    positions = data.get("positions", {})
    non_comment = {k: v for k, v in positions.items() if not k.startswith("_")}
    if non_comment and isinstance(next(iter(non_comment.values())), dict):
        return "new"
    return "legacy"


# ── New format parser ─────────────────────────────────────────────────────────

def _parse_new_format(data: dict) -> PositionSnapshot:
    """Parse amount-driven format.  Weights are derived from market_value_usd."""
    nav = float(data["nav_usd"])
    cash_usd = float(data["cash_usd"])
    cash_fraction = cash_usd / nav

    positions: dict[str, float] = {}
    position_metadata: dict[str, dict] = {}

    raw_positions = data.get("positions", {})
    total_mv = 0.0

    for raw_sym, pos_data in raw_positions.items():
        if raw_sym.startswith("_"):
            continue
        sym = raw_sym.upper()
        mv = float(pos_data["market_value_usd"])
        if mv <= 0:
            continue

        # Derive weight from exact market value — this is the precision improvement
        positions[sym] = mv / nav
        total_mv += mv

        meta: dict = {"market_value_usd": round(mv, 2)}
        if "quantity" in pos_data:
            meta["quantity"] = int(pos_data["quantity"])
        if "last_price_usd" in pos_data:
            meta["last_price_usd"] = float(pos_data["last_price_usd"])

        # Cross-check: quantity × price ≈ market_value_usd
        if "quantity" in pos_data and "last_price_usd" in pos_data:
            implied_mv = float(pos_data["quantity"]) * float(pos_data["last_price_usd"])
            diff = abs(implied_mv - mv)
            if diff > 0.50:
                logger.warning(
                    f"{sym}: quantity×price=${implied_mv:,.2f} ≠ market_value_usd=${mv:,.2f}"
                    f" (diff ${diff:.2f}). Using market_value_usd as authoritative."
                )

        # Cross-check: stated weight ≈ derived weight
        if "weight" in pos_data:
            stated_w = float(pos_data["weight"])
            derived_w = mv / nav
            if abs(stated_w - derived_w) > 0.0005:
                logger.warning(
                    f"{sym}: stated weight={stated_w:.6f} ≠ derived weight={derived_w:.6f}"
                    f" (diff {abs(stated_w - derived_w):.6f}). Using derived weight."
                )

        position_metadata[sym] = meta

    # Validate: sum(market_value) + cash ≈ nav
    discrepancy = abs(total_mv + cash_usd - nav)
    if discrepancy > 1.00:
        logger.warning(
            f"current_positions.json balance check: "
            f"Σ(market_value_usd)=${total_mv:,.2f} + cash_usd=${cash_usd:,.2f}"
            f" = ${total_mv + cash_usd:,.2f}  ≠  nav_usd=${nav:,.2f}"
            f"  (diff ${discrepancy:,.2f}). Using nav_usd as authoritative NAV."
        )

    as_of_str = data.get("as_of")
    timestamp = datetime.fromisoformat(as_of_str) if as_of_str else datetime.now()

    return PositionSnapshot(
        timestamp=timestamp,
        nav=nav,
        positions=positions,
        cash_fraction=cash_fraction,
        source=data.get("source", "manual"),
        position_metadata=position_metadata,
    )


# ── Legacy format parser ──────────────────────────────────────────────────────

def _parse_legacy_format(data: dict) -> PositionSnapshot:
    """Parse weight-fraction format (original schema).  No metadata."""
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

    return PositionSnapshot(
        timestamp=timestamp,
        nav=nav,
        positions=positions,
        cash_fraction=cash_fraction,
        source=data.get("source", "manual"),
        position_metadata={},   # no metadata in legacy format
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_positions_json(path: str | Path) -> PositionSnapshot:
    """Load a PositionSnapshot from a JSON file.

    Detects the format automatically (new amount-driven vs. legacy weight-fraction).
    See module docstring for schema details.

    Args:
        path: Path to current_positions.json.

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
            "Create it from data/current_positions.example.json"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    fmt = _detect_format(data)
    logger.debug(f"Detected positions format: {fmt}")

    if fmt == "new":
        return _parse_new_format(data)
    else:
        return _parse_legacy_format(data)


def load_positions_csv(path: str | Path, nav: float) -> PositionSnapshot:
    """Load a PositionSnapshot from a two-column CSV (symbol, weight).

    Args:
        path: Path to CSV with columns [symbol, weight].
        nav:  Total portfolio value in account currency.

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
        position_metadata={},
    )
