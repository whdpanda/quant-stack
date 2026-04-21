"""Date utilities used across all layers."""

from __future__ import annotations

from datetime import date, timedelta


def parse_date(value: str | date) -> date:
    """Accept a date object or an ISO-8601 string and return a date."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def date_range(start: str | date, end: str | date) -> tuple[date, date]:
    """Parse and validate a (start, end) date pair."""
    s, e = parse_date(start), parse_date(end)
    if s >= e:
        raise ValueError(f"start ({s}) must be before end ({e})")
    return s, e


def last_n_years(n: int, end: date | None = None) -> tuple[date, date]:
    """Return (start, end) covering the last n calendar years."""
    end = end or date.today()
    start = end.replace(year=end.year - n)
    return start, end


def trading_days_approx(start: date, end: date) -> int:
    """Rough estimate: 252 trading days per year."""
    delta = (end - start).days
    return int(delta * 252 / 365)
