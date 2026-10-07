"""Strategy allocations (v2 P2-1): when the monthly rebalance falls. The allocations themselves, and their ledger,
are the Data Architect's tables."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def first_monday(year: int, month: int) -> datetime:
    """00:00 UTC on the first Monday of the month."""
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    return first + timedelta(days=(7 - first.weekday()) % 7)


def next_rebalance(after: datetime) -> datetime:
    """The next monthly rebalance strictly after `after`: the first Monday of each month at 00:00 UTC (PM, 6 Oct
    2026, 15:51)."""
    if after.tzinfo is None:
        raise ValueError("a time with its time zone, so the rebalance is in UTC")
    after = after.astimezone(timezone.utc)
    due = first_monday(after.year, after.month)
    if due <= after:
        year, month = (after.year + 1, 1) if after.month == 12 else (after.year, after.month + 1)
        due = first_monday(year, month)
    return due
