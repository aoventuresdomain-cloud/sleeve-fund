"""P2-5 regime: loading what the venue lists into `instruments` (DA design data-architect/p2-5-data-design.md).
Network calls stay in sleeve_fund.venues; this only books their rows, once, and says what it refused."""

from __future__ import annotations

from collections import Counter
from datetime import datetime


def sync_instruments(store, venue: str, listings: list[dict], now: datetime | None = None) -> dict:
    """Book each listing (the venue loader's listing rows) into `instruments`: counts of "new", "same" and "delisted",
    and "refused": the listings that differ from the contract held (a reused symbol, a listing day or delisting
    that moved), each with the store's reason. A refusal changes nothing and stops nothing else: the caller raises
    it as an alert."""
    counts: Counter = Counter()
    refused = []
    for row in listings:
        try:
            counts[store.record_instrument(venue, row["symbol"], underlying=row["underlying"], quote=row["quote"],
                                           contract_type=row["contract_type"], listed_at=row["listed_at"],
                                           source=row["source"], delisted_at=row.get("delisted_at"),
                                           seen_at=now)] += 1
        except ValueError as exc:
            refused.append((row["symbol"], str(exc)))
    return {"new": counts["new"], "same": counts["same"], "delisted": counts["delisted"], "refused": refused}
