"""Holdout locks (v2 P1-7, C4): an idea's held-back period is opened once per underlying, then locked.

The lock is keyed by the idea and the underlying asset, not a dataset name, so the same instrument at another
timeframe or venue can't open a fresh look at the same days. A holdout is also refused while any trial of the
idea read data inside it: those days are no longer unseen (Independent Quant Advisor, 5 Oct 2026). A trial with
unknown dates, from the old idea counter, counts as reading them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from sleeve_fund.store import Store

# Quote currencies a dataset name can end in, longest first, to find the underlying of an old counter entry.
_QUOTES = ("usdt", "usdc", "usd", "eur", "gbp")
_DATASET = re.compile(r"^[a-z0-9_]+-([a-z0-9]+?)(" + "|".join(_QUOTES) + r")-store(?:-\d+m)?$")


def underlying_of(pair: str) -> str:
    """The base asset of BASE/QUOTE, upper-case."""
    return pair.split("/")[0].strip().upper()


def underlying_of_dataset(dataset: str) -> str | None:
    """The base asset a stored-history dataset name was built from ("venue-btcusdt-store-60m" -> "BTC"), or None
    for a name that isn't one (synthetic data)."""
    m = _DATASET.match(dataset)
    return m.group(1).upper() if m else None


def _ts(x) -> pd.Timestamp | None:
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t


class HoldoutLocks:
    def __init__(self, store: Store) -> None:
        self.store = store

    def lock(self, idea_hash: str, underlying: str) -> dict | None:
        u = underlying.upper()
        return next((r for r in self.store.holdout_locks() if r["idea_hash"] == idea_hash and r["underlying"] == u),
                    None)

    def refusal(self, idea_hash: str, underlying: str, start, end) -> str:
        """Why this idea can't open its holdout on this underlying from `start` to `end`, or '' when it can."""
        held = self.lock(idea_hash, underlying)
        if held is not None:
            when = _ts(held["opened_at"])
            return (f"this idea's holdout on {underlying.upper()} was opened on {when:%d %b %Y}, and a second look "
                    "can't be fresh")
        start, end = _ts(start), _ts(end)
        seen = [t for t in self.store.trials() if t["idea_hash"] == idea_hash and t["stage"] != "holdout"]
        unknown = [t for t in seen if t["data_start"] is None or t["data_end"] is None]
        if unknown:
            return (f"{len(unknown)} earlier evaluation{'s' if len(unknown) != 1 else ''} of this idea "
                    "kept no dates, so they may have read the held-back days")
        inside = [t for t in seen if _ts(t["data_start"]) <= end and _ts(t["data_end"]) >= start]
        if inside:
            last = max(_ts(t["data_end"]) for t in inside)
            return (f"{len(inside)} earlier evaluation{'s' if len(inside) != 1 else ''} of this idea read data "
                    f"inside the held-back period (up to {last:%d %b %Y})")
        return ""

    def open(self, idea_hash: str, underlying: str, start, end, trial_id: str | None = None) -> bool:
        """Lock the holdout as opened now. False when it was already locked."""
        row_id = hashlib.sha256(f"{idea_hash}|{underlying.upper()}".encode()).hexdigest()[:16]
        return self.store.add_holdout_lock({
            "id": row_id, "idea_hash": idea_hash, "underlying": underlying, "period_start": _ts(start),
            "period_end": _ts(end), "trial_id": trial_id, "source": "study",
        })

    def import_ledger(self, path: str | Path) -> int:
        """Lock every holdout the old idea counter records as opened. Dates were not kept, so the period is
        unknown. Running this again adds nothing. Returns how many locks were added."""
        from sleeve_fund.research.trials import legacy_idea_hash

        path = Path(path)
        if not path.exists():
            return 0
        added = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            underlying = underlying_of_dataset(e["dataset"]) if e.get("stage") == "holdout" else None
            if underlying is None:
                continue
            idea_hash = legacy_idea_hash(e["idea"])
            added += self.store.add_holdout_lock({
                "id": hashlib.sha256(f"{idea_hash}|{underlying}".encode()).hexdigest()[:16],
                "idea_hash": idea_hash, "underlying": underlying, "opened_at": pd.Timestamp(e["ts"]).to_pydatetime(),
                "source": "ledger_import",
            })
        return added
