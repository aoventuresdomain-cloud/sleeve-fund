"""Holdout locks (v2 P1-7, C4): an idea's held-back period is opened once per underlying, then locked.

The lock is keyed by the idea and the underlying asset, not a dataset name, so the same instrument at another
timeframe or venue can't open a fresh look at the same days. A holdout is also refused while any trial of the
idea read data inside it: those days are no longer unseen (Independent Quant Advisor, 5 Oct 2026).

Trials from the old idea counter kept no dates. Everything before the counter was retired counts as read by
them, so such an idea's holdout must fall wholly after that, last at least MIN_UNDATED_HOLDOUT_DAYS and hold at
least MIN_UNDATED_HOLDOUT_TRADES trades before it can be opened (Advisor, 5 Oct 2026, option b).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from sleeve_fund.store import Store

# When the file-based idea counter stopped being the record: every study from here on dates its trials. Undated
# trials count as having read every bar before this (or before their own run, if later).
COUNTER_RETIRED = pd.Timestamp("2026-10-06", tz="UTC")
MIN_UNDATED_HOLDOUT_DAYS = 90
MIN_UNDATED_HOLDOUT_TRADES = 20  # the per-instrument floor of the pooled 100-trade rule

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


def lock_id(idea_hash: str, underlying: str) -> str:
    return hashlib.sha256(f"{idea_hash}|{underlying.upper()}".encode()).hexdigest()[:16]


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
        return next((r for r in self.store.holdout_locks(idea_hash) if r["underlying"] == u), None)

    def refusal(self, idea_hash: str, underlying: str, start, end) -> str:
        """Why this idea can't open its holdout on this underlying from `start` to `end`, or '' when it can."""
        held = self.lock(idea_hash, underlying)
        if held is not None:
            when = _ts(held["opened_at"])
            return (f"this idea's holdout on {underlying.upper()} was opened on {when:%d %b %Y}, and a second look "
                    "can't be fresh")
        start, end = _ts(start), _ts(end)
        seen = [t for t in self.store.trials(idea_hash) if t["stage"] != "holdout"]
        undated = [t for t in seen if t["data_start"] is None or t["data_end"] is None]
        if undated:
            after = self.undated_until(undated)
            wanted = max(after, start) + pd.Timedelta(days=MIN_UNDATED_HOLDOUT_DAYS)
            if start < after or end < wanted:
                short = max(1, (wanted - end).days + (1 if (wanted - end) % pd.Timedelta(days=1) else 0))
                return (f"no holdout yet: {len(undated)} earlier evaluation{'s' if len(undated) != 1 else ''} of "
                        f"this idea kept no dates, so only days after {after:%d %b %Y} are unseen, and a holdout "
                        f"there needs {MIN_UNDATED_HOLDOUT_DAYS} days: {short} more days of data")
        inside = [t for t in seen if t not in undated and _ts(t["data_start"]) <= end and _ts(t["data_end"]) >= start]
        if inside:
            last = max(_ts(t["data_end"]) for t in inside)
            return (f"{len(inside)} earlier evaluation{'s' if len(inside) != 1 else ''} of this idea read data "
                    f"inside the held-back period (up to {last:%d %b %Y})")
        return ""

    def undated(self, idea_hash: str) -> bool:
        """True when some earlier evaluation of this idea kept no dates: its holdout also needs enough trades."""
        return any(t["data_start"] is None or t["data_end"] is None
                   for t in self.store.trials(idea_hash) if t["stage"] != "holdout")

    @staticmethod
    def undated_until(undated: list[dict]) -> pd.Timestamp:
        """The last moment undated evaluations may have read: the counter's retirement, or a later run."""
        return max([COUNTER_RETIRED, *(_ts(t["created_at"]) for t in undated)])

    def open(self, idea_hash: str, underlying: str, start, end) -> bool:
        """Claim the holdout before looking at it. False when another study already holds the lock: then this
        one must not look. A look that fails after the claim has still spent the holdout."""
        return self.store.add_holdout_lock({
            "id": lock_id(idea_hash, underlying), "idea_hash": idea_hash, "underlying": underlying,
            "period_start": _ts(start), "period_end": _ts(end), "source": "study",
        })

    def link_trial(self, idea_hash: str, underlying: str, trial_id: str) -> None:
        """Name the trial the look produced, once it exists."""
        self.store.link_holdout_trial(lock_id(idea_hash, underlying), trial_id)

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
                "id": lock_id(idea_hash, underlying),
                "idea_hash": idea_hash, "underlying": underlying, "opened_at": pd.Timestamp(e["ts"]).to_pydatetime(),
                "source": "ledger_import",
            })
        return added
