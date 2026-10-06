"""Settled funding rates of a perpetual venue, kept beside its price history and charged as the venue charged them.

A perpetual's funding is exchanged at fixed settlements (every 8 hours on most venues): longs pay shorts
position x price x rate when the rate is positive, and the reverse when it is negative. A venue that lists
perpetuals publishes every settled rate (VenueProfile.funding_loader); this keeps them per venue and
instrument as one JSON file in the history store's directory, topped up from the last one kept, and answers
"what was the rate at this settlement" for backtests and paper alike.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from sleeve_fund.history import DEFAULT_ROOT

PAGE = 1000  # settlements a venue returns per request
MATCH = pd.Timedelta(minutes=1)  # a settlement is stamped within this of its scheduled time
_lock = threading.Lock()
_cache: dict[tuple[str, str, str], tuple[float, pd.Series]] = {}  # (root, venue, pair) -> (file mtime, rates)


def _path(venue: str, pair: str, root: str | Path | None = None) -> Path:
    return Path(root or DEFAULT_ROOT) / venue.upper() / pair.upper().replace("/", "-") / "funding.json"


def rates(venue: str, pair: str, root: str | Path | None = None) -> pd.Series:
    """Every settled rate kept for the instrument, indexed by settlement time (UTC), oldest first."""
    path = _path(venue, pair, root)
    if not path.exists():
        return pd.Series(dtype=float)
    key, mtime = (str(path.parent.parent.parent), venue.upper(), pair.upper()), path.stat().st_mtime
    hit = _cache.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    raw = json.loads(path.read_text())["rates"]
    s = pd.Series([r for _, r in raw], index=pd.to_datetime([t for t, _ in raw], unit="ms", utc=True), dtype=float)
    _cache[key] = (mtime, s)
    return s


def refresh(venue: str, pair: str, root: str | Path | None = None, since: pd.Timestamp | None = None,
            loader=None, max_pages: int = 100) -> pd.Series:
    """Top up the kept rates from the venue, from the last one kept (or `since`, or the listing)."""
    from sleeve_fund.venues import venue as venue_profile

    loader = loader or venue_profile(venue).funding_loader
    if loader is None:
        raise ValueError(f"{venue_profile(venue).label} publishes no funding rates")
    path = _path(venue, pair, root)
    with _lock:
        kept = json.loads(path.read_text())["rates"] if path.exists() else []
        start = kept[-1][0] + 1 if kept else (int(since.timestamp() * 1000) if since is not None else 0)
        for _ in range(max_pages):
            page = loader(pair, start)
            kept.extend([t, r] for t, r in page if not kept or t > kept[-1][0])
            if len(page) < PAGE:
                break
            start = page[-1][0] + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rates": kept}))
        tmp.replace(path)
    return rates(venue, pair, root)


def fetch(venue: str, pair: str, since: pd.Timestamp, loader=None) -> pd.Series:
    """The venue's settled rates from `since`, asked for directly and not kept (paper reads the store only)."""
    from sleeve_fund.venues import venue as venue_profile

    loader = loader or venue_profile(venue).funding_loader
    if loader is None:
        raise ValueError(f"{venue_profile(venue).label} publishes no funding rates")
    page = loader(pair, int(since.timestamp() * 1000))
    return pd.Series([r for _, r in page], index=pd.to_datetime([t for t, _ in page], unit="ms", utc=True), dtype=float)


def rate_at(series: pd.Series, ts: pd.Timestamp) -> float | None:
    """The rate settled at `ts` (within a minute of it), or None if none was kept."""
    if series.empty:
        return None
    i = series.index.searchsorted(ts - MATCH)
    if i < len(series) and abs(series.index[i] - ts) <= MATCH:
        return float(series.iloc[i])
    return None


_NEAR = 3  # intervals either side of one that set what it is expected to be


def gaps(venue: str, pair: str, root: str | Path | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Settlements missing between the first and last rate kept, as (previous kept, next kept). A venue can change
    an instrument's settlement interval (8 hours to 4, say), so there is no fixed grid: a hole is an interval
    longer than one and a half times the shortest of the few intervals before it AND of the few after it. Taking
    the shortest finds two holes in a row (QA P1-O9), and needing both sides keeps a clean change of interval from
    counting. A hole exactly at a change can't be told from data alone (8h to 16:00 then 4h from 00:00 reads the
    same as a missed 20:00), so it is not reported."""
    t = rates(venue, pair, root).index
    steps = [b - a for a, b in zip(t, t[1:])]
    out = []
    for j, here in enumerate(steps):
        before, after = steps[max(j - _NEAR, 0):j], steps[j + 1:j + 1 + _NEAR]
        if (before or after) and all(here > 1.5 * min(side) for side in (before, after) if side):
            out.append((t[j], t[j + 1]))
    return out