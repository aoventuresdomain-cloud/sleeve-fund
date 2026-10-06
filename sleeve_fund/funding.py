"""Settled funding rates of a perpetual venue, kept beside its price history and charged as the venue charged them.

A perpetual's funding is exchanged at fixed settlements (every 8 hours on most venues): longs pay shorts
position x price x rate when the rate is positive, and the reverse when it is negative. A venue that lists
perpetuals publishes every settled rate (VenueProfile.funding_loader); this keeps them per venue and
instrument as one JSON file in the history store's directory, topped up from the last one kept, and answers
"what was the rate at this settlement" for backtests and paper alike.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import pandas as pd

from sleeve_fund.history import DEFAULT_ROOT

PAGE = 1000  # settlements a venue returns per request
MATCH = pd.Timedelta(minutes=1)  # a settlement is stamped within this of its scheduled time
# A settled rate beyond the venue's cap (VenueProfile.funding_cap), either way, is not believed: kept as missing and
# said. This is the default cap, a sanity bound well past any a venue sets per settlement, until each instrument's
# published cap is kept beside its rates (DA-11).
CAP = 0.05
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
            loader=None, max_pages: int = 100, refused: list | None = None) -> pd.Series:
    """Top up the kept rates from the venue, from the last one kept (or `since`, or the listing). Settlements whose
    rate can't be believed are left out and added to `refused` as (time in ms, rate), for the collector to alert."""
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
            kept.extend([t, r] for t, r in _usable(page, venue, pair, refused) if not kept or t > kept[-1][0])
            if len(page) < PAGE:
                break
            start = page[-1][0] + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rates": kept}))
        tmp.replace(path)
    return rates(venue, pair, root)


def _usable(page, venue: str, pair: str, refused: list | None = None) -> list:
    """The page's settlements whose rate can be believed. A null from the venue (the loader passes it as None), a NaN,
    an infinity or a rate beyond the venue's cap is left out and said, so it is a hole gaps() reports rather than a
    page lost or a bad number charged (QA P1-O13, P1-O17)."""
    cap = cap_of(venue, pair)
    out = [(t, r) for t, r in page if believable(r, cap)]
    for t, r in page:
        if not believable(r, cap):
            print(f"{venue.upper()} {pair}: funding at {pd.Timestamp(t, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} refused: "
                  f"rate {r!r} is not a number within {cap:.2%}")
            if refused is not None:
                refused.append((t, r))
    return out


def cap_of(venue: str, pair: str) -> float:
    """The venue's cap on |rate| per settlement for the instrument (VenueProfile.funding_cap)."""
    from sleeve_fund.venues import venue as venue_profile

    return venue_profile(venue).funding_cap(pair)


def believable(rate, cap: float = CAP) -> bool:
    """A settled rate that can be charged: a finite number within the cap. Anything else is a missing rate (QA P1-O17)."""
    try:
        r = float(rate)
    except (TypeError, ValueError):
        return False
    return math.isfinite(r) and abs(r) <= cap


def fetch(venue: str, pair: str, since: pd.Timestamp, loader=None) -> pd.Series:
    """The venue's settled rates from `since`, asked for directly and not kept (paper reads the store only)."""
    from sleeve_fund.venues import venue as venue_profile

    loader = loader or venue_profile(venue).funding_loader
    if loader is None:
        raise ValueError(f"{venue_profile(venue).label} publishes no funding rates")
    page = _usable(loader(pair, int(since.timestamp() * 1000)), venue, pair)
    return pd.Series([r for _, r in page], index=pd.to_datetime([t for t, _ in page], unit="ms", utc=True), dtype=float)


def rate_at(series: pd.Series, ts: pd.Timestamp, cap: float = CAP) -> float | None:
    """The rate settled at `ts` (within a minute of it), or None if none was kept or it can't be believed."""
    if series.empty:
        return None
    i = series.index.searchsorted(ts - MATCH)
    if i < len(series) and abs(series.index[i] - ts) <= MATCH:
        rate = float(series.iloc[i])
        # A NaN, infinite or unbelievable rate in a file the collector didn't write is missing, never charged
        # (QA P1-O17: a NaN crashed the strategy).
        return rate if believable(rate, cap) else None
    return None


_NEAR = 3  # intervals either side of one that set what it is expected to be


def gaps(venue: str, pair: str, root: str | Path | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Settlements missing between the first and last rate kept, as (previous kept, next kept). A venue can change
    an instrument's settlement interval (8 hours to 4, say), so there is no fixed grid: a hole is an interval
    longer than one and a half times the shortest of the few intervals before it AND of the few after it. Taking
    the shortest finds two holes in a row (QA P1-O9), and needing both sides keeps a clean change of interval from
    counting. A hole exactly at a change can't be told from data alone (8h to 16:00 then 4h from 00:00 reads the
    same as a missed 20:00), so it is not reported here: interval_changes() lists each such interval as a possible
    hole, and the collector's log says so."""
    t = rates(venue, pair, root).index
    steps = [b - a for a, b in zip(t, t[1:])]
    out = []
    for j, here in enumerate(steps):
        before, after = steps[max(j - _NEAR, 0):j], steps[j + 1:j + 1 + _NEAR]
        if (before or after) and all(here > 1.5 * min(side) for side in (before, after) if side):
            out.append((t[j], t[j + 1]))
    return out


def interval_changes(venue: str, pair: str, root: str | Path | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Each interval where the settlement interval changes, as (previous kept, next kept) for the longer interval at
    the change: it can't be told from the rates alone whether that interval was the old schedule or the new one with
    settlements missed (QA P1-O9), so it is a possible hole, reported beside gaps() rather than passed silently.
    The venue's published interval, once kept with each rate, would settle it (board follow-up)."""
    t = rates(venue, pair, root).index
    steps = [b - a for a, b in zip(t, t[1:])]
    holes = set(gaps(venue, pair, root))
    out = []
    for j in range(len(steps) - 1):
        if abs(steps[j] - steps[j + 1]) > MATCH:  # the venue's stamps jitter by milliseconds
            k = j if steps[j] > steps[j + 1] else j + 1
            span = (t[k], t[k + 1])
            if span not in holes and span not in out:
                out.append(span)
    return out


def settled_holes(venue: str, pair: str, root: str | Path | None = None) -> tuple[list, list]:
    """gaps() and interval_changes() for the spans a later settlement has been kept after. While a span is the newest
    interval there is nothing after it to compare with, so the first long interval after a change to a longer
    schedule would read as a missed settlement (QA P1-O18): it is judged once the next settlement is kept, and a
    real hole is reported one interval later."""
    t = rates(venue, pair, root).index
    if not len(t):
        return [], []
    return ([g for g in gaps(venue, pair, root) if g[1] < t[-1]],
            [g for g in interval_changes(venue, pair, root) if g[1] < t[-1]])


def schedule_mismatch(venue: str, pair: str, hours: tuple[int, ...], root: str | Path | None = None,
                      start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                      latest: bool = False) -> str | None:
    """Why the instrument's stored settlements don't fit the schedule funding is charged on, or None. The venue can
    move an instrument to a shorter interval (8 hours to 4 or 1) when funding hits its cap, exactly when it is largest,
    and the engine still charges on its fixed schedule, so those settlements would be skipped (Advisor, 6 Oct 2026).
    Interim guard until funding is charged at each stored rate's own time (DA-11). start/end: only steps inside the
    run; latest: only the newest step (paper, before a strategy starts)."""
    from sleeve_fund.markets import funding_interval

    t = rates(venue, pair, root).index
    if not len(t):
        return None
    if start is not None:
        t = t[t >= start]
    if end is not None:
        t = t[t <= end]
    expected = pd.Timedelta(funding_interval(hours))
    steps = list(zip(t, t[1:]))[-1:] if latest else list(zip(t, t[1:]))
    short = [(a, b) for a, b in steps if b - a < expected - MATCH]
    if not short:
        return None
    a, b = short[0]
    return (f"funding schedule mismatch: {pair} settled {(b - a) / pd.Timedelta(hours=1):g} hours apart at "
            f"{b:%Y-%m-%d %H:%M} UTC ({len(short)} such step{'s' if len(short) != 1 else ''}), but funding is charged "
            f"every {expected / pd.Timedelta(hours=1):g} hours, so settlements would be missed")


# A store is stale once its newest rate is this many settlement intervals old: one late settlement is the venue
# publishing late, two missed means the collector has stopped (QA P1-O17).
STALE_INTERVALS = 2


def stale(venue: str, pair: str, root: str | Path | None = None, now: pd.Timestamp | None = None) -> str | None:
    """Why this instrument's funding store has stopped keeping up, or None. gaps() only sees holes between kept
    rates, so a feed that simply stops would pass unnoticed; backtests over the missing stretch then charge the
    baseline (QA P1-O17). The interval is the newest kept step (8 hours when fewer than two are kept)."""
    t = rates(venue, pair, root).index
    if not len(t):
        return None  # nothing collected yet: the instrument isn't followed, or its first pass is still to come
    now = now or pd.Timestamp.now(tz="UTC")
    step = t[-1] - t[-2] if len(t) > 1 else pd.Timedelta(hours=8)
    if now - t[-1] > STALE_INTERVALS * step + MATCH:
        missed = int((now - t[-1]) / step)
        return (f"{venue.upper()} {pair}: funding last kept {t[-1]:%Y-%m-%d %H:%M} UTC, about {missed} settlement"
                f"{'s' if missed != 1 else ''} ago; backtests over the missing stretch charge the baseline rate")
    return None


# Funding charged at the baseline for a missing rate (Advisor, 6 Oct 2026, QA P1-O17): from BASELINE_WARN of the held
# settlements a result warns; above BASELINE_LIMIT, or a stretch of held time longer than BASELINE_RUN_LIMIT, it is
# not judged until the rates are backfilled.
BASELINE_WARN = 0.01
BASELINE_LIMIT = 0.05
BASELINE_RUN_LIMIT = pd.Timedelta(days=7)


def baseline_summary(marks: list, interval: pd.Timedelta | None = None) -> tuple[int, int, pd.Timedelta]:
    """(N held settlements charged the baseline, M held settlements, the longest stretch) from (ts, missing, held)
    marks. A stretch is an unbroken run of missing rates that only a real rate breaks; its length is the held time
    inside it, one interval per held settlement, so flat time neither breaks it nor adds to it (Advisor, 18:18).
    The interval, when not given, is the shortest step between the marks."""
    if interval is None:
        steps = [pd.Timestamp(b[0]) - pd.Timestamp(a[0]) for a, b in zip(marks, marks[1:])]
        interval = min((t for t in steps if t > pd.Timedelta(0)), default=pd.Timedelta(hours=8))
    n = m = 0
    run = longest = pd.Timedelta(0)
    for _, missing, held in marks:
        m += held
        if not missing:
            run = pd.Timedelta(0)
            continue
        if held:
            n += 1
            run += interval
            longest = max(longest, run)
    return n, m, longest


def baseline_check(at_baseline: int, held: int, longest_run: pd.Timedelta, where: str = "") -> tuple[str, str]:
    """PASS, WARN or NOT JUDGED for a result's funding, with "N of M" in words."""
    share = at_baseline / held if held else 0.0
    what = f"{where} " if where else ""
    words = (f"{at_baseline} of {held} {what}funding settlements held through charged the baseline for a missing "
             f"rate ({share:.1%})" + (f", the longest stretch {_held_words(longest_run)} held" if at_baseline else ""))
    if held and (share > BASELINE_LIMIT or longest_run > BASELINE_RUN_LIMIT):
        return "NOT JUDGED", f"{words}; backfill the funding rates and run it again"
    if held and share >= BASELINE_WARN:
        return "WARN", f"{words}; backfill the funding rates before relying on it"
    return "PASS", words


def _held_words(t: pd.Timedelta) -> str:
    days = t / pd.Timedelta(days=1)
    return f"{days:.1f} days" if days >= 1 else f"{t / pd.Timedelta(hours=1):.0f} hours"
