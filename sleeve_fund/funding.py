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
from sleeve_fund.markets import SNAP_WINDOW as _SNAP_WINDOW

PAGE = 1000  # settlements a venue returns per request
MATCH = pd.Timedelta(_SNAP_WINDOW)  # a settlement is stamped within this of its time (markets.SNAP_WINDOW)
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


def backfill(venue: str, pair: str, after: pd.Timestamp, root: str | Path | None = None, loader=None,
             max_pages: int = 10) -> int:
    """Ask the venue's history again for the settlements from `after` up to the newest kept, and keep any it now has
    that the store lacks (a missing settlement, before it is called never published; Advisor, 7 Oct 2026, QA
    P1-O17a-11). refresh() only tops up past the newest kept, so a hole is never refetched otherwise. How many were
    added."""
    from sleeve_fund.venues import venue as venue_profile

    loader = loader or venue_profile(venue).funding_loader
    if loader is None:
        return 0
    path = _path(venue, pair, root)
    with _lock:
        if not path.exists():
            return 0
        kept = json.loads(path.read_text())["rates"]
        if not kept:
            return 0
        have, last = {t for t, _ in kept}, kept[-1][0]
        start, added = int(after.timestamp() * 1000), []
        for _ in range(max_pages):
            page = loader(pair, start)
            added += [[t, r] for t, r in _usable(page, venue, pair) if t < last and t not in have]
            if len(page) < PAGE or not page or page[-1][0] >= last:
                break
            start = page[-1][0] + 1
        if added:
            kept = sorted(kept + added, key=lambda tr: tr[0])
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"rates": kept}))
            tmp.replace(path)
            _cache.pop((str(path.parent.parent.parent), venue.upper(), pair.upper()), None)
    return len(added)


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


def snap_note(series: pd.Series, ts) -> str:
    """The audit note " (record stamped HH:MM UTC, snapped +N min)" when the record charged at settlement `ts` is stamped off it
    (inside MATCH, so it is that
    settlement's rate, published late or early: Advisor, 7 Oct 2026), else "": kept for the audit."""
    if series is None or series.empty:
        return ""
    ts = pd.Timestamp(ts)
    i = series.index.searchsorted(ts - MATCH)
    if i + 1 < len(series) and abs(series.index[i + 1] - ts) < abs(series.index[i] - ts):
        i += 1
    if i >= len(series) or abs(series.index[i] - ts) > MATCH:
        return ""
    off = round((series.index[i] - ts) / pd.Timedelta(minutes=1))
    return f" (record stamped {series.index[i]:%H:%M} UTC, snapped {off:+d} min)" if off else ""


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
    if i + 1 < len(series) and abs(series.index[i + 1] - ts) < abs(series.index[i] - ts):
        i += 1  # the nearer of two inside the minute either side: one record to one settlement (QA P1-O17a-14)
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


def stale_tag(venue: str, pair: str) -> str:
    """The prefix of an instrument's funding_stale / funding_stale_cleared messages, the same whether the collector
    or a paper strategy raises it, so one episode per instrument is alerted once, whoever notices first (CR, #163).
    The venue stays out of it: the alerts are read by the PM, and each instrument is followed on one venue."""
    return f"[{pair.upper()}]"


# The journal's events for an instrument's funding (Advisor, 7 Oct 2026, QA P1-O17a-11): an episode is a run of
# missing settlements, opened by one funding_stale and closed by one funding_stale_cleared, both naming the
# settlement it opened on ("episode from ... UTC"); each missing settlement is marked once (funding_missing), and one still
# missing a day after it was due, once a later one is published, once more (funding_never_published).
EPISODE_KINDS = ("funding_stale", "funding_stale_cleared", "funding_missing", "funding_never_published")
NEVER_PUBLISHED_AFTER = pd.Timedelta(hours=24)
_STAMP = "%Y-%m-%d %H:%M"


def _stamp_in(message: str, before: str) -> pd.Timestamp | None:
    """The 'YYYY-MM-DD HH:MM UTC' time just after `before` in an episode message, or None."""
    at = message.find(before)
    if at < 0:
        return None
    try:
        return pd.Timestamp(message[at + len(before):at + len(before) + 16], tz="UTC")
    except ValueError:
        return None


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def journal_state(store, tag: str) -> dict:
    """The instrument's funding episodes as the journal has them: {"open": {opening settlement: its funding_stale},
    "missing": settlements marked missing, "never": settlements marked never published}. A funding_stale written
    before episodes named their settlement opens one at its own time, and a funding_stale_cleared naming none
    closes every one then open (one episode per instrument, as they were). Raises when the journal can't be read."""
    events = [e for e in store.events_of(EPISODE_KINDS, limit=2000) if e["message"].startswith(tag)]
    opened: dict = {}
    missing, never = set(), set()
    for e in reversed(events):  # oldest first
        kind, msg = e["kind"], e["message"]
        if kind == "funding_missing" or kind == "funding_never_published":
            t = _stamp_in(msg, f"{tag} ")
            if t is not None:
                (missing if kind == "funding_missing" else never).add(t)
            continue
        key = _stamp_in(msg, "episode from ")
        if kind == "funding_stale":
            key = key if key is not None else (_utc(e["ts"]) if e.get("ts") is not None else None)
            opened.setdefault(key, e)
        elif key is None:
            opened.clear()
        else:
            opened.pop(key, None)
    return {"open": opened, "missing": missing, "never": never}


def stale_open(store, tag: str) -> bool | None:
    """Whether any of the instrument's staleness episodes is open in the journal, however long ago it opened, so an
    outage of days alerts once and a rate arriving days late still logs its recovery (QA P1-O17a-1). None when the
    journal can't be read: the caller then alerts, since alerting twice beats never."""
    try:
        return bool(journal_state(store, tag)["open"])
    except Exception:  # noqa: BLE001 - no database (locally), or a stub inbox
        return None


def stale_since(store, tag: str) -> pd.Timestamp | None:
    """The settlement the instrument's oldest open staleness episode opened on (UTC), or None when none is open or
    the journal can't be read."""
    try:
        keys = [k for k in journal_state(store, tag)["open"] if k is not None]
    except Exception:  # noqa: BLE001 - no database (locally), or a stub inbox
        return None
    return min(keys) if keys else None


def episode_of(state: dict, t: pd.Timestamp, due: list) -> pd.Timestamp | None:
    """The open episode missing settlement `t` belongs to: one opened at or before it with every settlement from its
    opening to `t` (`due`, the instrument's settlements in that span) marked missing or never published, so one
    outage is one episode, and a missing settlement after a published one opens another (Advisor, 7 Oct 2026)."""
    gone = state["missing"] | state["never"]
    for o in sorted((k for k in state["open"] if k is not None and k <= t), reverse=True):
        if all(_utc(u) in gone for u in due if o <= _utc(u) < t):
            return o
    return None


def mark(store, tag: str, kind: str, t: pd.Timestamp, ts=None, inferred: bool = False) -> None:
    """Journal one settlement's state: funding_missing (info) or funding_never_published (a warning)."""
    if kind == "funding_missing":
        store.event(None, "info", kind, f"{tag} {t:{_STAMP}} UTC settlement missing"
                    + (" (inferred time)" if inferred else ""), ts=ts)
    else:
        store.event(None, "warning", kind, f"{tag} {t:{_STAMP}} UTC rate never published; baseline kept, true-up "
                    "impossible", ts=ts)


def from_words(o: pd.Timestamp) -> str:
    """The episode's name in its funding_stale and funding_stale_cleared messages."""
    return f"episode from {o:{_STAMP}} UTC"


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
