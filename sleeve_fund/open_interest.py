"""Open interest of a perpetual venue's instruments, kept beside their price history (DA-10).

The venue publishes open interest as a snapshot at the end of each fixed period (5 minutes on Binance), and keeps
only a recent window of them (30 days on Binance). What it no longer publishes can't be fetched again, so this is
a record, not a cache: snapshots are only ever added. One JSON file per venue and instrument in the history store's
directory, topped up from the last snapshot kept. A snapshot already kept is never replaced: a later copy with
different numbers is written to the instrument's provenance.jsonl, as the hub's bars are. A missed period stays a
hole, and gaps() reports it. No strategy reads this yet; it is collected so the history exists when one does.

The venue's positioning ratios are kept the same way, as their own series: the share of all accounts long and short
(`long_short_global`) and of the top traders' positions (`long_short_top`). They are UNUSED and UNVALIDATED (HoE,
19:07): collected only because the venue drops them after 30 days. Nothing trades on or shows them.

Point in time (Independent Quant Advisor, 19:06): a snapshot is stamped at the end of its period, but the venue
publishes it some minutes later and the collector fetches it later still. Each snapshot therefore also keeps
`first_seen` (when this collector first held it). A snapshot counts as known only once stamp + lag() has passed,
where lag() is the 95th percentile of (first_seen - stamp) over snapshots collected as they were published, so
research never sees open interest that live would not yet have had. READ ONLY THROUGH as_of(), which applies that
rule; reading the files or snapshots() directly in a backtest or a strategy is refused in review.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from sleeve_fund.history import DEFAULT_ROOT, _durable_replace, _record, _writing

# Each series' values, in the venue loader's order. Open interest's notional is contracts x price: derived.
SERIES = {"open_interest": ("contracts", "notional"),
          "long_short_global": ("long_short_ratio", "long_share", "short_share"),
          "long_short_top": ("long_short_ratio", "long_share", "short_share")}
UNMEASURED_LAG = pd.Timedelta(minutes=20)  # until lag() has live snapshots to measure: the slow end of what we see
AT_RISK = pd.Timedelta(days=7)  # the venue keeps 30 days: say so loudly long before anything is lost
NEVER_KEPT = pd.Timedelta(days=1)  # a series with nothing kept a day after the collector started is failing
_STARTED = pd.Timestamp.now(tz="UTC")  # when this process started collecting


def _path(venue: str, pair: str, root: str | Path | None = None, series: str = "open_interest") -> Path:
    if series not in SERIES:
        raise ValueError(f"series is one of {sorted(SERIES)}, not {series!r}")
    return Path(root or DEFAULT_ROOT) / venue.upper() / pair.upper().replace("/", "-") / f"{series}.json"


def _kept(path: Path) -> list[list]:
    return json.loads(path.read_text())["snapshots"] if path.exists() else []


def snapshots(venue: str, pair: str, root: str | Path | None = None, series: str = "open_interest") -> pd.DataFrame:
    """Every snapshot kept, indexed by its time (UTC, the end of its period), oldest first. For audit and the
    collector: research and strategies read through as_of()."""
    rows = _kept(_path(venue, pair, root, series))
    index = pd.DatetimeIndex(pd.to_datetime([r[0] for r in rows], unit="ms", utc=True), name="timestamp")
    df = pd.DataFrame([r[1:-2] for r in rows], index=index, columns=list(SERIES[series]), dtype=float)
    df["first_seen"] = pd.to_datetime([r[-2] for r in rows], unit="ms", utc=True)
    df["backfill"] = [bool(r[-1]) for r in rows]
    return df


def as_of(venue: str, pair: str, at, root: str | Path | None = None, series: str = "open_interest") -> pd.Series | None:
    """The newest snapshot known at `at`: stamped at least lag() before it (UNMEASURED_LAG until lag() has live
    snapshots to measure). None if none was known yet. The only read path for backtests and strategies."""
    at = pd.Timestamp(at)
    df = snapshots(venue, pair, root, series)
    known = df[df.index <= at - (lag(venue, pair, root, series=series) or UNMEASURED_LAG)]
    return None if known.empty else known.iloc[-1][list(SERIES[series])]


def lag(venue: str, pair: str, root: str | Path | None = None, q: float = 0.95,
        series: str = "open_interest") -> pd.Timedelta | None:
    """How long after its stamp a snapshot reaches us: the q-quantile of (first_seen - stamp) over snapshots
    collected as the venue published them (not the first backfill). None until there are any."""
    df = snapshots(venue, pair, root, series)
    live = df[~df["backfill"]]
    if live.empty:
        return None
    return pd.Timedelta((live["first_seen"] - live.index).quantile(q))


def latest(venue: str, pair: str, root: str | Path | None = None, series: str = "open_interest") -> pd.Timestamp | None:
    rows = _kept(_path(venue, pair, root, series))
    return pd.Timestamp(rows[-1][0], unit="ms", tz="UTC") if rows else None


def refresh(venue: str, pair: str, root: str | Path | None = None, loader=None, max_pages: int = 100,
            series: str = "open_interest") -> dict:
    """Top up the kept snapshots from the venue, from the last one kept (or as far back as the venue keeps).
    Returns {"written", "unchanged", "conflicts", "latest"}. Safe to stop at any point and run again: the file is
    replaced whole, and a snapshot is written once."""
    from sleeve_fund.venues import venue as venue_profile

    loader = loader or venue_profile(venue).stats_loaders.get(series)
    if loader is None:
        raise ValueError(f"{venue_profile(venue).label} publishes no {series}")
    path = _path(venue, pair, root, series)
    # Fetch with no lock held (a backfill is many pages), then merge and write under the series directory's write
    # lock, shared across processes with the price history's writer (QA F3): a manual refresh while the hub runs
    # waits for the hub's write instead of overwriting it.
    kept = _kept(path)
    start = kept[-1][0] + 1 if kept else 0
    pages = []
    for _ in range(max_pages):
        page = loader(pair, start)
        pages.append((page, int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)))
        if not page or len(page) < 500 or page[-1][0] + 1 <= start:
            break
        start = page[-1][0] + 1
    written = unchanged = 0
    conflicts: list[dict] = []
    with _writing(path.parent):
        kept = _kept(path)  # again: another writer may have added snapshots while we fetched
        backfill = not kept  # the first fetch reaches back over the venue's whole window
        by_time = {r[0]: r for r in kept}
        for page, seen in pages:
            for t, *values in page:
                have = by_time.get(int(t))
                if have is None:
                    row = [int(t), *map(float, values), seen, int(backfill)]
                    kept.append(row)
                    by_time[int(t)] = row
                    written += 1
                elif have[1:-2] == [float(v) for v in values]:
                    unchanged += 1
                else:
                    conflicts.append({"kind": "conflict", "series": series,
                                      "at": pd.Timestamp.now(tz="UTC").isoformat(), "source": "venue_rest",
                                      "snapshot": pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat(),
                                      "stored": have[1:-2], "offered": [float(v) for v in values]})
        if written:
            kept.sort(key=lambda r: r[0])
            tmp = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp"
            tmp.write_text(json.dumps({"snapshots": kept}))
            _durable_replace(tmp, path)  # a power loss leaves the old file or the new one, never an empty one
        _record(path.parent, conflicts)
    return {"written": written, "unchanged": unchanged, "conflicts": len(conflicts),
            "latest": pd.Timestamp(kept[-1][0], unit="ms", tz="UTC") if kept else None}


def gaps(venue: str, pair: str, root: str | Path | None = None, period: pd.Timedelta | None = None,
         series: str = "open_interest") -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Missing periods between the first and last snapshot kept, as (first missing, last missing)."""
    from sleeve_fund.venues import venue as venue_profile

    period = period or pd.Timedelta(minutes=venue_profile(venue).stats_minutes)
    times = [r[0] for r in _kept(_path(venue, pair, root, series))]
    step = int(period / pd.Timedelta(milliseconds=1))
    out = []
    for a, b in zip(times, times[1:]):
        if b - a > step:
            out.append((pd.Timestamp(a + step, unit="ms", tz="UTC"), pd.Timestamp(b - step, unit="ms", tz="UTC")))
    return out


def at_risk(venue: str, pair: str, root: str | Path | None = None, now: pd.Timestamp | None = None,
            series: str = "open_interest", since: pd.Timestamp | None = None) -> str | None:
    """Why this instrument's open interest is close to being lost for good, or None. The venue keeps 30 days, so
    once the newest snapshot kept is a week old the collector is failing and must be fixed while it can catch up.
    A series never kept at all is raised a day after the collector started (`since`, this process's start by
    default): the venue's window moves on daily, so a loader failing from the first deploy is never silent."""
    newest = latest(venue, pair, root, series)
    now = now or pd.Timestamp.now(tz="UTC")
    name = series.replace("_", " ")
    if newest is None:
        since = since or _STARTED
        if now - since > NEVER_KEPT:
            return (f"{venue.upper()} {pair}: {name} last kept never, though collected since {since:%Y-%m-%d %H:%M} "
                    f"UTC; the venue keeps 30 days, so each day without a snapshot loses a day for good")
        return None
    if now - newest > AT_RISK:
        lost = newest + pd.Timedelta(days=30)
        return (f"{venue.upper()} {pair}: {name} last kept {newest:%Y-%m-%d %H:%M} UTC; the venue drops "
                f"snapshots after 30 days, so they start being lost for good from {lost:%Y-%m-%d %H:%M} UTC")
    return None
