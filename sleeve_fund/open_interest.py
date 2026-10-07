"""Open interest of a perpetual venue's instruments, kept beside their price history (DA-10).

The venue publishes open interest as a snapshot at the end of each fixed period (5 minutes on the perpetual venue
collected from), and keeps only a recent window of them (30 days there). What it no longer publishes can't be fetched again, so this is
a record, not a cache: snapshots are only ever added. One JSON file per venue and instrument in the history store's
directory, topped up from the last snapshot kept. A snapshot already kept is never replaced: a later copy with
different numbers is written to the series' own `<series>.provenance.jsonl` (never the price series' file, whose
readers expect bar minutes). A missed period stays a hole, and gaps() reports it. No strategy reads this yet; it is collected so the history exists when one does.

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
import math
import uuid
from pathlib import Path

import pandas as pd

from sleeve_fund.history import DEFAULT_ROOT, _durable_replace, _writing

# Each series' values, in the venue loader's order. Open interest's notional is contracts x price: derived.
SERIES = {"open_interest": ("contracts", "notional"),
          "long_short_global": ("long_short_ratio", "long_share", "short_share"),
          "long_short_top": ("long_short_ratio", "long_share", "short_share")}
UNMEASURED_LAG = pd.Timedelta(minutes=20)  # until lag() has live snapshots to measure: the slow end of what we see
AT_RISK = pd.Timedelta(days=7)  # the venue keeps 30 days: say so loudly long before anything is lost
MIN_LIVE = 20  # live captures needed before their p95 is trusted over UNMEASURED_LAG (Advisor, 15:07 6 Oct)
NEVER_KEPT = pd.Timedelta(days=1)  # a series with nothing kept a day after collection started is failing


def _path(venue: str, pair: str, root: str | Path | None = None, series: str = "open_interest") -> Path:
    if series not in SERIES:
        raise ValueError(f"series is one of {sorted(SERIES)}, not {series!r}")
    return Path(root or DEFAULT_ROOT) / venue.upper() / pair.upper().replace("/", "-") / f"{series}.json"


def _file(path: Path) -> dict:
    """{"snapshots": [[stamp ms, *values, first_seen ms, backfill], ...], "since": ms the collector first tried}."""
    return json.loads(path.read_text()) if path.exists() else {"snapshots": []}


def _kept(path: Path) -> list[list]:
    return _file(path)["snapshots"]


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
    """The newest snapshot known at `at`, or None. A snapshot collected as the venue published it is known once it
    was first seen (first_seen <= at) AND stamped at least the lag before `at`; one from the first backfill, whose
    real arrival nobody saw, only by the lag (UNMEASURED_LAG until lag() has live snapshots to measure). The lag
    is measured from what was first seen by `at` alone. The only read path for backtests and strategies."""
    at = pd.Timestamp(at)
    df = snapshots(venue, pair, root, series)
    wait = _lag(df[df["first_seen"] <= at]) or UNMEASURED_LAG
    known = df[(df.index <= at - wait) & (df["backfill"] | (df["first_seen"] <= at))]
    return None if known.empty else known.iloc[-1][list(SERIES[series])]


def lag(venue: str, pair: str, root: str | Path | None = None, q: float = 0.95,
        series: str = "open_interest") -> pd.Timedelta | None:
    """How long after its stamp a snapshot reaches us: the q-quantile of (first_seen - stamp) over the newest
    snapshot of each fetch made as the venue published them (not the first backfill). The older snapshots in a
    fetch waited on this collector (an outage, a slow poll), not on the venue, so they would make the lag the
    outage (QA P1-O4). None until there are MIN_LIVE of them."""
    return _lag(snapshots(venue, pair, root, series), q)


def _lag(df: pd.DataFrame, q: float = 0.95) -> pd.Timedelta | None:
    live = df[~df["backfill"]]
    live = live[~live["first_seen"].duplicated(keep="last")]  # snapshots oldest first: last of a fetch is newest
    if len(live) < MIN_LIVE:
        return None
    return pd.Timedelta((live["first_seen"] - live.index).quantile(q))


def latest(venue: str, pair: str, root: str | Path | None = None, series: str = "open_interest") -> pd.Timestamp | None:
    rows = _kept(_path(venue, pair, root, series))
    return pd.Timestamp(rows[-1][0], unit="ms", tz="UTC") if rows else None


def provenance(venue: str, pair: str, root: str | Path | None = None, series: str = "open_interest") -> list[dict]:
    """The series' own record of differing venue copies and refused snapshots: never the price series' provenance
    (QA P1-O3), whose readers expect bar minutes."""
    path = _path(venue, pair, root, series).with_suffix(".provenance.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def refresh(venue: str, pair: str, root: str | Path | None = None, loader=None, max_pages: int = 100,
            series: str = "open_interest") -> dict:
    """Top up the kept snapshots from the venue, from the last one kept (or as far back as the venue keeps).
    Returns {"written", "unchanged", "conflicts", "refused", "latest"}. Safe to stop at any point and run again:
    the file is replaced whole, and a snapshot is written once. A snapshot with a value that isn't a finite
    number is refused and recorded, never kept."""
    from sleeve_fund.venues import venue as venue_profile

    loader = loader or venue_profile(venue).stats_loaders.get(series)
    if loader is None:
        raise ValueError(f"{venue_profile(venue).label} publishes no {series}")
    path = _path(venue, pair, root, series)
    now_ms = lambda: int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)  # noqa: E731
    with _writing(path.parent):  # when collection was first tried, kept across restarts for at_risk()
        if "since" not in (held := _file(path)):
            _replace(path, {**held, "since": now_ms()})
    # Fetch with no lock held (a backfill is many pages), then merge and write under the series directory's write
    # lock, shared across processes with the price history's writer (QA F3): a manual refresh while the hub runs
    # waits for the hub's write instead of overwriting it.
    kept = _kept(path)
    start = kept[-1][0] + 1 if kept else 0
    pages = []
    for _ in range(max_pages):
        pages.append(page := loader(pair, start))
        if not page or len(page) < 500 or page[-1][0] + 1 <= start:
            break
        start = page[-1][0] + 1
    written = unchanged = 0
    records: list[dict] = []
    with _writing(path.parent):
        held = _file(path)  # again: another writer may have added snapshots while we fetched
        kept = held["snapshots"]
        backfill = not kept  # the first fetch reaches back over the venue's whole window
        by_time = {r[0]: r for r in kept}
        # One first_seen for the whole fetch, taken once it is complete: no row is dated before it was held, and
        # a catch-up of many pages reads as one fetch, whose newest row alone counts towards lag() (QA P1-O4).
        seen = now_ms()
        for page in pages:
            for t, *values in page:
                values = [_number(v) for v in values]
                at = pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat()
                if not all(math.isfinite(v) for v in values):
                    records.append({"kind": "refused", "series": series, "snapshot": at, "offered": repr(values),
                                    "at": pd.Timestamp.now(tz="UTC").isoformat(), "source": "venue_rest"})
                    continue
                have = by_time.get(int(t))
                if have is None:
                    row = [int(t), *values, seen, int(backfill)]
                    kept.append(row)
                    by_time[int(t)] = row
                    written += 1
                elif have[1:-2] == values:
                    unchanged += 1
                else:
                    records.append({"kind": "conflict", "series": series, "snapshot": at, "stored": have[1:-2],
                                    "offered": values, "at": pd.Timestamp.now(tz="UTC").isoformat(),
                                    "source": "venue_rest"})
        if written:
            kept.sort(key=lambda r: r[0])
            _replace(path, {**held, "snapshots": kept})
        if records:
            with open(path.with_suffix(".provenance.jsonl"), "a") as f:
                f.writelines(json.dumps(e) + "\n" for e in records)
    return {"written": written, "unchanged": unchanged,
            "conflicts": sum(e["kind"] == "conflict" for e in records),
            "refused": sum(e["kind"] == "refused" for e in records),
            "latest": pd.Timestamp(kept[-1][0], unit="ms", tz="UTC") if kept else None}


def _number(v) -> float:
    """A venue value as a float; anything that isn't a number (null, junk) as NaN, so it is refused, not raised."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _replace(path: Path, content: dict) -> None:
    tmp = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(content))
    _durable_replace(tmp, path)  # a power loss leaves the old file or the new one, never an empty one


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
    once the newest snapshot this collector received (by first_seen, the Advisor's measure) is a week old, the
    collector is failing and must be fixed while it can catch up. A series with nothing kept is raised a day after
    collection was first tried (`since`, else as recorded in the series' file, which survives restarts), and at
    once if it was never even tried: the venue's window moves on daily, so each such day is lost for good."""
    path = _path(venue, pair, root, series)
    held = _file(path)
    now = now or pd.Timestamp.now(tz="UTC")
    name = series.replace("_", " ")
    if not held["snapshots"]:
        if since is None and "since" in held:
            since = pd.Timestamp(held["since"], unit="ms", tz="UTC")
        if since is None:
            return f"{venue.upper()} {pair}: {name} last kept never, and collection was never tried"
        if now - since > NEVER_KEPT:
            return (f"{venue.upper()} {pair}: {name} last kept never, though collected since {since:%Y-%m-%d %H:%M} "
                    f"UTC; the venue keeps 30 days, so each day without a snapshot loses a day for good")
        return None
    newest = max(pd.Timestamp(r[-2], unit="ms", tz="UTC") for r in held["snapshots"])
    if now - newest > AT_RISK:
        stamp = pd.Timestamp(held["snapshots"][-1][0], unit="ms", tz="UTC")
        lost = stamp + pd.Timedelta(days=30)
        return (f"{venue.upper()} {pair}: {name} last kept {newest:%Y-%m-%d %H:%M} UTC (newest snapshot "
                f"{stamp:%Y-%m-%d %H:%M}); the venue drops snapshots after 30 days, so they start being lost for good "
                f"from {lost:%Y-%m-%d %H:%M} UTC")
    return None
