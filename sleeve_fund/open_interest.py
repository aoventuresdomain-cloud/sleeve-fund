"""Open interest of a perpetual venue's instruments, kept beside their price history (DA-10).

The venue publishes open interest as a snapshot at the end of each fixed period (5 minutes on Binance), and keeps
only a recent window of them (30 days on Binance). What it no longer publishes can't be fetched again, so this is
a record, not a cache: snapshots are only ever added. One JSON file per venue and instrument in the history store's
directory, topped up from the last snapshot kept. A snapshot already kept is never replaced: a later copy with
different numbers is written to the instrument's provenance.jsonl, as the hub's bars are. A missed period stays a
hole, and gaps() reports it. No strategy reads this yet; it is collected so the history exists when one does.

Point in time (Independent Quant Advisor, 19:06): a snapshot is stamped at the end of its period, but the venue
publishes it some minutes later and the collector fetches it later still. Each snapshot therefore also keeps
`first_seen` (when this collector first held it). A backtest may read a snapshot only once stamp + lag() has passed,
where lag() is the 95th percentile of (first_seen - stamp) over snapshots collected as they were published, so
research never sees open interest that live would not yet have had.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from sleeve_fund.history import DEFAULT_ROOT

COLUMNS = ("contracts", "notional", "first_seen", "backfill")  # notional is contracts x price: derived, not independent
AT_RISK = pd.Timedelta(days=7)  # the venue keeps 30 days: say so loudly long before anything is lost
_lock = threading.Lock()


def _path(venue: str, pair: str, root: str | Path | None = None) -> Path:
    return Path(root or DEFAULT_ROOT) / venue.upper() / pair.upper().replace("/", "-") / "open_interest.json"


def _kept(path: Path) -> list[list]:
    return json.loads(path.read_text())["snapshots"] if path.exists() else []


def snapshots(venue: str, pair: str, root: str | Path | None = None) -> pd.DataFrame:
    """Every snapshot kept, indexed by its time (UTC, the end of its period), oldest first."""
    rows = _kept(_path(venue, pair, root))
    index = pd.DatetimeIndex(pd.to_datetime([r[0] for r in rows], unit="ms", utc=True), name="timestamp")
    df = pd.DataFrame([r[1:3] for r in rows], index=index, columns=list(COLUMNS[:2]), dtype=float)
    df["first_seen"] = pd.to_datetime([r[3] for r in rows], unit="ms", utc=True)
    df["backfill"] = [bool(r[4]) for r in rows]
    return df


def lag(venue: str, pair: str, root: str | Path | None = None, q: float = 0.95) -> pd.Timedelta | None:
    """How long after its stamp a snapshot reaches us: the q-quantile of (first_seen - stamp) over snapshots
    collected as the venue published them (not the first backfill). None until there are any."""
    df = snapshots(venue, pair, root)
    live = df[~df["backfill"]]
    if live.empty:
        return None
    return pd.Timedelta((live["first_seen"] - live.index).quantile(q))


def latest(venue: str, pair: str, root: str | Path | None = None) -> pd.Timestamp | None:
    rows = _kept(_path(venue, pair, root))
    return pd.Timestamp(rows[-1][0], unit="ms", tz="UTC") if rows else None


def refresh(venue: str, pair: str, root: str | Path | None = None, loader=None, max_pages: int = 100) -> dict:
    """Top up the kept snapshots from the venue, from the last one kept (or as far back as the venue keeps).
    Returns {"written", "unchanged", "conflicts", "latest"}. Safe to stop at any point and run again: the file is
    replaced whole, and a snapshot is written once."""
    from sleeve_fund.venues import venue as venue_profile

    loader = loader or venue_profile(venue).open_interest_loader
    if loader is None:
        raise ValueError(f"{venue_profile(venue).label} publishes no open interest")
    path = _path(venue, pair, root)
    written = unchanged = 0
    conflicts: list[dict] = []
    with _lock:
        kept = _kept(path)
        backfill = not kept  # the first fetch reaches back over the venue's whole window
        by_time = {r[0]: r for r in kept}
        start = kept[-1][0] + 1 if kept else 0
        for _ in range(max_pages):
            page = loader(pair, start)
            seen = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
            for t, *values in page:
                have = by_time.get(int(t))
                if have is None:
                    row = [int(t), *map(float, values), seen, int(backfill)]
                    kept.append(row)
                    by_time[int(t)] = row
                    written += 1
                elif have[1:3] == [float(v) for v in values]:
                    unchanged += 1
                else:
                    conflicts.append({"kind": "conflict", "series": "open_interest",
                                      "at": pd.Timestamp.now(tz="UTC").isoformat(), "source": "venue_rest",
                                      "snapshot": pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat(),
                                      "stored": have[1:3], "offered": [float(v) for v in values]})
            if not page or len(page) < 500 or page[-1][0] + 1 <= start:
                break
            start = page[-1][0] + 1
        if written:
            kept.sort(key=lambda r: r[0])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"snapshots": kept}))
            tmp.replace(path)
        if conflicts:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path.parent / "provenance.jsonl", "a") as f:
                f.writelines(json.dumps(e) + "\n" for e in conflicts)
    return {"written": written, "unchanged": unchanged, "conflicts": len(conflicts),
            "latest": pd.Timestamp(kept[-1][0], unit="ms", tz="UTC") if kept else None}


def gaps(venue: str, pair: str, root: str | Path | None = None,
         period: pd.Timedelta | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Missing periods between the first and last snapshot kept, as (first missing, last missing)."""
    from sleeve_fund.venues import venue as venue_profile

    period = period or pd.Timedelta(minutes=venue_profile(venue).open_interest_minutes)
    times = [r[0] for r in _kept(_path(venue, pair, root))]
    step = int(period / pd.Timedelta(milliseconds=1))
    out = []
    for a, b in zip(times, times[1:]):
        if b - a > step:
            out.append((pd.Timestamp(a + step, unit="ms", tz="UTC"), pd.Timestamp(b - step, unit="ms", tz="UTC")))
    return out


def at_risk(venue: str, pair: str, root: str | Path | None = None, now: pd.Timestamp | None = None) -> str | None:
    """Why this instrument's open interest is close to being lost for good, or None. The venue keeps 30 days, so
    once the newest snapshot kept is a week old the collector is failing and must be fixed while it can catch up."""
    newest = latest(venue, pair, root)
    now = now or pd.Timestamp.now(tz="UTC")
    if newest is not None and now - newest > AT_RISK:
        lost = newest + pd.Timedelta(days=30)
        return (f"{venue.upper()} {pair}: open interest last kept {newest:%Y-%m-%d %H:%M} UTC; the venue drops "
                f"snapshots after 30 days, so they start being lost for good from {lost:%Y-%m-%d %H:%M} UTC")
    return None
