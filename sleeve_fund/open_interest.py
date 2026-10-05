"""Open interest of a perpetual venue's instruments, kept beside their price history (DA-10).

The venue publishes open interest as a snapshot at the end of each fixed period (5 minutes on Binance), and keeps
only a recent window of them (30 days on Binance). What it no longer publishes can't be fetched again, so this is
a record, not a cache: snapshots are only ever added. One JSON file per venue and instrument in the history store's
directory, topped up from the last snapshot kept. A snapshot already kept is never replaced: a later copy with
different numbers is written to the instrument's provenance.jsonl, as the hub's bars are. A missed period stays a
hole, and gaps() reports it. No strategy reads this yet; it is collected so the history exists when one does.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from sleeve_fund.history import DEFAULT_ROOT

COLUMNS = ("contracts", "notional")
_lock = threading.Lock()


def _path(venue: str, pair: str, root: str | Path | None = None) -> Path:
    return Path(root or DEFAULT_ROOT) / venue.upper() / pair.upper().replace("/", "-") / "open_interest.json"


def _kept(path: Path) -> list[list]:
    return json.loads(path.read_text())["snapshots"] if path.exists() else []


def snapshots(venue: str, pair: str, root: str | Path | None = None) -> pd.DataFrame:
    """Every snapshot kept, indexed by its time (UTC, the end of its period), oldest first."""
    rows = _kept(_path(venue, pair, root))
    index = pd.DatetimeIndex(pd.to_datetime([r[0] for r in rows], unit="ms", utc=True), name="timestamp")
    return pd.DataFrame([r[1:] for r in rows], index=index, columns=list(COLUMNS), dtype=float)


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
        by_time = {r[0]: r for r in kept}
        start = kept[-1][0] + 1 if kept else 0
        for _ in range(max_pages):
            page = loader(pair, start)
            for t, *values in page:
                have = by_time.get(int(t))
                if have is None:
                    row = [int(t), *map(float, values)]
                    kept.append(row)
                    by_time[int(t)] = row
                    written += 1
                elif have[1:] == [float(v) for v in values]:
                    unchanged += 1
                else:
                    conflicts.append({"kind": "conflict", "series": "open_interest",
                                      "at": pd.Timestamp.now(tz="UTC").isoformat(), "source": "venue_rest",
                                      "snapshot": pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat(),
                                      "stored": have[1:], "offered": [float(v) for v in values]})
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
