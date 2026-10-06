"""One price-history store for every venue, read by research and the backtest page.

Bars are 1-minute OHLCV, stored per venue and instrument as one compressed .npz per month
(no extra dependencies), so a refresh only rewrites the current month. Longer bars are
resampled on read. Each series also records its coverage: the span the venue's loader has
fetched without a break. Inside that span a minute with no trades is a flat bar with zero
volume, not a gap; outside it there is no data, and readers are told so.

Times: a stored bar is stamped at its OPEN (the convention venues publish); read() returns bars
stamped at their CLOSE, as the engine needs, so a bar is never known before it has finished.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
DEFAULT_ROOT = Path(os.environ.get("HISTORY_DIR", Path(__file__).resolve().parent.parent / "data" / "history"))
_lock = threading.Lock()


@dataclass(frozen=True)
class Coverage:
    first: pd.Timestamp  # open time of the first stored minute
    last: pd.Timestamp  # open time of the last stored minute (from the REST loader it may still be forming)
    cursor: str  # the loader's resume point, opaque to the store
    closed: pd.Timestamp | None = None  # open time of the newest minute known closed (append_bars), if any
    # open time of the REST loader's newest stored minute, which may be a part bar, until a complete bar for it
    # is stored (by the loader's next page or by the hub); None when no stored minute is still forming
    forming: pd.Timestamp | None = None

    def as_dict(self) -> dict:
        out = {"first": self.first.isoformat(), "last": self.last.isoformat(), "cursor": self.cursor,
               "forming": self.forming.isoformat() if self.forming is not None else None}
        if self.closed is not None:
            out["closed"] = self.closed.isoformat()
        return out


@dataclass(frozen=True)
class AppendResult:
    """What append_bars did with a batch, so the caller can raise events."""
    written: int  # minutes stored for the first time (or replacing a minute that was still forming)
    unchanged: int  # minutes already stored with the same values: idempotent repeats
    conflicts: int  # minutes already stored with different values: the stored bar kept, the difference recorded


class HistoryStore:
    def __init__(self, root: str | Path | None = None, clock=None) -> None:
        self.root = Path(root or DEFAULT_ROOT)
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))  # tests pass their own

    def _dir(self, venue: str, pair: str) -> Path:
        return self.root / venue.upper() / pair.upper().replace("/", "-")

    def series(self) -> list[tuple[str, str]]:
        """Every (venue, pair) with data."""
        if not self.root.exists():
            return []
        return sorted((v.name, p.name.replace("-", "/")) for v in self.root.iterdir() if v.is_dir()
                      for p in v.iterdir() if (p / "coverage.json").exists())

    def coverage(self, venue: str, pair: str) -> Coverage | None:
        path = self._dir(venue, pair) / "coverage.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        closed = pd.Timestamp(raw["closed"]) if raw.get("closed") else None
        last = pd.Timestamp(raw["last"])
        if "forming" in raw:
            forming = pd.Timestamp(raw["forming"]) if raw["forming"] else None
        else:  # written before the forming minute was recorded: the loader's newest, unless the hub has passed it
            forming = last if closed is None or last > closed else None
        return Coverage(pd.Timestamp(raw["first"]), last, raw.get("cursor", ""), closed, forming)

    def append(self, venue: str, pair: str, minutes: pd.DataFrame, cursor: str, merge: bool = False) -> Coverage:
        """Add 1-minute bars (indexed by open time, UTC) and move the loader's cursor.

        The loader vouches that nothing is missing between the stored end and these bars (it
        resumes from its cursor), so minutes in between had no trades and are stored flat. An
        overlapping minute is replaced, or with merge=True (bars built from trades that continue
        exactly where the last page stopped) combined: first open, highest high, lowest low,
        last close, volumes added. The last stored minute may still be forming; read() leaves it out.
        Once the hub writes the series (append_bars), its closed minutes are never overwritten: see below."""
        if minutes.empty:
            raise ValueError("no bars to append")
        df = _check_minutes(minutes)
        d = self._dir(venue, pair)
        with _writing(d):
            cov = self.coverage(venue, pair)
            if cov is not None and cov.closed is not None:
                # The hub writes this series too (append_bars). Minutes up to its newest closed one are
                # first-wins, as in append_bars: the loader fills holes but never overwrites a closed bar, and
                # the span to the hub's end is not the loader's to vouch for, so it is not filled flat.
                done = df[df.index <= cov.closed]
                stored = done.index[:0]
                if not done.empty:
                    stored = _write_first_wins(d, done, cov, "loader")[3]
                # Past the hub's end the loader writes as before. Those minutes, apart from its newest (forming),
                # are the venue's own closed candles, so they are first-wins too: a hub live bar arriving later for
                # one of them is recorded as a conflict and the venue's candle kept. Intended: the REST candle is
                # the venue's record of the minute, and the parity report counts such differences.
                df = df[df.index > cov.closed]
                if df.empty:
                    # the page's newest minute is still the loader's own part bar only if it was stored here
                    forming = done.index[-1] if done.index[-1] in stored else None
                    new = Coverage(cov.first, cov.last, cursor, cov.closed, forming)
                    _write_coverage(d, new)
                    return new
            elif cov is not None and df.index[0] > cov.last:  # quiet minutes since the stored end
                prev = self._month(d, cov.last).loc[[cov.last]]
                df = _fill_quiet(pd.concat([prev, df])).iloc[1:]
            if cov is not None and merge and df.index[0] == cov.last and cov.closed is None:
                old = self._month(d, cov.last).loc[cov.last]
                first = df.iloc[0]
                df.iloc[0] = [old["open"], max(old["high"], first["high"]), min(old["low"], first["low"]),
                              first["close"], old["volume"] + first["volume"]]
            for month, chunk in df.groupby(df.index.strftime("%Y-%m")):
                path = d / f"{month}.npz"
                if path.exists():
                    old = _load(path)
                    chunk = pd.concat([old[~old.index.isin(chunk.index)], chunk]).sort_index()
                _save(path, chunk)
            first = cov.first if cov is not None else df.index[0]
            new = Coverage(first, max(df.index[-1], cov.last if cov else df.index[-1]), cursor,
                           cov.closed if cov else None, df.index[-1])
            _write_coverage(d, new)
            return new

    def append_bars(self, venue: str, pair: str, bars, source: str) -> AppendResult:
        """Store closed 1-minute bars from the market data hub. The hub's write path.

        bars: (open_time_ns, open, high, low, close, volume) rows, each a CLOSED minute stamped at its open.
        source: "live" (built from the hub's own stream) or "refill" (fetched from the venue's REST API after
        a reconnect).

        A bar's identity is (venue, instrument, open minute). Writes are idempotent on it: the same bar
        again changes nothing. The first complete bar for a minute wins; a later one with different values
        (a refill over a live bar, say) never overwrites it, the difference is written to provenance.jsonl.
        A minute stored but still forming (the REST loader's newest) is replaced. Refilled minutes are
        recorded there too. Unlike append(), missing minutes between batches are NOT filled flat: a hole
        means the hub was not listening, so it stays a hole, gaps() reports it, and a refill closes it."""
        if source not in ("live", "refill"):
            raise ValueError(f"source must be 'live' or 'refill', not {source!r}")
        rows = list(bars)
        if not rows:
            return AppendResult(0, 0, 0)
        df = _bars_frame(rows)
        d = self._dir(venue, pair)
        with _writing(d):
            # Defence in depth (QA F4): a minute that has not closed yet by this clock is not stored; the hub
            # sends only closed bars, so one here is a bug upstream, recorded rather than trusted.
            open_by = self.clock() + CLOCK_SKEW - pd.Timedelta(minutes=1)
            early = df[df.index > open_by]
            if not early.empty:
                _record(d, [{"kind": "refused", "at": self.clock().isoformat(), "source": source,
                             "reason": "minute not closed yet", "minutes": [t.isoformat() for t in early.index]}])
                df = df[df.index <= open_by]
                if df.empty:
                    return AppendResult(0, 0, 0)
            cov = self.coverage(venue, pair)
            written, unchanged, conflicts, _ = _write_first_wins(d, df, cov, source)
            lo, hi = df.index[0], df.index[-1]
            # a complete bar for the loader's part bar replaces it (_forming), so it is no longer forming
            forming = cov.forming if cov and cov.forming is not None and cov.forming not in df.index else None
            # On a store only the REST loader wrote, everything before its forming minute is already closed, so
            # an old refill must not pull `closed` back to its own minute (QA F5).
            if cov is None:
                closed = hi
            elif cov.closed is not None:
                closed = max(hi, cov.closed)
            else:
                closed = max(hi, cov.last - pd.Timedelta(minutes=1) if cov.forming == cov.last else cov.last)
            new_cov = Coverage(min(lo, cov.first) if cov else lo, max(hi, cov.last) if cov else hi,
                               cov.cursor if cov else "", closed, forming)
            _write_coverage(d, new_cov)
        return AppendResult(written, unchanged, conflicts)

    def provenance(self, venue: str, pair: str) -> list[dict]:
        """The hub's refill and conflict records for one series, oldest first."""
        path = self._dir(venue, pair) / "provenance.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _month(self, d: Path, ts: pd.Timestamp) -> pd.DataFrame:
        return _load(d / f"{ts.strftime('%Y-%m')}.npz")

    def read(self, venue: str, pair: str, minutes: int = 1440, start=None, end=None) -> pd.DataFrame:
        """Bars of `minutes` length, stamped at their CLOSE (UTC), complete bars only."""
        cov = self.coverage(venue, pair)
        if cov is None:
            raise KeyError(f"no stored history for {venue} {pair}")
        d = self._dir(venue, pair)
        lo = pd.Timestamp(start, tz="UTC") if start is not None and pd.Timestamp(start).tzinfo is None else start
        hi = pd.Timestamp(end, tz="UTC") if end is not None and pd.Timestamp(end).tzinfo is None else end
        # Bars that divide a day never span two months, so each month is resampled on its own and
        # years of minutes are never held in memory at once.
        by_month = minutes > 1 and 1440 % minutes == 0
        parts = []
        for path in _month_files(d):
            month = pd.Timestamp(path.stem + "-01", tz="UTC")
            if lo is not None and month + pd.offsets.MonthBegin(1) <= lo:
                continue
            if hi is not None and month > hi:
                continue
            part = _load(path)
            part = part[(part.index >= cov.first) & (part.index <= cov.last) & ~_forming(part.index, cov)]
            parts.append(_resample(part, minutes) if by_month else part)
        if not parts:
            return pd.DataFrame(columns=OHLCV)
        bars = pd.concat(parts).sort_index()
        if minutes > 1 and not by_month:
            bars = _resample(bars, minutes)
        bars = bars.set_axis(bars.index + pd.Timedelta(minutes=minutes))
        bars.index.name = "timestamp"
        if lo is not None:
            bars = bars[bars.index > lo]
        if hi is not None:
            bars = bars[bars.index <= hi]
        return bars

    def report(self, venue: str, pair: str) -> dict:
        """What is stored and whether it is whole: span, bar count, gaps and duplicates."""
        cov = self.coverage(venue, pair)
        if cov is None:
            return {"venue": venue, "pair": pair, "stored": False}
        df = pd.concat([_load(p) for p in _month_files(self._dir(venue, pair))]).sort_index()
        df = df[(df.index >= cov.first) & (df.index <= cov.last)]
        expected = int((cov.last - cov.first) / pd.Timedelta("1min")) + 1
        return {"venue": venue, "pair": pair, "stored": True, "first": cov.first, "last": cov.last,
                "minutes": len(df), "missing": expected - df.index.nunique(),
                "duplicates": int(df.index.duplicated().sum()),
                "quiet_over_an_hour": quiet_runs(df[~df.index.duplicated()], 1)}

    def gaps(self, venue: str, pair: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Runs of minutes missing inside the coverage, as (first, last) missing open times. append() fills
        quiet minutes, so for the REST loader a run is a write that never finished, not a market with no
        trades; for the hub it is a stretch it was not listening, until a refill closes it. Each
        month's file is read once per version (its size and time), so the current month is the only one
        re-read as the collector adds to it."""
        cov = self.coverage(venue, pair)
        if cov is None:
            return []
        lo, hi = int(cov.first.timestamp()), int(cov.last.timestamp())
        runs, seen = [], lo - 60  # the last minute held so far
        for path in _month_files(self._dir(venue, pair)):
            first, last, holes = _month_minutes(path)
            if first > seen + 60:
                runs.append((seen + 60, first - 60))
            runs += holes
            seen = max(seen, last)
        if hi > seen:
            runs.append((seen + 60, hi))
        if cov.forming is not None and cov.closed is not None and cov.forming < cov.closed:
            # the loader's part bar, which the hub has since passed without storing a complete one: a hole
            # until a refill closes it
            f = int(cov.forming.timestamp())
            runs = sorted(runs + [(f, f)])
        clipped = [(max(a, lo), min(b, hi)) for a, b in runs if b >= lo and a <= hi]
        return [(pd.Timestamp(a, unit="s", tz="UTC"), pd.Timestamp(b, unit="s", tz="UTC")) for a, b in clipped]


_months: dict[Path, tuple[tuple[int, int, int], tuple[int, int, list[tuple[int, int]]]]] = {}


def _month_minutes(path: Path) -> tuple[int, int, list[tuple[int, int]]]:
    """A month file's first and last minute (epoch seconds) and the runs missing between them."""
    stat = path.stat()
    key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)  # _save renames a new file in: a new inode (QA F6)
    hit = _months.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    with np.load(path) as z:
        t = np.unique(z["t"])
    steps = np.nonzero(np.diff(t) > 60)[0]
    info = (int(t[0]), int(t[-1]), [(int(t[i]) + 60, int(t[i + 1]) - 60) for i in steps])
    _months[path] = (key, info)
    return info


def quiet_runs(bars: pd.DataFrame, minutes: int, at_least: int = 60) -> dict:
    """Stretches of bars with no trades lasting `at_least` minutes or more. The store keeps them
    flat at the last price, because the venue reported no trades; a long one may be a venue outage,
    so it is reported rather than hidden. Bars are stamped at their close."""
    if bars.empty:
        return {"count": 0, "longest_minutes": 0, "longest_end": None}
    quiet = (bars["volume"] <= 0).to_numpy()
    run_id = np.cumsum(~quiet)  # each quiet stretch shares the id of the traded bar before it
    lengths = pd.Series(quiet.astype(int)).groupby(run_id).sum()
    long = lengths[lengths * minutes >= at_least]
    if long.empty:
        return {"count": 0, "longest_minutes": 0, "longest_end": None}
    rid = long.idxmax()
    end = bars.index[(run_id == rid) & quiet][-1]
    return {"count": int(len(long)), "longest_minutes": int(long.max() * minutes), "longest_end": end}


def trades_to_minutes(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades (index = trade time UTC, columns price, volume) -> 1-minute OHLCV by open time.
    Only minutes with trades come out; the store fills quiet minutes inside its coverage."""
    if trades.empty:
        return pd.DataFrame(columns=OHLCV)
    g = trades.groupby(trades.index.floor("1min"))
    out = pd.DataFrame({"open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
                        "close": g["price"].last(), "volume": g["volume"].sum()})
    out.index.name = "timestamp"
    return out


def _fill_quiet(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes with no trades: flat at the last close, zero volume."""
    full = pd.date_range(df.index[0], df.index[-1], freq="1min")
    out = df.reindex(full)
    close = out["close"].ffill()
    for col in ("open", "high", "low"):
        out[col] = out[col].fillna(close)
    out["close"] = close
    out["volume"] = out["volume"].fillna(0.0)
    out.index.name = "timestamp"
    return out


def _check_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        raise ValueError("bar times must be UTC")
    if (df.index != df.index.floor("1min")).any():
        raise ValueError("bars must be stamped on whole minutes (their open time)")
    df = df[~df.index.duplicated(keep="last")].sort_index()[OHLCV].astype(float)
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any() or (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise ValueError("bars with impossible prices or volume")
    return _fill_quiet(df)


def _forming(index: pd.DatetimeIndex, cov: Coverage | None) -> np.ndarray:
    """Stored minutes that may still be forming: the REST loader's newest minute, until a complete bar for it
    is stored. Recorded rather than inferred from the hub's progress: a hub bar for a LATER minute says nothing
    about this one (its refill may have failed), so the part bar is never served as complete (QA P1-H1)."""
    if cov is None or cov.forming is None:
        return np.zeros(len(index), dtype=bool)
    return np.asarray(index == cov.forming)


def _bars_frame(rows: list) -> pd.DataFrame:
    """Hub bar rows -> a checked frame of closed minutes by open time. Not quiet-filled. The open time is
    checked as an integer: through a float, a stamp a few hundred ns off the minute would round onto it."""
    if any(len(r) != 6 for r in rows):
        raise ValueError("each bar is (open_time_ns, open, high, low, close, volume)")
    if any(isinstance(r[0], float) and not r[0].is_integer() for r in rows):
        raise ValueError("bars must be stamped on whole minutes (their open time, in ns)")
    t = np.array([int(r[0]) for r in rows], dtype=np.int64)
    if (t % 60_000_000_000).any():
        raise ValueError("bars must be stamped on whole minutes (their open time, in ns)")
    values = np.array([r[1:] for r in rows], dtype=float)
    df = pd.DataFrame(values, columns=OHLCV, index=pd.to_datetime(t, unit="ns", utc=True))
    df.index.name = "timestamp"
    if df.index.duplicated().any():
        raise ValueError("the same minute twice in one batch")
    if not np.isfinite(values).all():
        raise ValueError("bars with missing values")
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any() or (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise ValueError("bars with impossible prices or volume")
    return df.sort_index()


def _write_first_wins(d: Path, df: pd.DataFrame, cov: Coverage | None,
                      source: str) -> tuple[int, int, int, pd.DatetimeIndex]:
    """Write minutes nobody has stored yet (or that were still forming); keep every stored closed minute, and
    record in provenance.jsonl each offered minute that differs from it and, for a refill, what was filled.
    Returns (written, unchanged, conflicts, the minutes written). The caller holds _lock and writes the coverage."""
    written = unchanged = 0
    stored = []
    log: list[dict] = []
    now = pd.Timestamp.now(tz="UTC").isoformat()
    for month, chunk in df.groupby(df.index.strftime("%Y-%m")):
        path = d / f"{month}.npz"
        old = _load(path) if path.exists() else df.iloc[:0]
        old = old[~old.index.duplicated(keep="last")]
        # Minutes past coverage.last are a write whose coverage never landed (a crash between the two): not held,
        # so the restart's refill rewrites them. They were never served (read() and gaps() stop at last).
        mine = old.index[~_forming(old.index, cov)]
        mine = mine[mine <= cov.last] if cov is not None else mine[:0]
        held = chunk.index.isin(mine)
        for ts in chunk.index[held]:
            have, offer = old.loc[ts, OHLCV].to_numpy(float), chunk.loc[ts, OHLCV].to_numpy(float)
            if np.allclose(have, offer, rtol=1e-12, atol=0.0):  # "the same bar": equal to 12 significant figures
                unchanged += 1
            else:
                log.append({"kind": "conflict", "at": now, "source": source, "minute": ts.isoformat(),
                            "stored": have.tolist(), "offered": offer.tolist()})
        new = chunk[~held]
        if new.empty:
            continue
        written += len(new)
        stored.append(new.index)
        _save(path, pd.concat([old[~old.index.isin(new.index)], new]).sort_index())
    done = stored[0].append(stored[1:]) if stored else df.index[:0]
    if source != "live" and written:
        log.append({"kind": "refill", "at": now, "source": source, "first": done.min().isoformat(),
                    "last": done.max().isoformat(), "minutes": written})
    conflicts = sum(e["kind"] == "conflict" for e in log)
    if conflicts:
        # The same difference offered again is one difference, recorded once (QA F7): the count is of minutes.
        seen = {(e["minute"], tuple(e["stored"]), tuple(e["offered"])) for e in _provenance(d) if e["kind"] == "conflict"}
        log = [e for e in log if e["kind"] != "conflict" or (e["minute"], tuple(e["stored"]), tuple(e["offered"])) not in seen]
    _record(d, log)
    return written, unchanged, conflicts, done


CLOCK_SKEW = pd.Timedelta(seconds=2)  # how far ahead of this clock the venue's may run when a minute closes
_MONTH = re.compile(r"\d{4}-\d{2}\.npz")


def _month_files(d: Path) -> list[Path]:
    """The month files of a series, oldest first: never a writer's temporary file (QA F2)."""
    return sorted(p for p in d.glob("*.npz") if _MONTH.fullmatch(p.name)) if d.exists() else []


@contextlib.contextmanager
def _writing(d: Path):
    """One writer per series at a time, across threads AND processes (QA F3): a manual `history refresh` while
    the hub runs waits for the hub's write to finish instead of interleaving with it."""
    with _lock:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / ".write.lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _durable_replace(tmp: Path, path: Path) -> None:
    """Rename a fully written file into place, flushed to disk first so a power loss leaves the old or the new."""
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _provenance(d: Path) -> list[dict]:
    path = d / "provenance.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _record(d: Path, entries: list[dict]) -> None:
    if entries:
        with open(d / "provenance.jsonl", "a") as f:
            f.writelines(json.dumps(e) + "\n" for e in entries)


def _write_coverage(d: Path, cov: Coverage) -> None:
    tmp = d / f".coverage.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(cov.as_dict()))
    _durable_replace(tmp, d / "coverage.json")


def _save(path: Path, df: pd.DataFrame) -> None:
    t = (df.index.as_unit("s").asi8).astype(np.int64)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")  # unique: never another writer's
    try:
        np.savez_compressed(tmp, t=t, **{c: df[c].to_numpy(float) for c in OHLCV})
        _durable_replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """1-minute bars by open time -> complete `minutes` bars by open time."""
    if minutes <= 1:
        return df
    g = df.resample(f"{minutes}min", origin="epoch", label="left", closed="left")
    bars = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return bars[g["close"].count() == minutes]  # complete bars only: a part-day isn't a daily bar


def _load(path: Path) -> pd.DataFrame:
    with np.load(path) as z:
        idx = pd.to_datetime(z["t"], unit="s", utc=True)
        df = pd.DataFrame({c: z[c] for c in OHLCV}, index=idx)
    df.index.name = "timestamp"
    return df


def refresh(store: HistoryStore, profile, pair: str, max_pages: int = 1_000_000, sleep=None, log=print,
            since: pd.Timestamp | None = None) -> dict:
    """Bring one instrument's stored history up to date from its venue, resuming from the cursor.
    The venue profile's minute_loader returns (1-minute bars, next cursor, caught_up). since: where a
    first backfill starts, when the venue can start mid-history (else at the instrument's listing)."""
    import time

    if profile.minute_loader is None:
        raise ValueError(f"{profile.label} has no history loader")
    pages = 0
    cov = store.coverage(profile.name, pair)
    cursor = cov.cursor if cov else (
        profile.minute_cursor_at(pd.Timestamp(since)) if since is not None and profile.minute_cursor_at else "")
    while pages < max_pages:
        bars, cursor_next, caught_up = profile.minute_loader(pair, cursor)
        pages += 1
        if not bars.empty:
            cov = store.append(profile.name, pair, bars, cursor_next, merge=profile.merge_minutes)
        cursor = cursor_next
        if caught_up:
            break
        if pages % 500 == 0 and cov is not None:
            log(f"{profile.name} {pair}: stored to {cov.last:%Y-%m-%d %H:%M}")
        (sleep or time.sleep)(profile.request_interval)
    return {"pair": pair, "pages": pages, "last": cov.last if cov else None}


# Always kept, so research has them before any sleeve trades them; sleeves' own instruments are added. A venue
# whose instruments are named differently (USDT-quoted perpetuals) lists its own (VenueProfile.core_pairs).
CORE_PAIRS = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "SUI/USD")


# Where the backfill starts for an instrument the PM asks Research for: enough for the default study
# (holdout, training and test windows) without fetching every trade since the instrument listed.
REQUEST_YEARS = 5


def _pairs_in_use(venue: str, store=None) -> list[tuple[str, pd.Timestamp | None]]:
    """(instrument, where a first backfill starts) for the core list, every strategy's instrument and
    every instrument the PM asked Research for."""
    from sleeve_fund.venues import VENUES

    profile = VENUES.get(venue.upper())
    pairs: dict[str, pd.Timestamp | None] = {p: None for p in (profile and profile.core_pairs) or CORE_PAIRS}
    try:
        from sleeve_fund.paper.config import from_store
        from sleeve_fund.store import Store

        store = store or Store()
        for s in store.sleeves():
            cfg = from_store(s)
            if cfg.venue == venue:
                pairs.setdefault(cfg.instrument, None)
        for r in store.history_requests(venue):
            pairs.setdefault(r["instrument"], pd.Timestamp(r["since"]))
    except Exception as exc:  # noqa: BLE001 - no database (e.g. locally): the core list still loads
        print(f"could not read sleeves or history requests: {exc!r}")
    return list(pairs.items())


def _refresh_funding(profile, pair: str, root, since) -> None:
    """A perpetual venue's settled funding for the instrument, kept beside its prices (sleeve_fund.funding)."""
    if profile.funding_loader is None:
        return
    from sleeve_fund import funding

    try:
        funding.refresh(profile.name, pair, root=root, since=since)
    except Exception as exc:  # noqa: BLE001 - the prices are stored; funding catches up on the next pass
        print(f"{profile.name} {pair}: funding refresh failed: {exc!r}")


def unwritable(path: Path) -> str | None:
    """Why this process can't write under `path`, or None if it can. Checked once at start, so a store
    it may not write says so in one line instead of failing every instrument on every pass."""
    probe = path / f".write-check-{os.getpid()}"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("")
        probe.unlink()
        return None
    except OSError as exc:
        owner = next((p for p in (path, *path.parents) if p.exists()), path)
        try:
            owned = f"owned by uid {owner.stat().st_uid}"
        except OSError:
            owned = "owner unknown"
        return f"cannot write {path} as uid {os.getuid()} ({owner} is {owned}): {exc!r}"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import time

    from sleeve_fund.venues import venue as venue_profile

    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.history", description="Venue price-history store")
    ap.add_argument("--root", default=None, help=f"store directory (default {DEFAULT_ROOT})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh", help="bring instruments up to date once")
    r.add_argument("pairs", nargs="*", help="instruments, e.g. BTC/USD (default: core list plus sleeves')")
    r.add_argument("--venue", default=None)
    run = sub.add_parser("run", help="keep every instrument in use up to date, forever")
    run.add_argument("--venue", default=None)
    run.add_argument("--pages", type=int, default=200, help="pages per instrument per pass, so all progress")
    run.add_argument("--idle", type=float, default=900, help="seconds to wait once everything is current")
    rep = sub.add_parser("report", help="what is stored, and any gaps or duplicates")
    rep.add_argument("--venue", default=None)
    args = ap.parse_args(argv)

    store = HistoryStore(args.root)
    profile = venue_profile(args.venue)
    if args.cmd in ("refresh", "run") and (problem := unwritable(store.root / profile.name.upper())):
        print(f"{profile.name}: history store can't start: {problem}")
        return 2
    if args.cmd == "report":
        for v, pair in store.series():
            if v == profile.name:
                print(store.report(v, pair))
        return 0
    if args.cmd == "refresh":
        for pair, since in [(p, None) for p in args.pairs] or _pairs_in_use(profile.name):
            print(refresh(store, profile, pair, since=since))
            _refresh_funding(profile, pair, args.root, since)
        return 0
    while True:  # run: round-robin so a long backfill on one instrument doesn't starve the others
        behind = False
        for pair, since in _pairs_in_use(profile.name):
            try:
                out = refresh(store, profile, pair, max_pages=args.pages, since=since)
                behind |= out["pages"] >= args.pages
                _refresh_funding(profile, pair, args.root, since)
            except Exception as exc:  # noqa: BLE001 - one bad pair or a venue hiccup must not stop the rest
                print(f"{profile.name} {pair}: refresh failed: {exc!r}")
        if not behind:
            time.sleep(args.idle)


if __name__ == "__main__":
    raise SystemExit(main())
