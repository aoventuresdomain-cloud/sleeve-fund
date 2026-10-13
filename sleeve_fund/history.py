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
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sleeve_fund import bars as bar_rule

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
    replaced: int = 0  # hub bars replaced by the venue's own candle (P1-1-CANON), each recorded with both values


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
                # The hub writes this series too (append_bars). Up to its newest closed minute the loader fills
                # holes, and the span to the hub's end is not the loader's to vouch for, so it is not filled flat.
                # P1-1-CANON: the venue's candle is the record of a minute, so it replaces a differing stored bar
                # (the hub's live build, provisional) once the minute is settled. Never the page's newest minute,
                # which may still be forming: the loader resumes on it, so the next page offers it again, closed.
                # A trade-built page (merge) splits minutes at its ends, so there the stored bar is kept.
                done = df[df.index <= cov.closed]
                stored = done.index[:0]
                if not done.empty:
                    canon = None if merge else self._settled(done.index[done.index < df.index[-1]])
                    stored = _write_first_wins(d, done, cov, "loader", canon)[3]
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
            # A refill is the venue's own closed candle (hub.relay.refill_bars), so it replaces a differing live
            # bar once settled (P1-1-CANON); a live bar never replaces anything.
            canon = self._settled(df.index) if source == "refill" else None
            written, unchanged, conflicts, _, replaced = _write_first_wins(d, df, cov, source, canon)
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
        return AppendResult(written, unchanged, conflicts, replaced)

    def canonise(self, venue: str, pair: str, minutes: pd.DataFrame, dry_run: bool = False) -> AppendResult:
        """P1-1-CANON backfill: make the stored minutes the venue's own closed 1-minute candles (indexed by open
        time, UTC; the caller leaves out a candle that may still be forming). Only minutes inside the stored span
        up to the newest closed one are touched: a differing bar is replaced and recorded (kind "replaced"), a
        hole is filled and recorded as a refill. Coverage and the loader's cursor do not move. dry_run: count what
        would be replaced and filled, and write nothing."""
        if minutes.empty:
            return AppendResult(0, 0, 0)
        df = _check_minutes(minutes)
        d = self._dir(venue, pair)
        with _writing(d):
            cov = self.coverage(venue, pair)
            if cov is None:
                return AppendResult(0, 0, 0)
            top = _canon_top(cov)
            df = df[(df.index >= cov.first) & (df.index <= top)]
            if df.empty:
                return AppendResult(0, 0, 0)
            written, unchanged, conflicts, done, replaced = _write_first_wins(d, df, cov, "canon",
                                                                              self._settled(df.index), dry_run)
            if not dry_run and cov.forming is not None and cov.forming in done:  # its complete candle is stored now
                _write_coverage(d, Coverage(cov.first, cov.last, cov.cursor, cov.closed, None))
        return AppendResult(written, unchanged, conflicts, replaced)

    def _settled(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """The minutes (by open time) closed at least CANON_SETTLE ago by this clock: the venue's candle for them
        is final, so it may replace a stored bar."""
        return index[index + pd.Timedelta(minutes=1) + CANON_SETTLE <= self.clock()]

    def provenance(self, venue: str, pair: str) -> list[dict]:
        """The hub's refill and conflict records for one series, oldest first."""
        path = self._dir(venue, pair) / "provenance.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _month(self, d: Path, ts: pd.Timestamp) -> pd.DataFrame:
        return _load(d / f"{ts.strftime('%Y-%m')}.npz")

    def read(self, venue: str, pair: str, minutes: int = 1440, start=None, end=None) -> pd.DataFrame:
        """Bars of `minutes` length, stamped at their CLOSE (UTC), built by the one bar-build rule
        (sleeve_fund.bars): columns OHLCV, missing and degraded; part bars at either end of the coverage left out."""
        cov = self.coverage(venue, pair)
        if cov is None:
            raise KeyError(f"no stored history for {venue} {pair}")
        d = self._dir(venue, pair)
        lo = pd.Timestamp(start, tz="UTC") if start is not None and pd.Timestamp(start).tzinfo is None else start
        hi = pd.Timestamp(end, tz="UTC") if end is not None and pd.Timestamp(end).tzinfo is None else end
        # Bars that divide a day never span two months, so each month is resampled on its own and
        # years of minutes are never held in memory at once.
        by_month = minutes > 1 and 1440 % minutes == 0
        # Just past the last minute that is surely closed: a longer bar ending after it is a part bar.
        stop = cov.last + pd.Timedelta(minutes=0 if _forming(pd.DatetimeIndex([cov.last]), cov)[0] else 1)
        parts = []
        for path in _month_files(d):
            month = pd.Timestamp(path.stem + "-01", tz="UTC")
            if lo is not None and month + pd.offsets.MonthBegin(1) <= lo:
                continue
            if hi is not None and month > hi:
                continue
            part = _load(path)
            part = part[(part.index >= cov.first) & (part.index <= cov.last) & ~_forming(part.index, cov)]
            parts.append(_resample(part, minutes, cov.first, stop) if by_month else part)
        if not parts:
            return pd.DataFrame(columns=bar_rule.COLUMNS)
        bars = pd.concat(parts).sort_index()
        if not by_month:  # longer bars that don't divide a day, and 1-minute bars (given missing 0)
            bars = _resample(bars, minutes, cov.first, stop)
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


def _write_first_wins(d: Path, df: pd.DataFrame, cov: Coverage | None, source: str,
                      canon: pd.DatetimeIndex | None = None,
                      dry_run: bool = False) -> tuple[int, int, int, pd.DatetimeIndex, int]:
    """Write minutes nobody has stored yet (or that were still forming); keep every stored closed minute, and
    record in provenance.jsonl each offered minute that differs from it and, for a refill, what was filled.
    canon: offered minutes that are the venue's own settled candles (P1-1-CANON). One of them that differs from
    the stored bar replaces it, recorded as kind "replaced" with both values, so the hub's live bar is never lost.
    Returns (written, unchanged, conflicts, the minutes written, replaced). The caller holds _lock and writes the
    coverage. dry_run: count only; nothing is saved or recorded."""
    written = unchanged = replaced = 0
    canon = canon if canon is not None else df.index[:0]
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
        swap: list[pd.Timestamp] = []
        for ts in chunk.index[held]:
            have, offer = old.loc[ts, OHLCV].to_numpy(float), chunk.loc[ts, OHLCV].to_numpy(float)
            if np.allclose(have, offer, rtol=1e-12, atol=0.0):  # "the same bar": equal to 12 significant figures
                unchanged += 1
            elif ts in canon:
                log.append({"kind": "replaced", "at": now, "source": source, "minute": ts.isoformat(),
                            "stored": have.tolist(), "offered": offer.tolist()})
                swap.append(ts)
            else:
                log.append({"kind": "conflict", "at": now, "source": source, "minute": ts.isoformat(),
                            "stored": have.tolist(), "offered": offer.tolist()})
        new = chunk[~held]
        if new.empty and not swap:
            continue
        written += len(new)
        replaced += len(swap)
        if not new.empty:
            stored.append(new.index)
        put = pd.concat([new, chunk.loc[swap]])
        if dry_run:
            continue
        _save(path, pd.concat([old[~old.index.isin(put.index)], put]).sort_index())
    done = stored[0].append(stored[1:]) if stored else df.index[:0]
    if source != "live" and written:
        log.append({"kind": "refill", "at": now, "source": source, "first": done.min().isoformat(),
                    "last": done.max().isoformat(), "minutes": written})
    conflicts = sum(e["kind"] == "conflict" for e in log)
    if dry_run:
        return written, unchanged, conflicts, done, replaced
    if conflicts:
        # The same difference offered again is one difference, recorded once (QA F7): the count is of minutes.
        seen = {(e["minute"], tuple(e["stored"]), tuple(e["offered"])) for e in _provenance(d) if e["kind"] == "conflict"}
        log = [e for e in log if e["kind"] != "conflict" or (e["minute"], tuple(e["stored"]), tuple(e["offered"])) not in seen]
    _record(d, log)
    return written, unchanged, conflicts, done, replaced


CLOCK_SKEW = pd.Timedelta(seconds=2)  # how far ahead of this clock the venue's may run when a minute closes
# P1-1-CANON: how long after a minute closes the venue's candle for it is taken as final and may replace a stored bar
# (late trades reach the venue's candle within a second or two; the hub's late-trade counter, parity 7 Oct).
CANON_SETTLE = pd.Timedelta(seconds=10)
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


def _resample(df: pd.DataFrame, minutes: int, first: pd.Timestamp | None = None,
              end: pd.Timestamp | None = None) -> pd.DataFrame:
    """1-minute bars by open time -> `minutes` bars by open time, by the one bar-build rule (sleeve_fund.bars,
    m13-E6): built from the minutes present, with how many are missing; part bars at either end left out."""
    return bar_rule.build_bars(df, minutes, first, end)


def _load(path: Path) -> pd.DataFrame:
    with np.load(path) as z:
        idx = pd.to_datetime(z["t"], unit="s", utc=True)
        df = pd.DataFrame({c: z[c] for c in OHLCV}, index=idx)
    df.index.name = "timestamp"
    return df


GOLDEN_HEADER = "open_utc,open,high,low,close,volume,missing,degraded"
GOLDEN_CANON_HEADER = ("minute_utc,at," + ",".join(f"stored_{c}" for c in OHLCV) + ","
                       + ",".join(f"offered_{c}" for c in OHLCV))


def _golden_window(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    end = end.tz_localize("UTC") if end.tzinfo is None else end
    if start.second or start.microsecond or end.second or end.microsecond or end <= start:
        raise ValueError("start and end must be whole minutes, start before end")
    return start, end


def export_csv(store: HistoryStore, venue: str, pair: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """GOLDEN-VENUE: the stored 1-minute candles opening in [start, end), read through the store's own read() (via
    parity.store_minutes, which shifts read()'s close stamps back to open times), as CSV with a fixed layout (open
    time UTC, floats as repr, missing and degraded 0/1) so the same candles always give the same bytes and so the same
    sha256. Read-only. A window with any minute absent, missing or degraded is not canonical and is refused rather
    than exported."""
    from sleeve_fund.parity import store_minutes  # parity imports this module

    start, end = _golden_window(start, end)
    if not store.coverage(venue, pair):
        raise KeyError(f"no stored history for {venue} {pair}")
    bars = store_minutes(store, venue, pair, start, end)
    want = pd.date_range(start, end, freq="1min", inclusive="left")
    if len(absent := want.difference(bars.index)):
        raise ValueError(f"{venue} {pair}: {len(absent)} of {len(want)} minutes absent in the window, first "
                         f"{absent[0]:%Y-%m-%dT%H:%MZ}; nothing exported")
    for flag in ("missing", "degraded"):
        if len(hit := bars.index[bars[flag].fillna(1).astype(bool).to_numpy()]):
            raise ValueError(f"{venue} {pair}: {len(hit)} minutes {flag} in the window, first "
                             f"{hit[0]:%Y-%m-%dT%H:%MZ}; the window is not canonical, nothing exported")
    lines = [GOLDEN_HEADER]
    for t, row in zip(bars.index, bars.itertuples(index=False)):
        lines.append(f"{t:%Y-%m-%dT%H:%MZ},{float(row.open)!r},{float(row.high)!r},{float(row.low)!r},"
                     f"{float(row.close)!r},{float(row.volume)!r},{int(bool(row.missing))},{int(bool(row.degraded))}")
    return "\n".join(lines) + "\n"


def export_canon_replaced(store: HistoryStore, venue: str, pair: str, start: pd.Timestamp, end: pd.Timestamp,
                          expect: int | None = None) -> str:
    """GOLDEN-VENUE provenance: for each minute opening in [start, end) that the canon backfill replaced, its latest
    "replaced" record with source "canon" (CR F224-1), as CSV: minute, when, the stored bar and the venue's offered bar
    (floats as repr). Read-only. expect: the count the canon run reported; a different count means something rewrote
    the window since, and is refused."""
    start, end = _golden_window(start, end)
    latest: dict[pd.Timestamp, dict] = {}
    for e in store.provenance(venue, pair):  # oldest first, so the last one per minute wins
        if e.get("kind") == "replaced" and e.get("source") == "canon":
            minute = pd.Timestamp(e["minute"])
            minute = minute.tz_localize("UTC") if minute.tzinfo is None else minute.tz_convert("UTC")
            if start <= minute < end:
                latest[minute] = e
    if expect is not None and len(latest) != expect:
        raise ValueError(f"{venue} {pair}: {len(latest)} canon replacements in the window, the canon run reported "
                         f"{expect}; something rewrote the window, nothing exported")
    lines = [GOLDEN_CANON_HEADER]
    for minute in sorted(latest):
        e = latest[minute]
        vals = ",".join(repr(float(v)) for v in [*e["stored"], *e["offered"]])
        lines.append(f"{minute:%Y-%m-%dT%H:%MZ},{e['at']},{vals}")
    return "\n".join(lines) + "\n"


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


def _canon_top(cov: Coverage) -> pd.Timestamp:
    """The newest stored minute the venue's candle may replace: the newest known closed, else the one before the
    loader's newest (which may still be forming). One rule for canonise() and canon() (CANON-F2)."""
    return cov.closed if cov.closed is not None else cov.last - pd.Timedelta(minutes=1)


def canon(store: HistoryStore, profile, pair: str, since: pd.Timestamp, max_pages: int = 10_000, sleep=None,
          dry_run: bool = False, until: pd.Timestamp | None = None) -> dict:
    """P1-1-CANON backfill: replace the hub's live bars stored since `since` with the venue's own 1-minute candles,
    up to the newest closed minute, each replacement recorded (HistoryStore.canonise). Reads the venue, writes only
    the history files; the loader's cursor is left alone. Not for a venue whose pages are built from trades (they
    split minutes at their ends). dry_run: report the counts, write nothing (the HoE sees them before a real run).
    until: minutes opening at or after it are left alone, so a dry run and the real run cover the same closed window
    however far apart they run (CANON-F3); an until past the newest closed minute stored is refused, so the two can't
    silently differ (F212-2). The result says the window it covered: its newest minute (top), the first and last
    minute the venue offered inside it, how many, and how many minutes the store holds in it."""
    import time

    if profile.minute_loader is None or profile.minute_cursor_at is None:
        raise ValueError(f"{profile.label} has no history loader that can start at a time")
    if profile.merge_minutes:
        raise ValueError(f"{profile.label} builds its minutes from trades; its stored bars are kept as they are")
    since = pd.Timestamp(since)
    if until is not None and pd.Timestamp(until) <= since:
        raise ValueError(f"until ({until}) must be after since ({since})")
    out = {"pair": pair, "pages": 0, "written": 0, "replaced": 0, "dry_run": dry_run,
           "until": None if until is None else pd.Timestamp(until).isoformat(), "top": None,
           "first": None, "last": None, "offered": 0, "stored": 0}
    cov = store.coverage(profile.name, pair)
    if cov is None:
        return out
    top = _canon_top(cov)
    if until is not None:
        bound = pd.Timestamp(until) - pd.Timedelta(minutes=1)
        if bound > top:  # the store can't vouch for minutes past its newest closed one, so the window would shrink
            raise ValueError(f"until ({until}) is past the newest closed minute stored ({top.isoformat()}); "
                             "a run now would cover less than the window asked for")
        top = bound
    out["top"] = top.isoformat()
    cursor, pages, written, replaced = profile.minute_cursor_at(since), 0, 0, 0
    first = last = None
    offered = 0
    while pages < max_pages:
        bars, nxt, caught_up = profile.minute_loader(pair, cursor)
        pages += 1
        page = bars.iloc[:-1]  # the page's newest may still be forming: the next page starts on it
        page = page[(page.index >= since) & (page.index <= top)]
        if len(page):
            res = store.canonise(profile.name, pair, page, dry_run=dry_run)
            written, replaced = written + res.written, replaced + res.replaced
            first = page.index[0] if first is None else first
            last, offered = page.index[-1], offered + len(page)
        if caught_up or bars.empty or nxt == cursor or bars.index[-1] > top:
            break
        cursor = nxt
        (sleep or time.sleep)(profile.request_interval)
    stored = 0
    if top >= since:
        held = store.read(profile.name, pair, 1, start=since, end=top + pd.Timedelta(minutes=1))  # close-stamped:
        # since < close <= top's close is the minutes since..top
        stored = int((held["missing"] == 0).sum())  # a minute the store holds, not a hole
    out.update(pages=pages, written=written, replaced=replaced, offered=offered, stored=stored,
               first=None if first is None else first.isoformat(), last=None if last is None else last.isoformat())
    return out




# Where the backfill starts for an instrument the PM asks Research for: enough for the default study
# (holdout, training and test windows) without fetching every trade since the instrument listed.
REQUEST_YEARS = 5


def _pairs_in_use(venue: str, store=None) -> list[tuple[str, pd.Timestamp | None]]:
    """(instrument, where a first backfill starts) for the venue's core list (VenueProfile.core_pairs, always
    kept so research has them before any strategy trades them), every strategy's instrument and every
    instrument the PM asked Research for."""
    from sleeve_fund.venues import VENUES

    profile = VENUES.get(venue.upper())
    pairs: dict[str, pd.Timestamp | None] = {p: None for p in (profile.core_pairs if profile else ())}
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
    """A perpetual venue's settled funding for the instrument, kept beside its prices (sleeve_fund.funding), and
    its open interest snapshots (sleeve_fund.open_interest)."""
    funding_to = None
    if profile.funding_loader is not None:
        from sleeve_fund import funding

        try:
            before = funding.rates(profile.name, pair, root).index
            refused: list = []
            kept = funding.refresh(profile.name, pair, root=root, since=since, refused=refused)
            funding_to = kept.index[-1] if len(kept) else None
            _alert_over_published_cap(profile, pair, kept[kept.index.difference(before)])
            if refused:  # never kept as real rates: charged as missing, and said once a day for these settlements
                at = ", ".join(f"{pd.Timestamp(t, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} ({r!r})" for t, r in refused)
                _alert(f"{profile.name} {pair}: funding invalid {refused[0][0]}-{refused[-1][0]}", "warning",
                       "funding_invalid", f"{pair}: {len(refused)} settled funding rate(s) refused "
                       f"as not a number within the {funding.cap_of(profile.name, pair):.2%} cap, kept as missing: {at}")
            missed, maybe = funding.settled_holes(profile.name, pair, root)  # QA P1-O18
            if missed or maybe:  # the hub's log (the status workflow) has the whole history every pass
                span = lambda ab: f"{ab[0]:%Y-%m-%d %H:%M} to {ab[1]:%Y-%m-%d %H:%M}"  # noqa: E731
                print(f"{profile.name} {pair}: funding: {len(missed)} missed settlement(s)"
                      + "".join(f"; missed between {span(g)}" for g in missed)
                      + "".join(f"; possible hole at interval change {span(g)}" for g in maybe))
                _alert_funding_holes(profile.name, pair, root, [f"missed between {span(g)}" for g in missed]
                                     + [f"possible hole at interval change {span(g)}" for g in maybe])
        except Exception as exc:  # noqa: BLE001 - the prices are stored; funding catches up on the next pass
            print(f"{profile.name} {pair}: funding refresh failed: {exc!r}")
        try:  # outside the refresh, so a feed failing on every pass is still raised
            now = _now()
            problem, key = funding.stale(profile.name, pair, root, now=now), f"{profile.name} {pair}: funding stale"
            # One episode per outage, whoever notices first: a paper strategy on it may have opened (or closed) it
            # already, under the same tag (Advisor, 6-7 Oct 2026; CR, #163).
            tag, inbox = funding.stale_tag(profile.name, pair), _inbox()
            rates = funding.rates(profile.name, pair, root)
            if problem:
                print(f"FUNDING STALE: {problem}")
                if key not in _stale:  # once per episode (Advisor, 6 Oct 2026)
                    _stale.add(key)
                    if funding.stale_open(inbox, tag) is not True:
                        from sleeve_fund import markets

                        _warned.pop(key, None)
                        after = markets.settlement_times(rates.index[-1].to_pydatetime(), now.to_pydatetime(),
                                                         profile.funding_hours, rates)
                        _alert(key, "warning", "funding_stale", f"{tag} {problem.split(': ', 1)[-1]}; "
                               f"{funding.from_words(pd.Timestamp(after[0]) if after else now)}", inbox)
            _open_inner_gaps(profile, pair, rates, inbox, tag, now)
            readable = _review_episodes(profile, pair, root, inbox, tag, now)
            if not problem and key in _stale:
                if not readable:  # the journal unreadable: this process's own episode ends with the rates keeping up
                    _warned.pop(f"{key}: cleared", None)
                    _alert(f"{key}: cleared", "info", "funding_stale_cleared",
                           f"{tag} funding kept up again, newest rate {rates.index[-1]:%Y-%m-%d %H:%M} UTC", inbox)
                    _stale.discard(key)
                elif funding.stale_open(inbox, tag) is False:
                    _stale.discard(key)
        except Exception as exc:  # noqa: BLE001 - an unreadable store is reported by the refresh itself
            print(f"{profile.name} {pair}: funding staleness check failed: {exc!r}")
    _refresh_open_interest(profile, pair, root, funding_to)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


# How long after a settlement the venue may take to publish its rate before the collector counts it as due (the
# strategies' FUNDING_WAIT).
PUBLISH_WAIT = pd.Timedelta(minutes=15)


def _open_inner_gaps(profile, pair: str, rates: pd.Series, inbox, tag: str, now: pd.Timestamp) -> None:
    """A settlement missing between two kept records (the gap inference: markets.settlement_times), due in the last
    day, opens an episode as a late newest one does: one funding_stale per run of contiguous missing settlements,
    from its first (Advisor, 7 Oct 2026; QA P1-O17a-10, -11). _review_episodes then marks each one and asks the
    venue's history for it. Older holes are history, reported by funding.settled_holes; a day is when one still
    unpublished is never published."""
    from sleeve_fund import funding, markets

    if len(rates) < 2:
        return
    try:
        state = funding.journal_state(inbox, tag)
    except Exception:  # noqa: BLE001 - no database (locally), or a stub inbox
        return
    since = max(rates.index[0], now - funding.NEVER_PUBLISHED_AFTER)
    upto = min(now - PUBLISH_WAIT, rates.index[-1])
    if upto <= since:
        return
    runs: list[list[pd.Timestamp]] = [[]]
    for t in markets.settlement_times(since.to_pydatetime(), upto.to_pydatetime(), profile.funding_hours, rates):
        t = pd.Timestamp(t)
        if (abs(rates.index - t) <= funding.MATCH).any():
            runs.append([])
        else:
            runs[-1].append(t)
    opened = [o for o in state["open"] if o is not None]
    for run in filter(None, runs):
        if any(t in state["missing"] or t in state["never"] for t in run) or \
                any(abs(o - t) <= funding.MATCH for o in opened for t in run):
            continue  # already marked, or the episode it belongs to is open
        n = len(run)
        inbox.event(None, "warning", "funding_stale",
                    f"{tag} No settled funding rate from the venue for {pair} at {run[0]:%d %b %Y %H:%M} UTC, "
                    f"{n} settlement{'s' if n != 1 else ''} missing between the rates kept; "
                    f"{funding.from_words(run[0])}")
        opened.append(run[0])


def _review_episodes(profile, pair: str, root, inbox, tag: str, now: pd.Timestamp) -> bool:
    """The instrument's open staleness episodes against the store, settlement by settlement (Advisor, 7 Oct 2026;
    QA P1-O17a-10, -11): every settlement due since the oldest one opened, a publication wait allowed, is kept,
    missing (marked once, and first asked of the venue's history again: funding.backfill) or, still missing a day
    after it was due once a later one is kept, never published (marked once, a warning: its baseline stays). An
    episode closes once every settlement from the one it opened on to the latest due is kept or never published,
    so one arriving late while a later one is missing leaves it open, and one never published can't hold it open
    forever. The settlements are the venue's own times (markets.settlement_times). False when the journal can't be
    read."""
    from sleeve_fund import funding, markets

    try:
        state = funding.journal_state(inbox, tag)
    except Exception:  # noqa: BLE001 - no database (locally), or a stub inbox
        return False
    opened = sorted(k for k in state["open"] if k is not None)
    if not opened:
        return True
    rates = funding.rates(profile.name, pair, root)
    if not len(rates):
        return True
    upto = (now - PUBLISH_WAIT).to_pydatetime()

    def due():
        before = markets.settlement_times(opened[0].to_pydatetime() - pd.Timedelta(days=1), opened[0].to_pydatetime(),
                                          profile.funding_hours, rates)
        start = before[-1] if before else opened[0].to_pydatetime()
        return [pd.Timestamp(t) for t in markets.settlement_times(start - pd.Timedelta(seconds=1), upto,
                                                                   profile.funding_hours, rates)]

    def kept(t) -> bool:
        return bool((abs(rates.index - t) <= funding.MATCH).any())

    lost = [t for t in due() if not kept(t)]
    if lost and funding.backfill(profile.name, pair, lost[0] - funding.MATCH, root):
        rates = funding.rates(profile.name, pair, root)
    settlements = due()
    newest = rates.index[-1]
    for t in settlements:
        if kept(t) or t in state["never"]:
            continue
        if t not in state["missing"]:
            funding.mark(inbox, tag, "funding_missing", t, inferred=t > rates.index[0])
            state["missing"].add(t)
        if now - t >= funding.NEVER_PUBLISHED_AFTER and newest > t + funding.MATCH:
            funding.mark(inbox, tag, "funding_never_published", t)
            state["never"].add(t)
    # From the settlement each opened on (one written before episodes named it opened at its alert, inside the
    # interval after it)
    gap = pd.Timedelta(markets.funding_interval(profile.funding_hours)) - funding.MATCH
    for o in opened:
        if all(kept(t) or t in state["never"] for t in settlements if t > o - gap):
            inbox.event(None, "info", "funding_stale_cleared", f"{tag} funding kept up again, {funding.from_words(o)}, "
                        f"newest rate {newest:%Y-%m-%d %H:%M} UTC")
    return True


def _alert_funding_holes(venue: str, pair: str, root, holes: list[str]) -> None:
    """Each funding hole goes to the alerts inbox once, ever (QA P1-O11, P1-O16): the holes already raised are
    kept beside the rates, so a second new hole the same day still goes in, a restart repeats nothing, and holes
    stored before this check existed are raised on its first pass. Old holes never clear, so the daily cap of
    _alert would either repeat them or hold new ones back. The file's own lock is held from reading it to saving
    it, so two refreshes of one instrument (a manual one while the hub runs) send a hole once (Code Reviewer); the
    store's lock is taken only to save, so a slow database never holds up the store's writes (QA P1-O21)."""
    from sleeve_fund import funding

    path = funding._path(venue, pair, root).with_name("funding.alerted.json")
    with _alerting(path):
        raised, problem = _raised(path)
        if problem:  # an incident in the alerts inbox, once a day (QA P1-O20)
            _alert(f"{path}: unreadable", "error", "funding_alerted_unreadable", f"{pair}: {problem}")  # no venue
        new = [h for h in holes if h not in raised]
        if new:
            try:
                from sleeve_fund.store import Store

                Store().event(None, "warning", "funding_gap", f"{pair}: funding: " + "; ".join(new))  # no venue
            except Exception as exc:  # noqa: BLE001 - no database (locally): the log line says it; retried next pass
                print(f"could not raise the funding_gap alert: {exc!r}")
                return
        elif not problem:
            return
        # Saved with the new holes, or rewritten clean when it couldn't be read, so its incident isn't raised daily.
        with _writing(path.parent):
            tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                tmp.write_text(json.dumps(sorted(raised | set(new))))
                _durable_replace(tmp, path)
            except OSError as exc:  # sent, but not kept: not sent again by this process; a restart sends them again
                _sent_holes.setdefault(str(path), set()).update(new)
                tmp.unlink(missing_ok=True)
                print(f"{venue} {pair}: could not keep the funding holes already raised in {path}: {exc!r}")


@contextlib.contextmanager
def _alerting(path: Path):
    """One sender of an instrument's funding holes at a time, across threads and processes: a lock of its own beside
    the alerted file, apart from the store's write lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name("funding.alerted.lock"), "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


_sent_holes: dict[str, set[str]] = {}  # alerted file -> holes this process sent but couldn't keep in it


def _raised(path: Path) -> tuple[set[str], str | None]:
    """The holes already raised for the instrument, and what is wrong with the file if it can't be used. A file that
    can't be read, or isn't a list of hole words, is read as empty and said (QA P1-O20): its holes go in once more,
    and the file is rewritten, rather than the instrument's alerts stopping for good."""
    raised = set(_sent_holes.get(str(path), ()))
    try:
        kept = json.loads(path.read_text())
    except FileNotFoundError:
        return raised, None
    except (OSError, ValueError) as exc:
        problem = f"the funding holes already raised can't be read ({exc!r}); reading it as empty"
    else:
        if isinstance(kept, list) and all(isinstance(h, str) for h in kept):
            return raised | set(kept), None
        problem = "the funding holes already raised are not a list of holes; reading it as empty"
    print(f"{path}: {problem}")
    return raised, problem


def _refresh_open_interest(profile, pair: str, root, funding_to) -> None:
    """The venue's open interest and positioning snapshots for the instrument (sleeve_fund.open_interest), then
    one line saying how far open interest and funding are kept, which the status workflow shows in the hub's log."""
    if not profile.stats_loaders:
        return
    from sleeve_fund import open_interest

    outs = {}
    for series in profile.stats_loaders:
        try:
            outs[series] = open_interest.refresh(profile.name, pair, root=root, series=series)
        except Exception as exc:  # noqa: BLE001 - snapshots the venue still keeps are fetched on the next pass
            print(f"{profile.name} {pair}: {series.replace('_', ' ')} refresh failed: {exc!r}")
        _warn_at_risk(open_interest.at_risk(profile.name, pair, root=root, series=series))
    out = outs.get("open_interest")
    if out is None:
        return
    oi = f"{out['latest']:%Y-%m-%d %H:%M}" if out["latest"] is not None else "none yet"
    fr = f"{funding_to:%Y-%m-%d %H:%M}" if funding_to is not None else "none yet"
    extra = f", {out['conflicts']} differing (open_interest.provenance.jsonl)" if out["conflicts"] else ""
    print(f"{profile.name} {pair}: open interest to {oi} UTC (+{out['written']}{extra}), funding to {fr} UTC")


_stale: set[str] = set()  # instruments whose funding was last seen stale, for the recovery line
_warned: dict[str, float] = {}  # message prefix -> when it last went to the alerts inbox


def _warn_at_risk(problem: str | None, store=None) -> None:
    """Open interest about to be lost for good goes to the alerts inbox, at most once a day per instrument."""
    if problem is None:
        return
    print(f"OPEN INTEREST AT RISK: {problem}")
    _alert(problem.split(" last kept")[0], "error", "open_interest_at_risk", problem, store)  # instrument and series


def _alert_over_published_cap(profile, pair: str, new: pd.Series) -> None:
    """A rate kept this pass beyond the instrument's published cap (VenueProfile.published_funding_caps) is alerted,
    and still kept and charged as it is: the cheap guard until each settlement's own cap is kept (Advisor, 6 Oct
    2026; DA-11)."""
    cap = (getattr(profile, "published_funding_caps", None) or {}).get(pair)
    over = new[new.abs() > cap] if cap and len(new) else new.iloc[:0]
    if not len(over):
        return
    at = ", ".join(f"{t:%Y-%m-%d %H:%M} ({r:.4%})" for t, r in over.items())
    _alert(f"{profile.name} {pair}: funding over the published cap {over.index[-1]:%Y%m%d%H%M}", "warning",
           "funding_over_cap", f"{pair}: {len(over)} settled funding rate(s) beyond the "
           f"instrument's published {cap:.2%} cap, kept and charged as the venue settled them: {at}")


_inbox_of: list = [None, None]  # (the Store class it was built from, the Store): one per process, not one per pass


def _inbox():
    """The journal the alerts go to, built once, or None without a database (locally)."""
    from sleeve_fund import store as store_mod

    if _inbox_of[0] is not store_mod.Store:  # built again only if the class changed (tests swap it)
        try:
            _inbox_of[:] = [store_mod.Store, store_mod.Store()]
        except Exception:  # noqa: BLE001
            return None
    return _inbox_of[1]


def _alert(key: str, level: str, kind: str, message: str, store=None) -> None:
    """One event in the alerts inbox, at most once a day per key."""
    import time

    if time.time() - _warned.get(key, 0) < 86_400:
        return
    try:
        from sleeve_fund.store import Store

        (store or Store()).event(None, level, kind, message)
        _warned[key] = time.time()
    except Exception as exc:  # noqa: BLE001 - no database (locally): the log line still says it
        print(f"could not raise the {kind} alert: {exc!r}")


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
    can = sub.add_parser("canon", help="make stored minutes since a time the venue's own candles (P1-1-CANON)")
    can.add_argument("pairs", nargs="*", help="instruments (default: core list plus sleeves')")
    can.add_argument("--venue", default=None)
    can.add_argument("--since", required=True, help="UTC time to start from, e.g. 2026-10-06")
    can.add_argument("--until", default=None,
                     help="UTC time to stop before (minutes opening at or after it are left alone), e.g. 2026-10-07T18:00")
    can.add_argument("--dry-run", action="store_true", help="count the minutes it would replace and fill; write nothing")
    rep = sub.add_parser("report", help="what is stored, and any gaps or duplicates")
    rep.add_argument("--venue", default=None)
    exp = sub.add_parser("export", help="one instrument's stored 1-minute candles in a window, as CSV (read-only)")
    exp.add_argument("pair", help="instrument, e.g. BTC/USDT")
    exp.add_argument("--venue", default=None)
    exp.add_argument("--start", required=True, help="first minute's open time, UTC, e.g. 2026-10-06T00:00")
    exp.add_argument("--end", required=True, help="UTC minute to stop before, e.g. 2026-10-08T00:00")
    exp.add_argument("--canon-replaced", action="store_true",
                     help="instead of the candles, the canon backfill's latest replacement record per minute")
    exp.add_argument("--expect", type=int, default=None,
                     help="with --canon-replaced: the count the canon run reported; refuse any other")
    args = ap.parse_args(argv)

    store = HistoryStore(args.root)
    profile = venue_profile(args.venue)
    if args.cmd in ("refresh", "run", "canon") and (problem := unwritable(store.root / profile.name.upper())):
        print(f"{profile.name}: history store can't start: {problem}")
        return 2
    if args.cmd == "export":  # GOLDEN-VENUE: read-only, CSV to stdout, a refusal to stderr
        try:
            start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
            sys.stdout.write(export_canon_replaced(store, profile.name, args.pair, start, end, args.expect)
                             if args.canon_replaced else export_csv(store, profile.name, args.pair, start, end))
        except (KeyError, ValueError) as exc:
            print(f"export refused: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "report":
        for v, pair in store.series():
            if v == profile.name:
                print(store.report(v, pair))
        return 0
    if args.cmd == "canon":
        since = pd.Timestamp(args.since, tz="UTC")
        until = pd.Timestamp(args.until, tz="UTC") if args.until else None
        pairs = args.pairs or [p for p, _ in _pairs_in_use(profile.name)]
        if until is not None:  # every instrument checked before any is written, so a refusal leaves none half done
            late = [(p, top) for p in pairs if (cov := store.coverage(profile.name, p)) is not None
                    and until - pd.Timedelta(minutes=1) > (top := _canon_top(cov))]
            for pair, top in late:
                print(f"{pair}: until {until.isoformat()} is past the newest closed minute stored ({top.isoformat()})")
            if late:
                return 2
        for pair in pairs:
            print(canon(store, profile, pair, since, dry_run=args.dry_run, until=until))
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
            except Exception as exc:  # noqa: BLE001 - one bad pair or a venue hiccup must not stop the rest
                print(f"{profile.name} {pair}: refresh failed: {exc!r}")
            try:  # on its own: failing prices must not skip funding, open interest or the week-old alert (QA P1-O5)
                _refresh_funding(profile, pair, args.root, since)
            except Exception as exc:  # noqa: BLE001
                print(f"{profile.name} {pair}: funding and open interest refresh failed: {exc!r}")
        if not behind:
            time.sleep(args.idle)


if __name__ == "__main__":
    raise SystemExit(main())
