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

import json
import os
import threading
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

    def as_dict(self) -> dict:
        out = {"first": self.first.isoformat(), "last": self.last.isoformat(), "cursor": self.cursor}
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
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or DEFAULT_ROOT)

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
        return Coverage(pd.Timestamp(raw["first"]), pd.Timestamp(raw["last"]), raw.get("cursor", ""), closed)

    def append(self, venue: str, pair: str, minutes: pd.DataFrame, cursor: str, merge: bool = False) -> Coverage:
        """Add 1-minute bars (indexed by open time, UTC) and move the loader's cursor.

        The loader vouches that nothing is missing between the stored end and these bars (it
        resumes from its cursor), so minutes in between had no trades and are stored flat. An
        overlapping minute is replaced, or with merge=True (bars built from trades that continue
        exactly where the last page stopped) combined: first open, highest high, lowest low,
        last close, volumes added. The last stored minute may still be forming; read() leaves it out."""
        if minutes.empty:
            raise ValueError("no bars to append")
        df = _check_minutes(minutes)
        d = self._dir(venue, pair)
        with _lock:
            d.mkdir(parents=True, exist_ok=True)
            cov = self.coverage(venue, pair)
            if cov is not None and df.index[0] > cov.last:  # quiet minutes since the stored end
                prev = self._month(d, cov.last).loc[[cov.last]]
                df = _fill_quiet(pd.concat([prev, df])).iloc[1:]
            elif cov is not None and merge and df.index[0] == cov.last:
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
                           cov.closed if cov else None)
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
        rows = np.asarray(list(bars), dtype=float).reshape(-1, 6)
        if not len(rows):
            return AppendResult(0, 0, 0)
        df = _bars_frame(rows)
        d = self._dir(venue, pair)
        written = unchanged = 0
        log: list[dict] = []
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with _lock:
            d.mkdir(parents=True, exist_ok=True)
            cov = self.coverage(venue, pair)
            for month, chunk in df.groupby(df.index.strftime("%Y-%m")):
                path = d / f"{month}.npz"
                old = _load(path) if path.exists() else df.iloc[:0]
                old = old[~old.index.duplicated(keep="last")]
                forming = _forming(old.index, cov)
                held = chunk.index.isin(old.index[~forming])
                for ts in chunk.index[held]:
                    have, offer = old.loc[ts, OHLCV].to_numpy(float), chunk.loc[ts, OHLCV].to_numpy(float)
                    if np.allclose(have, offer, rtol=1e-12, atol=0.0):
                        unchanged += 1
                    else:
                        log.append({"kind": "conflict", "at": now, "source": source, "minute": ts.isoformat(),
                                    "stored": have.tolist(), "offered": offer.tolist()})
                new = chunk[~held]
                if new.empty:
                    continue
                written += len(new)
                _save(path, pd.concat([old[~old.index.isin(new.index)], new]).sort_index())
            if source == "refill" and written:
                log.append({"kind": "refill", "at": now, "first": df.index[0].isoformat(),
                            "last": df.index[-1].isoformat(), "minutes": written})
            if log:
                with open(d / "provenance.jsonl", "a") as f:
                    f.writelines(json.dumps(e) + "\n" for e in log)
            lo, hi = df.index[0], df.index[-1]
            new_cov = Coverage(min(lo, cov.first) if cov else lo, max(hi, cov.last) if cov else hi,
                               cov.cursor if cov else "", max(hi, cov.closed) if cov and cov.closed is not None else hi)
            _write_coverage(d, new_cov)
        return AppendResult(written, unchanged, sum(e["kind"] == "conflict" for e in log))

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
        # Just past the last minute that is surely closed: a longer bar ending after it is a part bar.
        stop = cov.last + pd.Timedelta(minutes=0 if _forming(pd.DatetimeIndex([cov.last]), cov)[0] else 1)
        parts = []
        for path in sorted(d.glob("*.npz")):
            month = pd.Timestamp(path.stem + "-01", tz="UTC")
            if lo is not None and month + pd.offsets.MonthBegin(1) <= lo:
                continue
            if hi is not None and month > hi:
                continue
            part = _load(path)
            part = part[(part.index >= cov.first) & ~_forming(part.index, cov)]
            parts.append(_resample(part, minutes, cov.first, stop) if by_month else part)
        if not parts:
            return pd.DataFrame(columns=OHLCV)
        bars = pd.concat(parts).sort_index()
        if minutes > 1 and not by_month:
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
        df = pd.concat([_load(p) for p in sorted(self._dir(venue, pair).glob("*.npz"))]).sort_index()
        df = df[(df.index >= cov.first) & (df.index <= cov.last)]
        expected = int((cov.last - cov.first) / pd.Timedelta("1min")) + 1
        return {"venue": venue, "pair": pair, "stored": True, "first": cov.first, "last": cov.last,
                "minutes": len(df), "missing": expected - df.index.nunique(),
                "duplicates": int(df.index.duplicated().sum()),
                "quiet_over_an_hour": quiet_runs(df[~df.index.duplicated()], 1)}

    def gaps(self, venue: str, pair: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Runs of minutes missing inside the coverage, as (first, last) missing open times. append() fills
        quiet minutes, so a run here is a write that never finished, not a market with no trades. Each
        month's file is read once per version (its size and time), so the current month is the only one
        re-read as the collector adds to it."""
        cov = self.coverage(venue, pair)
        if cov is None:
            return []
        lo, hi = int(cov.first.timestamp()), int(cov.last.timestamp())
        runs, seen = [], lo - 60  # the last minute held so far
        for path in sorted(self._dir(venue, pair).glob("*.npz")):
            if path.name.endswith(".tmp.npz"):  # a month being rewritten (_save)
                continue
            first, last, holes = _month_minutes(path)
            if first > seen + 60:
                runs.append((seen + 60, first - 60))
            runs += holes
            seen = max(seen, last)
        if hi > seen:
            runs.append((seen + 60, hi))
        clipped = [(max(a, lo), min(b, hi)) for a, b in runs if b >= lo and a <= hi]
        return [(pd.Timestamp(a, unit="s", tz="UTC"), pd.Timestamp(b, unit="s", tz="UTC")) for a, b in clipped]


_months: dict[Path, tuple[tuple[int, int], tuple[int, int, list[tuple[int, int]]]]] = {}


def _month_minutes(path: Path) -> tuple[int, int, list[tuple[int, int]]]:
    """A month file's first and last minute (epoch seconds) and the runs missing between them."""
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
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
    """Stored minutes that may still be forming: the REST loader's newest minute, unless the hub has
    since stored that minute (or a later one) as closed."""
    if cov is None:
        return np.zeros(len(index), dtype=bool)
    closed = cov.closed if cov.closed is not None else cov.last - pd.Timedelta("1min")
    return np.asarray((index >= cov.last) & (index > closed))


def _bars_frame(rows: np.ndarray) -> pd.DataFrame:
    """Hub bar rows -> a checked frame of closed minutes by open time. Not quiet-filled."""
    t = rows[:, 0].astype(np.int64)
    if (t % 60_000_000_000).any():
        raise ValueError("bars must be stamped on whole minutes (their open time, in ns)")
    df = pd.DataFrame(rows[:, 1:], columns=OHLCV, index=pd.to_datetime(t, unit="ns", utc=True))
    df.index.name = "timestamp"
    if df.index.duplicated().any():
        raise ValueError("the same minute twice in one batch")
    if not np.isfinite(rows[:, 1:]).all():
        raise ValueError("bars with missing values")
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any() or (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise ValueError("bars with impossible prices or volume")
    return df.sort_index()


def _write_coverage(d: Path, cov: Coverage) -> None:
    tmp = d / "coverage.json.tmp"
    tmp.write_text(json.dumps(cov.as_dict()))
    tmp.replace(d / "coverage.json")


def _save(path: Path, df: pd.DataFrame) -> None:
    t = (df.index.as_unit("s").asi8).astype(np.int64)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=t, **{c: df[c].to_numpy(float) for c in OHLCV})
    tmp.replace(path)


def _resample(df: pd.DataFrame, minutes: int, first: pd.Timestamp | None = None,
              end: pd.Timestamp | None = None) -> pd.DataFrame:
    """1-minute bars by open time -> `minutes` bars by open time. A bar the stored minutes only partly cover at
    either end, from `first` (the first minute) to `end` (just past the last), is left out: a part-day isn't a
    daily bar. A bar inside the series missing a minute is kept from the minutes it has, as paper builds every
    bar (m13-E6); only a bar with no minutes at all is left out. The bounds default to the frame's own."""
    if minutes <= 1 or df.empty:
        return df
    first = df.index[0] if first is None else first
    end = df.index[-1] + pd.Timedelta(minutes=1) if end is None else end
    g = df.resample(f"{minutes}min", origin="epoch", label="left", closed="left")
    bars = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    whole = (bars.index >= first) & (bars.index + pd.Timedelta(minutes=minutes) <= end)
    return bars[whole & (g["close"].count() > 0).to_numpy()]


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
