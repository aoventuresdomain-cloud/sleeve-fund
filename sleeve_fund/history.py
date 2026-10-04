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
    last: pd.Timestamp  # open time of the last complete stored minute
    cursor: str  # the loader's resume point, opaque to the store

    def as_dict(self) -> dict:
        return {"first": self.first.isoformat(), "last": self.last.isoformat(), "cursor": self.cursor}


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
        return Coverage(pd.Timestamp(raw["first"]), pd.Timestamp(raw["last"]), raw.get("cursor", ""))

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
            new = Coverage(first, max(df.index[-1], cov.last if cov else df.index[-1]), cursor)
            tmp = d / "coverage.json.tmp"
            tmp.write_text(json.dumps(new.as_dict()))
            tmp.replace(d / "coverage.json")
            return new

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
        for path in sorted(d.glob("*.npz")):
            month = pd.Timestamp(path.stem + "-01", tz="UTC")
            if lo is not None and month + pd.offsets.MonthBegin(1) <= lo:
                continue
            if hi is not None and month > hi:
                continue
            part = _load(path)
            part = part[(part.index >= cov.first) & (part.index < cov.last)]  # the last minute may still be forming
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
        df = pd.concat([_load(p) for p in sorted(self._dir(venue, pair).glob("*.npz"))]).sort_index()
        df = df[(df.index >= cov.first) & (df.index <= cov.last)]
        expected = int((cov.last - cov.first) / pd.Timedelta("1min")) + 1
        return {"venue": venue, "pair": pair, "stored": True, "first": cov.first, "last": cov.last,
                "minutes": len(df), "missing": expected - df.index.nunique(),
                "duplicates": int(df.index.duplicated().sum()),
                "quiet_over_an_hour": quiet_runs(df[~df.index.duplicated()], 1)}


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


def _save(path: Path, df: pd.DataFrame) -> None:
    t = (df.index.as_unit("s").asi8).astype(np.int64)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=t, **{c: df[c].to_numpy(float) for c in OHLCV})
    tmp.replace(path)


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


# Always kept, so research has them before any sleeve trades them; sleeves' own instruments are added.
CORE_PAIRS = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "SUI/USD")


# Where the backfill starts for an instrument the PM asks Research for: enough for the default study
# (holdout, training and test windows) without fetching every trade since the instrument listed.
REQUEST_YEARS = 5


def _pairs_in_use(venue: str, store=None) -> list[tuple[str, pd.Timestamp | None]]:
    """(instrument, where a first backfill starts) for the core list, every strategy's instrument and
    every instrument the PM asked Research for."""
    pairs: dict[str, pd.Timestamp | None] = {p: None for p in CORE_PAIRS}
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
    if args.cmd == "report":
        for v, pair in store.series():
            if v == profile.name:
                print(store.report(v, pair))
        return 0
    if args.cmd == "refresh":
        for pair, since in [(p, None) for p in args.pairs] or _pairs_in_use(profile.name):
            print(refresh(store, profile, pair, since=since))
        return 0
    while True:  # run: round-robin so a long backfill on one instrument doesn't starve the others
        behind = False
        for pair, since in _pairs_in_use(profile.name):
            try:
                out = refresh(store, profile, pair, max_pages=args.pages, since=since)
                behind |= out["pages"] >= args.pages
            except Exception as exc:  # noqa: BLE001 - one bad pair or a venue hiccup must not stop the rest
                print(f"{profile.name} {pair}: refresh failed: {exc!r}")
        if not behind:
            time.sleep(args.idle)


if __name__ == "__main__":
    raise SystemExit(main())
