"""Minute-bar history for research.

Kraken's API serves only the last 720 candles, so long intraday history comes from Binance's
public archive (data.binance.vision): free, complete 1-minute candles with volume, quoted in
USDT. It stands in for the USD price of the same instrument; the tear sheet says so. Stored per
instrument as a compressed .npz (no extra dependencies) and extended month by month.
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE = "https://data.binance.vision/data/spot"
# Instrument (as the fund names it) -> archive symbol.
SYMBOLS = {"BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT", "SOL/USD": "SOLUSDT", "SUI/USD": "SUIUSDT",
           "XRP/USD": "XRPUSDT"}
FIRST_MONTH = date(2017, 8, 1)  # the archive's first month for any spot symbol
FIELDS = ("o", "h", "l", "c", "v")


def path_for(instrument: str, root: Path) -> Path:
    return root / f"{instrument.replace('/', '-')}.npz"


def _months(start: date, end: date) -> list[date]:
    out, d = [], date(start.year, start.month, 1)
    while d <= end:
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def _parse(raw: bytes) -> np.ndarray:
    """Archive CSV -> rows of (t_seconds, o, h, l, c, v). Times are ms before 2025 and µs after."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode()
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if not r or not r[0].isdigit():
            continue  # a header row, if present
        t = int(r[0])
        t = t // 1_000_000 if t > 10**14 else t // 1000
        rows.append((t, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    return np.array(rows, dtype=float).reshape(-1, 6)


def _get(url: str, opener=urllib.request.urlopen) -> bytes | None:
    try:
        with opener(url, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def update(instrument: str, root: Path, today: date | None = None, get=_get, log=print) -> Path:
    """Fetch any missing months (and the current month's days) and save. Idempotent."""
    sym = SYMBOLS[instrument]
    root.mkdir(parents=True, exist_ok=True)
    p = path_for(instrument, root)
    have = load_raw(p) if p.exists() else np.empty((0, 6))
    today = today or datetime.now(timezone.utc).date()
    last = int(have[-1, 0]) if len(have) else None
    # Re-fetch the last stored month in full: it may have been saved part-way through.
    start = datetime.fromtimestamp(last, timezone.utc).date().replace(day=1) if last else FIRST_MONTH
    parts = [have[have[:, 0] < pd.Timestamp(start).timestamp()]] if len(have) else []
    this_month = today.replace(day=1)
    for m in _months(start, this_month):
        if m < this_month:
            raw = get(f"{ARCHIVE}/monthly/klines/{sym}/1m/{sym}-1m-{m:%Y-%m}.zip")
            if raw is not None:
                parts.append(_parse(raw))
                log(f"{instrument} {m:%Y-%m}: {len(parts[-1])} bars")
        else:  # the current month only exists as daily files
            for d in pd.date_range(m, today - pd.Timedelta(days=1), freq="D"):
                raw = get(f"{ARCHIVE}/daily/klines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip")
                if raw is not None:
                    parts.append(_parse(raw))
    data = np.concatenate(parts) if parts else np.empty((0, 6))
    if len(data):
        data = data[np.argsort(data[:, 0], kind="stable")]
        keep = np.ones(len(data), bool)
        keep[1:] = np.diff(data[:, 0]) > 0  # drop duplicate minutes
        data = data[keep]
    np.savez_compressed(p, t=data[:, 0].astype(np.int64), **{f: data[:, i + 1] for i, f in enumerate(FIELDS)})
    return p


def load_raw(p: Path) -> np.ndarray:
    z = np.load(p)
    return np.column_stack([z["t"].astype(float)] + [z[f] for f in FIELDS])


def load(instrument: str, root: Path, start=None, end=None) -> pd.DataFrame:
    """Minute bars indexed by bar open time (UTC), columns open/high/low/close/volume."""
    z = np.load(path_for(instrument, root))
    idx = pd.to_datetime(z["t"], unit="s", utc=True)
    df = pd.DataFrame({"open": z["o"], "high": z["h"], "low": z["l"], "close": z["c"], "volume": z["v"]}, index=idx)
    if start is not None:
        df = df[df.index >= utc(start)]
    if end is not None:
        df = df[df.index < utc(end)]
    return df


def utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Minute bars -> N-minute bars, labelled by open time. Empty buckets are dropped."""
    if minutes == 1:
        return df
    out = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["close"])


def span(df: pd.DataFrame) -> dict:
    """What history an instrument actually has, for the tear sheet."""
    if df.empty:
        return {"bars": 0}
    expected = (df.index[-1] - df.index[0]) / pd.Timedelta(minutes=1) + 1
    return {"bars": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
            "years": round((df.index[-1] - df.index[0]).days / 365.25, 2),
            "missing_pct": round(100 * (1 - len(df) / expected), 3)}
