"""Price history loading.

Kraken publishes daily OHLCVT history as headerless CSVs
(timestamp, open, high, low, close, volume, trades), where timestamp is the bar
OPEN time in Unix seconds. Nautilus treats ts_event as the moment a bar is
known, so we stamp each bar at its CLOSE time. Getting this wrong is a
look-ahead bug: the strategy would trade on a close that hasn't happened yet.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.model import Bar, BarType, Price, Quantity

KRAKEN_OHLCVT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
OHLCV = ["open", "high", "low", "close", "volume"]


def load_kraken_ohlcvt(path: str | Path, interval: str = "1D") -> pd.DataFrame:
    """Load a Kraken OHLCVT CSV and return a frame indexed by bar CLOSE time (UTC)."""
    raw = pd.read_csv(path, header=None, names=KRAKEN_OHLCVT_COLUMNS)
    if raw.empty:
        raise ValueError(f"{path} is empty")
    opened = pd.to_datetime(raw["timestamp"], unit="s", utc=True)
    df = raw[OHLCV].astype(float)
    df.index = opened + pd.Timedelta(interval)
    df.index.name = "timestamp"
    return validate_ohlcv(df)


KRAKEN_API = "https://api.kraken.com/0/public"
_KRAKEN_ALIASES = {"XBT": "BTC", "XDG": "DOGE"}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as r:  # public endpoints only, no key
        return json.load(r)


def kraken_pair_key(pair: str, get_json=_get_json) -> str:
    """Kraken's own id for a BASE/QUOTE pair (e.g. BTC/USD -> XXBTZUSD), from its public pair list."""
    norm = lambda code: _KRAKEN_ALIASES.get(code, code)  # noqa: E731
    want = tuple(norm(c) for c in pair.upper().split("/"))
    data = get_json(f"{KRAKEN_API}/AssetPairs")
    for key, info in data.get("result", {}).items():
        ws = info.get("wsname", "")
        if "/" in ws and tuple(norm(c) for c in ws.split("/")) == want:
            return key
    raise ValueError(f"Kraken does not list {pair}")


def fetch_kraken_daily(pair: str, get_json=_get_json) -> pd.DataFrame:
    """Kraken's most recent daily candles (up to 720) for a pair, indexed by bar CLOSE time.

    The newest candle is still forming, so it is dropped.
    """
    key = kraken_pair_key(pair, get_json)
    data = get_json(f"{KRAKEN_API}/OHLC?" + urllib.parse.urlencode({"pair": key, "interval": 1440}))
    if data.get("error"):
        raise ValueError(f"Kraken: {'; '.join(data['error'])}")
    rows = next((v for k, v in data.get("result", {}).items() if k != "last"), [])[:-1]
    if not rows:
        raise ValueError(f"no daily history for {pair}")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    out = df[OHLCV].astype(float)
    out.index = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True) + pd.Timedelta("1D")
    out.index.name = "timestamp"
    return validate_ohlcv(out)


KRAKEN_INTERVALS = (1, 5, 15, 30, 60, 240, 1440, 10080)  # minutes Kraken's OHLC endpoint accepts


def fetch_kraken_ohlc(pair: str, interval: int, get_json=_get_json) -> pd.DataFrame:
    """Kraken's recent candles (up to 720) for charts, indexed by bar OPEN time, as charting tools
    expect. Unlike fetch_kraken_daily, the newest (still forming) candle is kept, so the chart is live."""
    if interval not in KRAKEN_INTERVALS:
        raise ValueError(f"interval must be one of {KRAKEN_INTERVALS} minutes")
    key = kraken_pair_key(pair, get_json)
    data = get_json(f"{KRAKEN_API}/OHLC?" + urllib.parse.urlencode({"pair": key, "interval": interval}))
    if data.get("error"):
        raise ValueError(f"Kraken: {'; '.join(data['error'])}")
    rows = next((v for k, v in data.get("result", {}).items() if k != "last"), [])
    if not rows:
        raise ValueError(f"no candles for {pair}")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    out = df[OHLCV].astype(float)
    out.index = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    out.index.name = "timestamp"
    return out


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on data that would silently corrupt a backtest."""
    missing = set(OHLCV) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps in price history")
    if df[OHLCV].isna().any().any():
        raise ValueError("NaNs in price history")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices in price history")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any() or (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("high/low inconsistent with open/close")
    return df


def synthetic_ohlcv(
    days: int = 2500,
    start: str = "2018-01-01",
    seed: int = 7,
    drift: float = 0.0004,
    vol: float = 0.035,
    start_price: float = 10_000.0,
) -> pd.DataFrame:
    """Random-walk daily bars for exercising the pipeline. Results on this mean nothing."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, days)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, vol / 2, days))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    idx = pd.date_range(start, periods=days, freq="1D", tz="UTC") + pd.Timedelta("1D")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000.0},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
    return validate_ohlcv(df)


def daily_bar_type(instrument) -> BarType:
    return BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")


def bar_type_for(instrument, minutes: int) -> BarType:
    """The bar type for bars of `minutes` length: 1440 -> 1-DAY, 240 -> 4-HOUR, 15 -> 15-MINUTE."""
    if minutes <= 0:
        raise ValueError("bar length must be positive")
    if minutes % 1440 == 0:
        spec = f"{minutes // 1440}-DAY"
    elif minutes % 60 == 0:
        spec = f"{minutes // 60}-HOUR"
    else:
        spec = f"{minutes}-MINUTE"
    return BarType.from_str(f"{instrument.id}-{spec}-LAST-EXTERNAL")


def bar_minutes(bar_type) -> int:
    """Bar length in minutes from a bar type such as BTC/USD.KRAKEN-4-HOUR-LAST-EXTERNAL."""
    step, unit = str(bar_type).rsplit("-", 4)[1:3]
    return int(step) * {"MINUTE": 1, "HOUR": 60, "DAY": 1440}[unit]


def to_bars(df: pd.DataFrame, instrument, bar_type: BarType | None = None) -> list[Bar]:
    """Build Nautilus bars; each index value is the bar's close time (ts_event)."""
    bar_type = bar_type or daily_bar_type(instrument)
    pp, sp = instrument.price_precision, instrument.size_precision
    ts = df.index.as_unit("ns").asi8
    return [
        Bar(
            bar_type,
            Price(o, pp),
            Price(h, pp),
            Price(lo, pp),
            Price(c, pp),
            Quantity(v, sp),
            int(t),
            int(t),
        )
        for o, h, lo, c, v, t in zip(df["open"], df["high"], df["low"], df["close"], df["volume"], ts)
    ]
