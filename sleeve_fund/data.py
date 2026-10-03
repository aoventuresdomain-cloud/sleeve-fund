"""Price history loading.

Kraken publishes daily OHLCVT history as headerless CSVs
(timestamp, open, high, low, close, volume, trades), where timestamp is the bar
OPEN time in Unix seconds. Nautilus treats ts_event as the moment a bar is
known, so we stamp each bar at its CLOSE time. Getting this wrong is a
look-ahead bug: the strategy would trade on a close that hasn't happened yet.
"""

from __future__ import annotations

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
