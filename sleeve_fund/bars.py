"""The one rule for building a longer bar from 1-minute bars (Head of Engineering, 5 Oct 2026, board 5a), for every
path that builds one: the history store's reads (backtests and warm-ups), the hub client's live bars and a
strategy's slower candles.

- A bar is built from the minutes present: the first open, the highest high, the lowest low, the last close and
  the summed volume.
- It carries `missing`, how many of its minutes are absent.
- More than DEGRADED_ABOVE of them missing makes it degraded: indicators still update on it, but no new entry is
  decided on it (exits still run). At or under that share it is a normal bar.
- A bar with no minutes at all is not built: it is missing, never made up from its neighbours.

combine() is the rule for one bar, as the live paths build bars a minute at a time. build_bars() applies it to
a whole frame at once for the store, where a bar at a time would be far too slow over years of minutes; a test
holds the two to the same bars."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume", "missing", "degraded"]  # every frame build_bars returns
DEGRADED_ABOVE = 0.10  # the Independent Quant Advisor's threshold: the share of minutes missing past which a bar is degraded


@dataclass(frozen=True)
class Built:
    open: float
    high: float
    low: float
    close: float
    volume: float
    missing: int  # minutes of the bar that are absent

    def degraded(self, minutes: int) -> bool:
        return degraded(self.missing, minutes)


def degraded(missing: int, minutes: int) -> bool:
    """Whether a `minutes` bar missing `missing` of its minutes is too thin to enter on."""
    return missing > DEGRADED_ABOVE * minutes


def combine(parts, minutes: int) -> Built | None:
    """One `minutes` bar from the (open, high, low, close, volume) minutes present in it, in time order; None
    when there are none."""
    parts = list(parts)
    if not parts:
        return None
    return Built(parts[0][0], max(p[1] for p in parts), min(p[2] for p in parts), parts[-1][3],
                 sum(p[4] for p in parts), minutes - len(parts))


def build_bars(minutes: pd.DataFrame, length: int, first: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None) -> pd.DataFrame:
    """1-minute bars by open time -> `length`-minute bars by open time, epoch-aligned, each as combine() builds
    it. Columns: open, high, low, close, volume, missing (minutes absent from the bar) and degraded (more than
    DEGRADED_ABOVE of them missing: no new entries on it). A bar only partly inside [first, end) is left out (a
    part-day isn't a daily bar), and so is a bar with no minutes at all. The bounds default to the frame's own.
    Pure: no I/O, no clock."""
    if minutes.empty:
        return pd.DataFrame(columns=COLUMNS, index=minutes.index[:0])
    if length <= 1:  # 1-minute bars are themselves: none of their minutes absent
        return minutes.assign(missing=0, degraded=False)
    first = minutes.index[0] if first is None else first
    end = minutes.index[-1] + pd.Timedelta(minutes=1) if end is None else end
    g = minutes.resample(f"{length}min", origin="epoch", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    present = g["close"].count()
    out["missing"] = (length - present).astype("int64")
    out["degraded"] = out["missing"] > DEGRADED_ABOVE * length
    whole = (out.index >= first) & (out.index + pd.Timedelta(minutes=length) <= end)
    return out[whole & (present > 0).to_numpy()]
