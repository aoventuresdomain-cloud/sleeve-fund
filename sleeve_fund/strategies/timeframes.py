"""Slower candles for a strategy (v2 P1-4): a strategy deciding on 15-minute candles can read 4-hour ones of the
same instrument, built here from its own decision candles, so a backtest and paper build them the same way.

Candles are aligned to the venue's daily anchor, 00:00 UTC on every venue so far (VenueProfile.daily_anchor_minutes):
4-hour ones close at 00, 04, 08 ... and daily ones at 00:00. A slower candle closes on the decision candle stamped at
its end; both close at that instant, so the decision on that candle sees it and no earlier one does, and the candle
still forming is never visible. When that decision candle is missing (a gap), the slower candle closes on the first
one after its end, stamped at its own end. A slower candle missing decision candles inside it is built from those it
has, and a part candle at the very start (begun before the data was) is dropped: the history store resamples the
same way (m13-E6), so warm-ups from it agree with candles built here. A slower candle with no decision candles at all
is missing: none is made up for it, and its close is kept in `missed`."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

MINUTE_NS = 60_000_000_000
DAY_MINUTES = 1440


def bar_spec(minutes: int) -> str:
    """A slower candle size as a bar spec the engine parses: 240 -> 4-HOUR-LAST-INTERNAL."""
    n, unit = ((minutes // DAY_MINUTES, "DAY") if minutes % DAY_MINUTES == 0 else
               (minutes // 60, "HOUR") if minutes % 60 == 0 else (minutes, "MINUTE"))
    return f"{n}-{unit}-LAST-INTERNAL"


def span(minutes: int) -> str:
    """A candle size in words: 240 -> 4-hour."""
    return (f"{minutes // DAY_MINUTES}-day" if minutes % DAY_MINUTES == 0 else
            f"{minutes // 60}-hour" if minutes % 60 == 0 else f"{minutes}-minute")


@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    end: int  # ns: the candle's close, which stamps it


class SlowerCandles:
    """Closed `minutes` candles from decision candles of `step_minutes`, each fed on closing to `blocks` (indicator
    blocks, by update_ohlcv). `last` is the latest closed candle, `count` how many have closed, `missed` the closes
    of the latest candles with no decision candles at all (missing, not made up), `need` how many
    the blocks' warm-up takes (set by the strategy). anchor: the venue's daily anchor, in minutes after 00:00 UTC,
    the candles align to."""

    def __init__(self, minutes: int, step_minutes: int, blocks=(), anchor: int = 0) -> None:
        if int(minutes) != minutes or minutes <= 0 or DAY_MINUTES % minutes:
            raise ValueError(f"slower candles of {minutes} minutes don't divide a day (e.g. 60, 240 or 1440)")
        if minutes <= step_minutes or minutes % step_minutes:
            raise ValueError(f"{minutes}-minute candles can't be built from {step_minutes}-minute ones: they must "
                             "be a whole multiple of them")
        self.minutes, self.step_minutes = int(minutes), int(step_minutes)
        self.period, self.step = self.minutes * MINUTE_NS, self.step_minutes * MINUTE_NS
        self.blocks = list(blocks)
        self.anchor = int(anchor)
        self._offset = self.anchor * MINUTE_NS % self.period
        self.last: Candle | None = None
        self.count = self.need = 0
        self.missed: deque[int] = deque(maxlen=1000)
        self._seen_end: int | None = None  # the close of the latest candle that had decision candles
        self._end: int | None = None  # the forming candle's close, None between candles
        self._whole = False
        self._ohlcv: list[float] = []

    def update(self, open_: float, high: float, low: float, close: float, volume: float, ts: int) -> Candle | None:
        """One decision candle, stamped at its close `ts` (ns). Returns the slower candle it closed, if any."""
        if self.last is not None and ts <= self.last.end:  # inside a candle already closed (seeded from the store)
            return None
        end = -(-(ts - self._offset) // self.period) * self.period + self._offset  # the slower candle's close
        out = None
        if self._end is not None and self._end != end:  # its closing decision candle never came
            out = self._close()
        if self._end is None:
            if self._seen_end is not None:
                self.missed.extend(range(self._seen_end + self.period, end, self.period))
            self._end, self._whole, self._seen_end = end, ts - self.step == end - self.period, end
            self._ohlcv = [open_, high, low, close, volume]
        else:
            o, h, lo, _, v = self._ohlcv
            self._ohlcv = [o, max(h, high), min(lo, low), close, v + volume]
        if ts == end:  # never with a late close above: a candle can't open and close on one decision candle
            out = self._close()
        return out

    def handle_bar(self, bar) -> Candle | None:
        return self.update(bar.open.as_double(), bar.high.as_double(), bar.low.as_double(), bar.close.as_double(),
                           bar.volume.as_double(), bar.ts_event)

    def seed(self, candles) -> None:
        """Warm-up from slower candles already closed (the history store's, resampled to this size): fed as if
        built here. Decision candles after the last of them follow through update()."""
        for c in candles:
            self._emit(c)

    def _close(self) -> Candle | None:
        end, whole, (o, h, lo, c, v) = self._end, self._whole, self._ohlcv
        self._end = None
        if not whole and self.last is None:  # a part candle at the start of the data
            return None
        return self._emit(Candle(o, h, lo, c, v, end))

    def _emit(self, candle: Candle) -> Candle:
        self._seen_end = max(self._seen_end or candle.end, candle.end)
        for block in self.blocks:
            block.update_ohlcv(candle.open, candle.high, candle.low, candle.close, candle.volume, candle.end)
        self.last, self.count = candle, self.count + 1
        return candle
