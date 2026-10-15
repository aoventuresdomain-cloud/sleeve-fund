"""Levels: the highest and lowest of a series over a window (tool gap G2) and the previous calendar period's high, low
and close (G4)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

from sleeve_fund.strategies.indicators._common import PERIOD_MAX, Block, Setting, warmup, whole

DAY_MINUTES = 1_440
PERIODS = {"day": 1, "week": 7}  # days in each calendar period


class Rolling(Block):
    """The highest (max) and lowest (min) of the last `period` values fed, the one just fed included. It takes one
    input, so a definition can feed it another block's output ("Bollinger width at its 120-candle low" is
    bb.width <= lo.min with lo = rolling(120) fed bb.width); unfed, it reads the close. `value` is max."""

    SETTINGS = (Setting("period", int, 20, 1, PERIOD_MAX),)
    OUTPUTS = ("max", "min")

    def __init__(self, period: int = 20) -> None:
        self.period = whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        # Monotonic queues of (value number, value): the front is the window's max (or min), O(1) a value on average.
        self._highs: deque[tuple[int, float]] = deque()
        self._lows: deque[tuple[int, float]] = deque()
        self.count = 0
        self._value = 0.0
        self._vals = dict.fromkeys(self.OUTPUTS, 0.0)

    def update_raw(self, value: float) -> None:
        x = float(value)
        while self._highs and self._highs[-1][1] <= x:
            self._highs.pop()
        self._highs.append((self.count, x))
        while self._lows and self._lows[-1][1] >= x:
            self._lows.pop()
        self._lows.append((self.count, x))
        self.count += 1
        oldest = self.count - self.period
        while self._highs[0][0] < oldest:
            self._highs.popleft()
        while self._lows[0][0] < oldest:
            self._lows.popleft()
        self._value = self._highs[0][1]
        self._vals = {"max": self._value, "min": self._lows[0][1]}

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(close)

    @property
    def initialized(self) -> bool:
        return self.count >= self.period

    def _outputs(self) -> dict:
        return dict(self._vals)

    @warmup
    def warmup_bars(cls, s) -> int:
        return s["period"]


class PriorPeriod(Block):
    """The previous calendar period's high, low and close: the last whole day, or the last whole week (Monday to
    Sunday), in UTC or the zone `tz` names. A candle belongs to the period its close falls in, as the day VWAP's
    does, so the candle closing at 00:00 ends the old one; candles are stamped at their close (ts_ns). The values
    change only as a new period starts, so nothing from the period being traded is ever read. The period the block
    starts in is a part one, so it reads as initialized only once a period it saw from its start has ended.

    Its warm-up depends on the candle it is fed: warmup_minutes() is two periods, the longest it can wait, and the
    rule builder counts it in candles of the block's size; warmup_bars counts 1-minute bars."""

    SETTINGS = (Setting("period", str, "week", choices=tuple(PERIODS)), Setting("tz", str, "UTC"))
    OUTPUTS = ("high", "low", "close")

    def __init__(self, period: str = "week", tz: str = "UTC") -> None:
        if period not in PERIODS:
            raise ValueError(f"period is one of {', '.join(PERIODS)}, got {period!r}")
        self.period, self.tz = period, str(tz)
        self._zone = _zone(self.tz)
        self.reset()

    @classmethod
    def check_settings(cls, settings: dict) -> dict:
        out = super().check_settings(settings)
        _zone(out["tz"])  # a zone that doesn't exist is refused with the definition, not when it starts
        return out

    def reset(self) -> None:
        self._key = None  # the start (a date) of the period being built
        self._whole = False  # whether it was seen from its start
        self._hi = self._lo = self._close = 0.0
        self._ready = False
        self._value = 0.0
        self._vals = dict.fromkeys(self.OUTPUTS, 0.0)

    def _period_of(self, ts_ns: int):
        local = datetime.fromtimestamp((int(ts_ns) - 1) // 1_000_000_000, tz=self._zone).date()
        return local - timedelta(days=local.weekday()) if self.period == "week" else local

    def update_raw(self, high: float, low: float, close: float, ts_ns: int | None = None) -> None:
        if ts_ns is None:
            raise ValueError("a previous-period level needs each candle's close time")
        key = self._period_of(ts_ns)
        if key != self._key:
            if self._key is not None:
                if self._whole:
                    self._vals = {"high": self._hi, "low": self._lo, "close": self._close}
                    self._value, self._ready = self._hi, True
                self._whole = True
            self._key, self._hi, self._lo = key, float(high), float(low)
        else:
            self._hi, self._lo = max(self._hi, float(high)), min(self._lo, float(low))
        self._close = float(close)

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(high, low, close, ts_ns)

    @property
    def initialized(self) -> bool:
        return self._ready

    @property
    def settled(self) -> bool:
        return self._ready  # its warm-up is in time, not candles: it is settled as soon as a whole period is behind it

    def _outputs(self) -> dict:
        return dict(self._vals)

    @warmup
    def warmup_bars(cls, s) -> int:
        return cls.warmup_minutes(s) + 1  # on 1-minute bars, as the day VWAP counts it; the builder uses the minutes

    @classmethod
    def warmup_minutes(cls, settings: dict) -> int:
        """Minutes of candles before it reads as settled, at most: the rest of a part period, then a whole one."""
        return 2 * PERIODS[cls.check_settings(settings)["period"]] * DAY_MINUTES


def _zone(tz: str):
    if tz.upper() == "UTC":
        return timezone.utc
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"no timezone called {tz!r}; use a name such as Europe/London or UTC") from None
