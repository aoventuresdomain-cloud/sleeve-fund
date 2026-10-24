"""Levels: the highest or lowest of a series over a window (tool gap G2), a series' value a set number of candles
back (the lag), the previous calendar period's high, low and close (G4), and the last Asia or London session's high
and low (R1)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, time, timedelta, timezone

from sleeve_fund.strategies.indicators._common import PERIOD_MAX, Block, Setting, warmup, whole

DAY_MINUTES = 1_440
PERIODS = {"day": 1, "week": 7}  # days in each calendar period
# R1: each session's own zone and local window, [start, end) (Advisor, 15 Oct; London on the hour, R1-DEF); the block's settings can override them
SESSIONS = {"asia": ("Asia/Tokyo", "09:00", "17:00"), "london": ("Europe/London", "08:00", "16:00")}


class _Rolling(Block):
    """The highest or lowest of the last `period` values fed, the one just fed included, as the rule builder's
    Donchian channel counts its window (the bar just closed in it). It takes one input, so a definition can feed it
    another block's output: "Bollinger width at its 120-candle low" is bb.width <= lo with lo = rolling_min(120)
    fed bb.width. Unfed, it reads the close. With fewer than `period` values it has no value."""

    SETTINGS = (Setting("period", int, 20, 1, PERIOD_MAX),)
    _keep = staticmethod(lambda last, x: last <= x)  # drop a queued value this one beats: max keeps the larger

    def __init__(self, period: int = 20) -> None:
        self.period = whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        # A monotonic queue of (value number, value): the front is the window's max (or min), O(1) a value on average.
        self._queue: deque[tuple[int, float]] = deque()
        self.count = 0
        self._value = 0.0

    def update_raw(self, value: float) -> None:
        x = float(value)
        while self._queue and self._keep(self._queue[-1][1], x):
            self._queue.pop()
        self._queue.append((self.count, x))
        self.count += 1
        while self._queue[0][0] < self.count - self.period:
            self._queue.popleft()
        self._value = self._queue[0][1]

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(close)

    @property
    def initialized(self) -> bool:
        return self.count >= self.period

    def _outputs(self) -> dict:
        return {"value": self._value}

    @warmup
    def warmup_bars(cls, s) -> int:
        return s["period"]


class RollingMax(_Rolling):
    """The highest of the last `period` values of its input (tool gap G2)."""


class RollingMin(_Rolling):
    """The lowest of the last `period` values of its input (tool gap G2)."""

    _keep = staticmethod(lambda last, x: last >= x)


class Lag(Block):
    """Its input's value `period` values back: fed the 200-day SMA on daily candles with period 20, it reads that
    average as it stood 20 completed days ago, so "the SMA200 is rising" is sma > lag (qr21 round 1, 24 Oct). It
    takes one input, so a definition feeds it another block's output; unfed, it reads the close. Until it has been
    fed period + 1 values it has no value."""

    SETTINGS = (Setting("period", int, 1, 1, PERIOD_MAX),)

    def __init__(self, period: int = 1) -> None:
        self.period = whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        self._window: deque[float] = deque(maxlen=self.period + 1)
        self.count = 0
        self._value = 0.0

    def update_raw(self, value: float) -> None:
        self._window.append(float(value))
        self.count += 1
        self._value = self._window[0]

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(close)

    @property
    def initialized(self) -> bool:
        return self.count > self.period

    def _outputs(self) -> dict:
        return {"value": self._value}

    @warmup
    def warmup_bars(cls, s) -> int:
        return s["period"] + 1


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



class SessionLevels(Block):
    """The last completed Asia or London session's high and low (R1), each in its own market's local time (Advisor,
    15 Oct): Asia is Asia/Tokyo [09:00, 17:00), London is Europe/London [08:00, 16:00), so London's clock change moves
    London in UTC and leaves Asia. `tz`, `start` and `end` ("HH:MM") override the session's own window. A candle
    belongs to the session its close falls in, as PriorPeriod's periods: the one closing at the session's end is its
    last, and the levels move on the next close. Only a session seen from its start is published, so the one the data
    starts in part-way is skipped, and the session in progress is never read. A session missing more than a tenth of
    its candles (counted against the window at the shortest candle step seen) is incomplete: the last levels stay, and
    it is listed in `skipped` as (local date, candles seen, candles expected). A day with no candles in the session
    leaves the last levels in place."""

    SETTINGS = (Setting("session", str, "asia", choices=tuple(SESSIONS)), Setting("tz", str), Setting("start", str),
                Setting("end", str))
    OUTPUTS = ("high", "low")
    MISSING = 0.10  # the most of a session's candles that may be missing before it is not a session (Advisor)
    MIN_CANDLES = 4  # the fewest candles of the idea's interval a window may hold (Advisor R1-WIN)
    MAX_CANDLE_MINUTES = 60  # the longest candle a session level is read on (Advisor R1-DEF)

    def __init__(self, session: str = "asia", tz: str | None = None, start: str | None = None,
                 end: str | None = None) -> None:
        out = self.check_settings({"session": session, "tz": tz, "start": start, "end": end})
        self.session, self.tz, self.start, self.end = out["session"], out["tz"], out["start"], out["end"]
        self._zone = _zone(self.tz)
        self._open, self._close = _clock("start", self.start), _clock("end", self.end)
        self.reset()

    @classmethod
    def check_settings(cls, settings: dict) -> dict:
        out = super().check_settings(settings)
        tz, start, end = SESSIONS[out["session"]]
        out["tz"] = out["tz"] if out["tz"] is not None else tz
        out["start"] = out["start"] if out["start"] is not None else start
        out["end"] = out["end"] if out["end"] is not None else end
        _zone(out["tz"])  # a zone that doesn't exist is refused with the definition, not when it starts
        if _clock("end", out["end"]) <= _clock("start", out["start"]):
            raise ValueError(f"a session's end comes after its start, got {out['start']} to {out['end']}")
        return out

    @classmethod
    def check_candle(cls, settings: dict, minutes: int) -> None:
        """Advisor R1-WIN: checked against the idea's candle when it is loaded. The window starts and ends on a candle
        boundary (candles are cut on UTC, so in each of the zone's offsets through the year), never rounded, and
        holds at least MIN_CANDLES of them. Candles longer than an hour are refused (R1-DEF), failing closed."""
        if minutes > cls.MAX_CANDLE_MINUTES:
            raise ValueError(f"session levels are read on candles of 1 hour or less, and this idea uses "
                             f"{minutes}-minute candles")
        out = cls.check_settings(settings)
        zone, start, end = _zone(out["tz"]), _clock("start", out["start"]), _clock("end", out["end"])
        year = datetime.now(timezone.utc).year
        offsets = {int(datetime(year, month, 15, tzinfo=zone).utcoffset().total_seconds()) // 60 for month in (1, 7)}
        for name, t in (("start", start), ("end", end)):
            if any((t.hour * 60 + t.minute - off) % minutes for off in offsets):
                raise ValueError(f"the session's {name} {t:%H:%M} isn't on a {minutes}-minute candle boundary; "
                                 "set it to one, it isn't rounded")
        length = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
        if length < cls.MIN_CANDLES * minutes:
            raise ValueError(f"the session {out['start']} to {out['end']} holds fewer than {cls.MIN_CANDLES} "
                             f"{minutes}-minute candles")

    def reset(self) -> None:
        self._key = None  # the local date of the session being built
        self._whole = False  # whether it was seen from its start
        self._outside = False  # a candle outside the session seen since the last one inside it
        self._hi = self._lo = 0.0
        self._n = 0  # candles in the session being built
        self._last_ts = None
        self._step = None  # the shortest gap between candle closes seen, in ns
        self._ready = False
        self._value = 0.0
        self._vals = dict.fromkeys(self.OUTPUTS, 0.0)
        self.skipped: list[tuple[str, int, int]] = []

    def _session_of(self, ts_ns: int):
        local = datetime.fromtimestamp((int(ts_ns) - 1) // 1_000_000_000, tz=self._zone)
        return local.date() if self._open <= local.time() < self._close else None

    def _expected(self, day) -> int:
        """Candles the window holds at the shortest step seen; the window is timed in UTC, so a clock change inside
        it counts as the hours it really ran."""
        utc = [datetime.combine(day, t, tzinfo=self._zone).astimezone(timezone.utc) for t in (self._open, self._close)]
        return round((utc[1] - utc[0]).total_seconds() * 1_000_000_000 / self._step)

    def _end_session(self) -> None:
        if not self._whole:
            return
        expected = self._expected(self._key)
        if expected - self._n > self.MISSING * expected:
            self.skipped.append((self._key.isoformat(), self._n, expected))
            return
        self._vals = {"high": self._hi, "low": self._lo}
        self._value, self._ready = self._hi, True

    def update_raw(self, high: float, low: float, ts_ns: int | None = None) -> None:
        if ts_ns is None:
            raise ValueError("a session level needs each candle's close time")
        ts_ns = int(ts_ns)
        if self._last_ts is not None and ts_ns > self._last_ts:
            gap = ts_ns - self._last_ts
            self._step = gap if self._step is None else min(self._step, gap)
        self._last_ts = ts_ns
        key = self._session_of(ts_ns)
        if self._key is not None and key != self._key:  # the session being built has ended
            self._end_session()
            self._key = None
        if key is None:
            self._outside = True
        elif self._key is None:
            self._key, self._whole, self._outside = key, self._outside, False
            self._hi, self._lo, self._n = float(high), float(low), 1
        else:
            self._hi, self._lo = max(self._hi, float(high)), min(self._lo, float(low))
            self._n += 1

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(high, low, ts_ns)

    @property
    def initialized(self) -> bool:
        return self._ready

    @property
    def settled(self) -> bool:
        return self._ready  # its warm-up is in time, not candles: settled once a whole session is behind it

    def _outputs(self) -> dict:
        return dict(self._vals)

    @warmup
    def warmup_bars(cls, s) -> int:
        return cls.warmup_minutes(s) + 1  # on 1-minute bars, as PriorPeriod counts it; the builder uses the minutes

    @classmethod
    def warmup_minutes(cls, settings: dict) -> int:
        """Minutes of candles before it reads as settled, at most: the rest of a part session, then a whole one
        the next day (two days)."""
        cls.check_settings(settings)
        return 2 * DAY_MINUTES


def _clock(name: str, hhmm: str) -> time:
    try:
        hh, mm = hhmm.split(":")
        if len(hh) != 2 or len(mm) != 2:
            raise ValueError
        return time(int(hh), int(mm))
    except (ValueError, AttributeError):
        raise ValueError(f"{name} is a time of day as HH:MM, such as 09:00, got {hhmm!r}") from None


def _zone(tz: str):
    if tz.upper() == "UTC":
        return timezone.utc
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"no timezone called {tz!r}; use a name such as Europe/London or UTC") from None
