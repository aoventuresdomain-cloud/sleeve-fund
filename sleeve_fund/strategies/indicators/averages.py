"""Exponential and weighted moving averages, and VWAP."""

from __future__ import annotations

from collections import deque

from sleeve_fund.strategies.indicators._common import PERIOD_MAX, Block, Setting, peek, settle_bars, warmup, whole

NS_PER_DAY = 86_400 * 1_000_000_000
# A day-anchored VWAP needs every bar since 00:00 UTC. Blocks don't know their candle size, so ask for a day
# of minute bars, which covers a full UTC day whatever candle the strategy builds from them.
DAY_OF_MINUTE_BARS = 1_440


class Ema(Block):
    """Exponential moving average, alpha 2 / (period + 1), started at the first value: the engine's
    ExponentialMovingAverage, which rsi_pullback trades today, to floating-point precision
    (tests/test_indicators_ref.py), without its period limit. It is initialized only once it has settled, after
    ten lengths (QA, F1): before that it still leans on its starting value, which the engine's version does not
    wait out."""

    SETTINGS = (Setting("period", int, 20, 1, PERIOD_MAX),)

    def __init__(self, period: int = 20) -> None:
        self.period = whole("period", period, 1)
        self.alpha = 2.0 / (self.period + 1)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self._value = 0.0

    def update_raw(self, x: float) -> None:
        x = float(x)
        self._value = x if self.count == 0 else self.alpha * x + (1.0 - self.alpha) * self._value
        self.count += 1

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(close)

    @property
    def initialized(self) -> bool:
        return self.count >= settle_bars(self.period)

    def _outputs(self) -> dict:
        return {"value": self._value}

    @warmup
    def warmup_bars(cls, s) -> int:
        return settle_bars(s["period"])

    def peek(self, close: float) -> float | None:
        return peek(self, close)


class Wma(Block):
    """Linearly weighted moving average: the newest of `period` values weighs `period`, the oldest 1."""

    SETTINGS = (Setting("period", int, 20, 1, PERIOD_MAX),)

    def __init__(self, period: int = 20) -> None:
        self.period = whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        self._window: deque[float] = deque(maxlen=self.period)
        self._sum = self._num = 0.0  # sum of the window, and of each value times its weight
        self._since_exact = 0
        self.count = 0
        self._value = 0.0

    def update_raw(self, x: float) -> None:
        x = float(x)
        w = self._window
        if len(w) == self.period:  # every weight drops by one and the oldest (weight 1) leaves
            self._num -= self._sum
            self._sum -= w[0]
        w.append(x)
        self._sum += x
        self._num += len(w) * x
        self.count += 1
        self._since_exact += 1
        if self._since_exact >= self.period:  # running sums drift; recompute them once a window
            self._sum = sum(w)
            self._num = sum((i + 1) * v for i, v in enumerate(w))
            self._since_exact = 0
        n = len(w)
        self._value = self._num / (n * (n + 1) / 2)

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

    def peek(self, close: float) -> float | None:
        return peek(self, close)


class Vwap(Block):
    """Volume-weighted average of each bar's typical price, (high + low + close) / 3.

    anchor="day" restarts at 00:00 UTC, because the market trades round the clock and the anchor has to be
    explicit. A bar belongs to the day its close falls in, and bars are stamped at their close, so the bar
    closing at 00:00 ends the old day. It is initialized only once it has seen a day start, so a strategy
    started mid-day never trades a VWAP built from part of the day. It needs each bar's close time (ts_ns).
    anchor="rolling" averages the last `period` bars instead.
    With no volume to weigh it reads the latest typical price."""

    SETTINGS = (Setting("anchor", str, "day", choices=("day", "rolling")), Setting("period", int, None, 1, PERIOD_MAX))

    def __init__(self, anchor: str = "day", period: int | None = None) -> None:
        if anchor not in ("day", "rolling"):
            raise ValueError(f"a VWAP is anchored to the UTC 'day' or 'rolling', got {anchor!r}")
        if anchor == "rolling":
            if period is None:
                raise ValueError("a rolling VWAP needs a period")
            period = whole("period", period, 1)
        elif period is not None:
            raise ValueError("a day-anchored VWAP takes no period")
        self.anchor, self.period = anchor, period
        self.reset()

    def reset(self) -> None:
        self._pv = self._v = 0.0
        self._window: deque[tuple[float, float]] = deque(maxlen=self.period or None)
        self._since_exact = 0
        self._day: int | None = None
        self._day_started = False
        self.count = 0
        self._value = 0.0

    def update_raw(self, high: float, low: float, close: float, volume: float, ts_ns: int | None = None) -> None:
        typical = (float(high) + float(low) + float(close)) / 3.0
        volume = float(volume)
        if volume < 0:
            raise ValueError(f"volume can't be negative, got {volume}")
        if self.anchor == "day":
            if ts_ns is None:
                raise ValueError("a day-anchored VWAP needs each bar's close time")
            day = (int(ts_ns) - 1) // NS_PER_DAY
            if self._day is not None and day != self._day:
                self._pv = self._v = 0.0
                self._day_started = True
            self._day = day
            self._pv += typical * volume
            self._v += volume
        else:
            w = self._window
            if len(w) == self.period:
                old_pv, old_v = w[0]
                self._pv -= old_pv
                self._v -= old_v
            w.append((typical * volume, volume))
            self._pv += typical * volume
            self._v += volume
            self._since_exact += 1
            if self._since_exact >= self.period:  # running sums drift; recompute them once a window
                self._pv, self._v = sum(p for p, _ in w), sum(v for _, v in w)
                self._since_exact = 0
        self.count += 1
        self._value = self._pv / self._v if self._v > 0 else typical

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(high, low, close, volume, ts_ns)

    @property
    def initialized(self) -> bool:
        return self._day_started if self.anchor == "day" else self.count >= self.period

    def _outputs(self) -> dict:
        return {"value": self._value}

    @warmup
    def warmup_bars(cls, s) -> int:
        if s["anchor"] == "day":
            return DAY_OF_MINUTE_BARS + 1
        if s["period"] is None:
            raise ValueError("a rolling VWAP needs a period")
        return s["period"]


# Ten lengths leave up to 0.12% of a Wilder ATR's start on fast candles, where a bar's range can be ten times the
# seed's; twenty leave under 1e-7 (QA P1-A1). A range average gets the longer settle; RSI keeps ten.
ATR_SETTLE_LENGTHS = 20


class Atr(Block):
    """Wilder's average true range, the standard ATR (Independent Quant Advisor and QA, P1-I4): a bar's true range
    is its high minus low, stretched to the previous close when the bar gapped, and the first bar's is its high
    minus low. The first average is the mean of the first `period` ranges; each later one is (previous x
    (period - 1) + this range) / period. Like RSI it never forgets its start, only discounts it, so it is
    initialized after ATR_SETTLE_LENGTHS lengths. `atr_sma` is the simple average the hand-coded models use."""

    SETTINGS = (Setting("period", int, 14, 1, PERIOD_MAX),)

    def __init__(self, period: int = 14) -> None:
        self.period = whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        self._prev_close: float | None = None
        self._seed: list[float] = []
        self.count = 0
        self._value = 0.0

    def update_raw(self, high: float, low: float, close: float) -> None:
        high, low, close = float(high), float(low), float(close)
        prev = self._prev_close
        tr = high - low if prev is None else max(high, prev) - min(low, prev)
        self._prev_close = close
        self.count += 1
        n = self.period
        if self.count <= n:
            self._seed.append(tr)
            self._value = sum(self._seed) / len(self._seed)
            if self.count == n:
                self._seed = []
        else:
            self._value = (self._value * (n - 1) + tr) / n

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(high, low, close)

    @property
    def initialized(self) -> bool:
        return self.count >= ATR_SETTLE_LENGTHS * self.period

    def _outputs(self) -> dict:
        return {"value": self._value}

    @warmup
    def warmup_bars(cls, s) -> int:
        return ATR_SETTLE_LENGTHS * int(s["period"])
