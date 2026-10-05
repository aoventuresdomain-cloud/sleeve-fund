"""The indicator library: shared blocks every strategy, the chart and research use.

Every block is built from plain settings and offers `update_raw(...)`, `handle_bar(bar)`, `value` (its main
output), `values` (every output by name, for the chart and the Signals tab), `initialized`, `warmup_bars`
(history it needs before it reads as settled, from its settings), `reset()` and, where only the close goes in,
`peek(close)`. A value depends only on bars already passed, and no block knows its candle size: a strategy
feeds a block closed slower candles to read it on a slower timeframe. `BLOCKS` names them for definitions,
and `warmup_for` gives the history a set of blocks needs.

The engine's own SimpleMovingAverage, and AverageTrueRange which averages through it, abort the
whole process (a Rust panic, not an exception) once the period passes 1,024. A 2,000-bar average is
an ordinary request on minute or 5-minute bars, so strategies use these instead. They match the
engine's versions to floating-point precision at any period both accept (tests/test_indicators.py).
"""

from __future__ import annotations

import copy
import math
from collections import deque

from nautilus_trader.model import Bar

# Wilder's averages (RSI, ATR) and exponential ones (EMA) never forget their starting value, only discount it:
# after k bars it still weighs (1 - 1/n)^k for Wilder, (1 - 2/(n+1))^k for an EMA. Ten lengths take it below
# 0.01% for RSI(14), so a strategy started on that much history reads what a long-running one would (PM, 5 Oct
# 2026: "if bare minimum it requires a warm up of 140+ then it requires 140+"). A simple average needs only its
# own length.
SETTLE_LENGTHS = 10


def settle_bars(period: int) -> int:
    """Bars of history a Wilder or exponential average of `period` needs before it reads as a settled one."""
    return SETTLE_LENGTHS * int(period)


class Sma:
    """Simple moving average over the last `period` values, O(1) per value. Before `period` values
    have arrived it averages what it has, like the engine's version, but is not yet initialized."""

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError(f"period must be at least 1, got {period}")
        self.period = period
        self.reset()

    def reset(self) -> None:
        self._window: deque[float] = deque(maxlen=self.period)
        self._sum = 0.0
        self._since_exact = 0
        self.count = 0
        self.value = 0.0

    def update_raw(self, x: float) -> None:
        if len(self._window) == self.period:
            self._sum -= self._window[0]
        self._window.append(x)
        self._sum += x
        self.count += 1
        self._since_exact += 1
        if self._since_exact >= self.period:  # a running sum drifts; recompute it once a window
            self._sum, self._since_exact = sum(self._window), 0
        self.value = self._sum / len(self._window)

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.close.as_double())

    @property
    def initialized(self) -> bool:
        return self.count >= self.period

    @property
    def warmup_bars(self) -> int:
        return self.period

    @property
    def values(self) -> dict[str, float]:
        return {"value": self.value}

    def peek(self, close: float) -> float | None:
        return _peek(self, close)


class Atr:
    """Average true range: the simple average of each bar's range, stretched to the previous close
    when the bar gapped (the engine's default settings)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.reset()

    def reset(self) -> None:
        self._avg = Sma(self.period)
        self._prev_close: float | None = None

    def update_raw(self, high: float, low: float, close: float) -> None:
        prev = self._prev_close
        tr = high - low if prev is None else max(high, prev) - min(low, prev)
        self._avg.update_raw(tr)
        self._prev_close = close

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double())

    @property
    def value(self) -> float:
        return self._avg.value

    @property
    def initialized(self) -> bool:
        return self._avg.initialized

    @property
    def warmup_bars(self) -> int:
        return self.period + 1  # a simple average of true ranges; the first range needs the bar before

    @property
    def values(self) -> dict[str, float]:
        return {"value": self.value}


class Rsi:
    """Wilder's RSI(period) on the usual 0 to 100 scale, updated with each bar's close: the first average
    gain and loss are the mean of the first `period` changes, each later one (previous x (period - 1) + this
    change) / period. The engine's RelativeStrengthIndex smooths exponentially (alpha 2 / (period + 1))
    instead, which reads about 5 points off the standard RSI on minute bars and touches 30/70 about twice
    as often, so the strategy traded a different RSI from the one the chart draws (review round 11, M11-1).
    dashboard/static/console.js draws these same values (tests/test_indicators.py)."""

    def __init__(self, period: int = 14) -> None:
        if int(period) != period or period < 2:
            raise ValueError("an RSI period is a whole number of bars, at least 2")
        self.period = int(period)
        self.reset()

    def reset(self) -> None:
        self._prev: float | None = None
        self._gains: list[float] = []
        self._losses: list[float] = []
        self._avg_gain = self._avg_loss = 0.0
        self.value = 0.0
        self.initialized = False

    def update_raw(self, close: float) -> None:
        close = float(close)
        if self._prev is None:
            self._prev = close
            return
        change, self._prev = close - self._prev, close
        gain, loss = max(change, 0.0), max(-change, 0.0)
        n = self.period
        if not self.initialized:
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) < n:
                return
            self._avg_gain, self._avg_loss = sum(self._gains) / n, sum(self._losses) / n
            self._gains = self._losses = []
            self.initialized = True
        else:
            self._avg_gain = (self._avg_gain * (n - 1) + gain) / n
            self._avg_loss = (self._avg_loss * (n - 1) + loss) / n
        if self._avg_loss == 0:
            self.value = 100.0 if self._avg_gain > 0 else 50.0
        else:
            self.value = 100.0 - 100.0 / (1.0 + self._avg_gain / self._avg_loss)

    def handle_bar(self, bar) -> None:
        self.update_raw(bar.close.as_double())

    @property
    def warmup_bars(self) -> int:
        return settle_bars(self.period)

    @property
    def values(self) -> dict[str, float]:
        return {"value": self.value}

    def peek(self, close: float) -> float | None:
        """The value this RSI would read if `close` closed the next bar, worked out on a copy so the RSI
        itself is untouched: the forming candle's value for display, never for a decision. None until the
        copy has enough bars."""
        probe = copy.deepcopy(self)
        probe.update_raw(close)
        return probe.value if probe.initialized else None


NS_PER_DAY = 86_400 * 1_000_000_000
# A day-anchored VWAP needs every bar since 00:00 UTC. Blocks don't know their candle size, so ask for a day
# of minute bars, which covers a full UTC day whatever candle the strategy builds from them.
DAY_OF_MINUTE_BARS = 1_440


def _whole(name: str, value, least: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < least:
        raise ValueError(f"{name} must be a whole number, at least {least}, got {value!r}")
    return int(value)


def _peek(block, close: float) -> float | None:
    """The value `block` would read if `close` closed the next bar, worked out on a copy so the block itself
    is untouched: the forming candle's value for display, never for a decision. None until the copy has
    enough bars."""
    probe = copy.deepcopy(block)
    probe.update_raw(close)
    return probe.value if probe.initialized else None


class Ema:
    """Exponential moving average, alpha 2 / (period + 1), started at the first value and initialized after
    `period` values: the engine's ExponentialMovingAverage, which rsi_pullback trades today, to floating-point
    precision (tests/test_indicators_ref.py), without its period limit."""

    def __init__(self, period: int) -> None:
        self.period = _whole("period", period, 1)
        self.alpha = 2.0 / (self.period + 1)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.value = 0.0

    def update_raw(self, x: float) -> None:
        x = float(x)
        self.value = x if self.count == 0 else self.alpha * x + (1.0 - self.alpha) * self.value
        self.count += 1

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.close.as_double())

    @property
    def initialized(self) -> bool:
        return self.count >= self.period

    @property
    def warmup_bars(self) -> int:
        return settle_bars(self.period)

    @property
    def values(self) -> dict[str, float]:
        return {"value": self.value}

    def peek(self, close: float) -> float | None:
        return _peek(self, close)


class Wma:
    """Linearly weighted moving average: the newest of `period` values weighs `period`, the oldest 1. Before
    `period` values have arrived it weights what it has the same way, but is not yet initialized."""

    def __init__(self, period: int) -> None:
        self.period = _whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        self._window: deque[float] = deque(maxlen=self.period)
        self._sum = self._num = 0.0  # sum of the window, and of each value times its weight
        self._since_exact = 0
        self.count = 0
        self.value = 0.0

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
        self.value = self._num / (n * (n + 1) / 2)

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.close.as_double())

    @property
    def initialized(self) -> bool:
        return self.count >= self.period

    @property
    def warmup_bars(self) -> int:
        return self.period

    @property
    def values(self) -> dict[str, float]:
        return {"value": self.value}

    def peek(self, close: float) -> float | None:
        return _peek(self, close)


class Vwap:
    """Volume-weighted average of each bar's typical price, (high + low + close) / 3.

    anchor="day" restarts at 00:00 UTC, because the market trades round the clock and the anchor has to be
    explicit. A bar belongs to the day its close falls in, and bars are stamped at their close, so the bar
    closing at 00:00 ends the old day. It is initialized only once it has seen a day start, so a strategy
    started mid-day never trades a VWAP built from part of the day.
    anchor="rolling" averages the last `period` bars instead.
    With no volume to weigh it reads the latest typical price."""

    def __init__(self, anchor: str = "day", period: int | None = None) -> None:
        if anchor not in ("day", "rolling"):
            raise ValueError(f"a VWAP is anchored to the UTC 'day' or 'rolling', got {anchor!r}")
        if anchor == "rolling":
            if period is None:
                raise ValueError("a rolling VWAP needs a period")
            period = _whole("period", period, 1)
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
        self.value = 0.0

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
        self.value = self._pv / self._v if self._v > 0 else typical

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double(),
                        bar.volume.as_double(), bar.ts_event)

    @property
    def initialized(self) -> bool:
        return self._day_started if self.anchor == "day" else self.count >= self.period

    @property
    def warmup_bars(self) -> int:
        return DAY_OF_MINUTE_BARS + 1 if self.anchor == "day" else self.period

    @property
    def values(self) -> dict[str, float]:
        return {"value": self.value}


FLAT_BAND = 1e-9


class Bollinger:
    """Bollinger Bands: the simple average of the last `period` closes (mid), k population standard deviations
    either side (upper, lower), the band width as a fraction of mid, and %B, where the close sits in the band
    (0 at lower, 1 at upper). `value` is mid. A flat window has no band, and %B then reads 0.5. A deviation
    under a billionth of the price is rounding left by the sliding sums, not a band, so it counts as flat."""

    def __init__(self, period: int = 20, k: float = 2.0) -> None:
        self.period = _whole("period", period, 2)
        if not (float(k) > 0 and math.isfinite(float(k))):
            raise ValueError(f"k must be a positive number of standard deviations, got {k!r}")
        self.k = float(k)
        self.reset()

    def reset(self) -> None:
        self._window: deque[float] = deque(maxlen=self.period)
        self._mean = self._m2 = 0.0  # Welford's mean and sum of squared deviations, slid with the window
        self._since_exact = 0
        self.count = 0
        self.value = 0.0
        self._close = 0.0
        self.values = {"mid": 0.0, "upper": 0.0, "lower": 0.0, "width": 0.0, "pct_b": 0.5}

    def update_raw(self, close: float) -> None:
        x = float(close)
        w = self._window
        if len(w) == self.period:
            y = w[0]
            w.append(x)
            old_mean = self._mean
            self._mean += (x - y) / self.period
            self._m2 += (x - y) * (x - self._mean + y - old_mean)
        else:
            w.append(x)
            delta = x - self._mean
            self._mean += delta / len(w)
            self._m2 += delta * (x - self._mean)
        self.count += 1
        self._since_exact += 1
        if self._since_exact >= self.period:  # sliding sums drift; recompute them once a window
            self._mean = sum(w) / len(w)
            self._m2 = sum((v - self._mean) ** 2 for v in w)
            self._since_exact = 0
        sd = math.sqrt(max(self._m2, 0.0) / len(w))
        if sd <= FLAT_BAND * abs(self._mean):
            sd = 0.0
        mid, half = self._mean, self.k * sd
        upper, lower = mid + half, mid - half
        self.value = mid
        self.values = {
            "mid": mid, "upper": upper, "lower": lower,
            "width": (upper - lower) / mid if mid else 0.0,
            "pct_b": (x - lower) / (upper - lower) if upper > lower else 0.5,
        }

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.close.as_double())

    @property
    def initialized(self) -> bool:
        return self.count >= self.period

    @property
    def warmup_bars(self) -> int:
        return self.period

    def peek(self, close: float) -> float | None:
        return _peek(self, close)


class RelativeVolume:
    """This bar's volume over the average volume of the `period` bars before it (rsi_pullback's volume spike).
    Initialized once those bars exist. If they traded nothing it reads 0, so a volume rule never passes on
    a ratio that has no meaning."""

    def __init__(self, period: int = 20) -> None:
        self.period = _whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        self._avg = Sma(self.period)
        self.initialized = False
        self.value = 0.0

    def update_raw(self, volume: float) -> None:
        volume = float(volume)
        prev = self._avg.value if self._avg.initialized else None
        self._avg.update_raw(volume)
        if prev is None:
            return
        self.initialized = True
        self.value = volume / prev if prev > 0 else 0.0

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.volume.as_double())

    @property
    def warmup_bars(self) -> int:
        return self.period + 1

    @property
    def values(self) -> dict[str, float]:
        return {"value": self.value}


class EfficiencyRatio:
    """Kaufman's efficiency ratio over `period` bars: the net change in close over the sum of the absolute
    bar-to-bar changes. 1 is a straight line, near 0 is chop. With no movement at all it reads 0."""

    def __init__(self, period: int = 10) -> None:
        self.period = _whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        self._closes: deque[float] = deque(maxlen=self.period + 1)
        self._moves: deque[float] = deque(maxlen=self.period)
        self._path = 0.0
        self._since_exact = 0
        self.value = 0.0

    def update_raw(self, close: float) -> None:
        x = float(close)
        if self._closes:
            move = abs(x - self._closes[-1])
            if len(self._moves) == self.period:
                self._path -= self._moves[0]
            self._moves.append(move)
            self._path += move
            self._since_exact += 1
            if self._since_exact >= self.period:  # a running sum drifts; recompute it once a window
                self._path, self._since_exact = sum(self._moves), 0
        self._closes.append(x)
        if self.initialized:
            self.value = abs(x - self._closes[0]) / self._path if self._path > 0 else 0.0

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.close.as_double())

    @property
    def initialized(self) -> bool:
        return len(self._closes) > self.period

    @property
    def warmup_bars(self) -> int:
        return self.period + 1

    @property
    def values(self) -> dict[str, float]:
        return {"value": self.value}

    def peek(self, close: float) -> float | None:
        return _peek(self, close)


BLOCKS: dict[str, type] = {
    "sma": Sma,
    "ema": Ema,
    "wma": Wma,
    "vwap": Vwap,
    "rsi": Rsi,
    "bollinger": Bollinger,
    "atr": Atr,
    "relative_volume": RelativeVolume,
    "efficiency_ratio": EfficiencyRatio,
}


def make_block(kind: str, **settings):
    """A block from its name in BLOCKS and its plain settings, as a definition names it."""
    try:
        cls = BLOCKS[kind]
    except KeyError:
        raise ValueError(f"no indicator block called {kind!r}; known: {', '.join(sorted(BLOCKS))}") from None
    return cls(**settings)


def warmup_for(blocks) -> int:
    """Bars of history a strategy using `blocks` needs before every one of them reads as settled."""
    return max((b.warmup_bars for b in blocks), default=0)
