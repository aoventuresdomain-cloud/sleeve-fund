"""Moving averages with no period limit, and the standard RSI.

The engine's own SimpleMovingAverage, and AverageTrueRange which averages through it, abort the
whole process (a Rust panic, not an exception) once the period passes 1,024. A 2,000-bar average is
an ordinary request on minute or 5-minute bars, so strategies use these instead. They match the
engine's versions to floating-point precision at any period both accept (tests/test_indicators.py).
"""

from __future__ import annotations

import copy
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
        self._window: deque[float] = deque(maxlen=period)
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


class Atr:
    """Average true range: the simple average of each bar's range, stretched to the previous close
    when the bar gapped (the engine's default settings)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._avg = Sma(period)
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

    def peek(self, close: float) -> float | None:
        """The value this RSI would read if `close` closed the next bar, worked out on a copy so the RSI
        itself is untouched: the forming candle's value for display, never for a decision. None until the
        copy has enough bars."""
        probe = copy.deepcopy(self)
        probe.update_raw(close)
        return probe.value if probe.initialized else None
