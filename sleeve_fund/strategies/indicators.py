"""Moving averages with no period limit.

The engine's own SimpleMovingAverage, and AverageTrueRange which averages through it, abort the
whole process (a Rust panic, not an exception) once the period passes 1,024. A 2,000-bar average is
an ordinary request on minute or 5-minute bars, so strategies use these instead. They match the
engine's versions to floating-point precision at any period both accept (tests/test_indicators.py).
"""

from __future__ import annotations

from collections import deque

from nautilus_trader.model import Bar


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
