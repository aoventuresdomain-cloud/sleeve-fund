"""Bands and channels around price."""

from __future__ import annotations

import math
from collections import deque

from sleeve_fund.strategies.indicators._common import (
    PERIOD_MAX, Block, Setting, peek, peek_values, warmup, whole,
)

# A deviation under a billionth of the price is rounding left by the sliding sums, not a band.
FLAT_BAND = 1e-9


class Bollinger(Block):
    """Bollinger Bands: the simple average of the last `period` closes (mid), k population standard deviations
    either side (upper, lower), the band width as a fraction of mid, and %B, where the close sits in the band
    (0 at lower, 1 at upper). `value` is mid. A flat window has no band (FLAT_BAND), and %B then reads 0.5."""

    SETTINGS = (Setting("period", int, 20, 2, PERIOD_MAX), Setting("k", float, 2.0, 0.1, 10.0))
    OUTPUTS = ("mid", "upper", "lower", "width", "pct_b")

    def __init__(self, period: int = 20, k: float = 2.0) -> None:
        self.period = whole("period", period, 2)
        if isinstance(k, bool) or not (float(k) > 0 and math.isfinite(float(k))):
            raise ValueError(f"k must be a positive number of standard deviations, got {k!r}")
        self.k = float(k)
        self.reset()

    def reset(self) -> None:
        self._window: deque[float] = deque(maxlen=self.period)
        self._mean = self._m2 = 0.0  # Welford's mean and sum of squared deviations, slid with the window
        self._since_exact = 0
        self.count = 0
        self._value = 0.0
        self._vals = dict.fromkeys(self.OUTPUTS, 0.0)

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
        self._value = mid
        self._vals = {
            "mid": mid, "upper": upper, "lower": lower,
            "width": (upper - lower) / mid if mid else 0.0,
            "pct_b": (x - lower) / (upper - lower) if upper > lower else 0.5,
        }

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

    def peek(self, close: float) -> float | None:
        return peek(self, close)

    def peek_values(self, close: float) -> dict:
        return peek_values(self, close)
