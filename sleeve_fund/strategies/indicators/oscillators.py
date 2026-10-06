"""Oscillators bounded 0 to 100: the stochastic."""

from __future__ import annotations

from collections import deque

from sleeve_fund.strategies.indicators._common import PERIOD_MAX, Block, Setting, warmup, whole
from sleeve_fund.strategies.indicators.classic import Sma


class Stochastic(Block):
    """The slow stochastic: where the close sits in the high-low range of the last `period` bars, this one
    included (0 at the low, 100 at the high), averaged over `smooth` bars (k), and k averaged over `d_period`
    bars (d). `value` is k. A range with no width reads 50. smooth=1 gives the fast stochastic."""

    SETTINGS = (Setting("period", int, 14, 1, PERIOD_MAX), Setting("smooth", int, 3, 1, PERIOD_MAX),
                Setting("d_period", int, 3, 1, PERIOD_MAX))
    OUTPUTS = ("k", "d")

    def __init__(self, period: int = 14, smooth: int = 3, d_period: int = 3) -> None:
        self.period = whole("period", period, 1)
        self.smooth = whole("smooth", smooth, 1)
        self.d_period = whole("d_period", d_period, 1)
        self.reset()

    def reset(self) -> None:
        # Monotonic queues of (bar number, price), as Donchian keeps: the front is the window's high (or low).
        self._highs: deque[tuple[int, float]] = deque()
        self._lows: deque[tuple[int, float]] = deque()
        self.count = 0
        self._k = Sma(self.smooth)
        self._d = Sma(self.d_period)
        self._value = 0.0
        self._vals = dict.fromkeys(self.OUTPUTS, 0.0)

    def update_raw(self, high: float, low: float, close: float) -> None:
        hi, lo, x = float(high), float(low), float(close)
        while self._highs and self._highs[-1][1] <= hi:
            self._highs.pop()
        self._highs.append((self.count, hi))
        while self._lows and self._lows[-1][1] >= lo:
            self._lows.pop()
        self._lows.append((self.count, lo))
        oldest = self.count - self.period + 1
        while self._highs[0][0] < oldest:
            self._highs.popleft()
        while self._lows[0][0] < oldest:
            self._lows.popleft()
        self.count += 1
        if self.count < self.period:
            return
        top, bottom = self._highs[0][1], self._lows[0][1]
        self._k.update_raw(100 * (x - bottom) / (top - bottom) if top > bottom else 50.0)
        if not self._k.initialized:
            return
        k = self._k.value
        self._d.update_raw(k)
        self._value = k
        self._vals = {"k": k, "d": self._d.value}

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(high, low, close)

    @property
    def initialized(self) -> bool:
        return self._d.initialized

    def _outputs(self) -> dict:
        return dict(self._vals)

    @warmup
    def warmup_bars(cls, s) -> int:
        return s["period"] + s["smooth"] + s["d_period"] - 2
