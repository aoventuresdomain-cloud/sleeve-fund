"""Filters on activity and trend quality: relative volume and the efficiency ratio."""

from __future__ import annotations

from collections import deque

from sleeve_fund.strategies.indicators._common import PERIOD_MAX, Block, Setting, peek, warmup, whole
from sleeve_fund.strategies.indicators.classic import Sma


class RelativeVolume(Block):
    """This bar's volume over the average volume of the `period` bars before it (rsi_pullback's volume spike).
    Initialized once those bars exist. If they traded nothing it reads 0, so a volume rule never passes on
    a ratio that has no meaning."""

    SETTINGS = (Setting("period", int, 20, 1, PERIOD_MAX),)

    def __init__(self, period: int = 20) -> None:
        self.period = whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        self._avg = Sma(self.period)
        self._ready = False
        self._value = 0.0

    def update_raw(self, volume: float) -> None:
        volume = float(volume)
        if volume < 0:
            raise ValueError(f"volume can't be negative, got {volume}")
        prev = self._avg.value if self._avg.initialized else None
        self._avg.update_raw(volume)
        if prev is None:
            return
        self._ready = True
        self._value = volume / prev if prev > 0 else 0.0

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(volume)

    @property
    def initialized(self) -> bool:
        return self._ready

    def _outputs(self) -> dict:
        return {"value": self._value}

    @warmup
    def warmup_bars(cls, s) -> int:
        return s["period"] + 1


class EfficiencyRatio(Block):
    """Kaufman's efficiency ratio over `period` bars: the net change in close over the sum of the absolute
    bar-to-bar changes. 1 is a straight line, near 0 is chop. With no movement at all it reads 0."""

    SETTINGS = (Setting("period", int, 10, 1, PERIOD_MAX),)

    def __init__(self, period: int = 10) -> None:
        self.period = whole("period", period, 1)
        self.reset()

    def reset(self) -> None:
        self._closes: deque[float] = deque(maxlen=self.period + 1)
        self._moves: deque[float] = deque(maxlen=self.period)
        self._path = 0.0
        self._since_exact = 0
        self._value = 0.0

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
            self._value = abs(x - self._closes[0]) / self._path if self._path > 0 else 0.0

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(close)

    @property
    def initialized(self) -> bool:
        return len(self._closes) > self.period

    def _outputs(self) -> dict:
        return {"value": self._value}

    @warmup
    def warmup_bars(cls, s) -> int:
        return s["period"] + 1

    def peek(self, close: float) -> float | None:
        return peek(self, close)
