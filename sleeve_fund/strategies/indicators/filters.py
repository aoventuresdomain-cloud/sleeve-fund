"""Filters on activity and trend quality: relative volume, the efficiency ratio and Wilder's ADX."""

from __future__ import annotations

from collections import deque

from sleeve_fund.strategies.indicators._common import PERIOD_MAX, Block, Setting, peek, settle_bars, warmup, whole
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


def _ewm(prev: float | None, x: float, alpha: float) -> float:
    """One step of pandas' ewm(alpha=..., adjust=False).mean() on a series without gaps, in its own arithmetic, so
    the Adx block reads regime_split.adx to the last bit (tests/test_adx_block.py)."""
    if prev is None:
        return x
    if prev == x:
        return prev
    old = 1.0 - alpha
    return (old * prev + alpha * x) / (old + alpha)


class Adx(Block):
    """Wilder's ADX(period), the formula of sleeve_fund.research.regime_split.adx worked bar by bar: true range and
    the directional moves (+DM, -DM) smoothed with Wilder's 1/period average started at the first bar's, DX from
    them (0 where +DI and -DI are both 0), and ADX the same average of DX. regime_split reads it from bar 2 x period
    on; like every Wilder average here the block is initialized only after ten lengths, once its start no longer
    shows."""

    SETTINGS = (Setting("period", int, 14, 1, PERIOD_MAX),)

    def __init__(self, period: int = 14) -> None:
        self.period = whole("period", period, 1)
        com = (1.0 - 1.0 / self.period) / (1.0 / self.period)  # pandas turns alpha into a centre of mass and back
        self.alpha = 1.0 / (1.0 + com)
        self.reset()

    def reset(self) -> None:
        self._prev: tuple[float, float, float] | None = None  # the bar before's high, low and close
        self._tr = self._plus = self._minus = self._adx = None
        self.count = 0
        self._value = 0.0

    def update_raw(self, high: float, low: float, close: float) -> None:
        high, low, close = float(high), float(low), float(close)
        if self._prev is None:
            tr, plus, minus = high - low, 0.0, 0.0
        else:
            ph, pl, pc = self._prev
            tr = max(high - low, abs(high - pc), abs(low - pc))
            up, down = high - ph, -(low - pl)
            plus = up if up > down and up > 0 else 0.0
            minus = down if down > up and down > 0 else 0.0
        self._prev = (high, low, close)
        a = self.alpha
        self._tr, self._plus, self._minus = _ewm(self._tr, tr, a), _ewm(self._plus, plus, a), _ewm(self._minus, minus, a)
        dx = 0.0
        if self._tr > 0:
            pdi, mdi = 100 * self._plus / self._tr, 100 * self._minus / self._tr
            if pdi + mdi > 0:
                dx = 100 * abs(pdi - mdi) / (pdi + mdi)
        self._adx = self._value = _ewm(self._adx, dx, a)
        self.count += 1

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(high, low, close)

    @property
    def initialized(self) -> bool:
        return self.count >= settle_bars(self.period)

    def _outputs(self) -> dict:
        return {"value": self._value}

    @warmup
    def warmup_bars(cls, s) -> int:
        return settle_bars(s["period"])
