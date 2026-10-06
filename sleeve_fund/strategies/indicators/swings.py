"""Blocks read from confirmed swing points. A swing is known only some bars after it, so each says how many
(`confirm_lag`) and signals on the bar that confirms it, never earlier."""

from __future__ import annotations

from collections import deque

from sleeve_fund.strategies.indicators._common import PERIOD_MAX, Block, Setting, warmup, whole
from sleeve_fund.strategies.indicators.classic import Rsi


class RsiDivergence(Block):
    """RSI divergence on confirmed swing points. A swing low is a bar whose low is the lowest of the `left` bars
    before it and strictly below the `right` bars after it, so it is confirmed `right` bars later (`confirm_lag`).
    Bullish divergence is a confirmed swing low below the previous one, at most `max_gap` bars earlier, while RSI
    at the new swing is higher. Bearish is the mirror on swing highs. Each fires on the bar that confirms the
    swing: `value` is 1 bullish, -1 bearish, 0 neither; `values` has both flags. Swings before the RSI is
    initialized don't count. It is itself initialized only after its warm-up."""

    SETTINGS = (Setting("rsi_period", int, 14, 2, PERIOD_MAX), Setting("left", int, 3, 1, 50),
                Setting("right", int, 3, 1, 50), Setting("max_gap", int, 50, 1, 5_000))
    OUTPUTS = ("bullish", "bearish")

    def __init__(self, rsi_period: int = 14, left: int = 3, right: int = 3, max_gap: int = 50) -> None:
        self.rsi_period = whole("rsi_period", rsi_period, 2)
        self.left = whole("left", left, 1)
        self.right = whole("right", right, 1)
        self.max_gap = whole("max_gap", max_gap, 1)
        self.reset()

    def reset(self) -> None:
        self._rsi = Rsi(self.rsi_period)
        self._recent: deque[tuple[float, float, float | None]] = deque(maxlen=self.left + self.right + 1)
        self._last_low: tuple[int, float, float] | None = None  # (bar number, low, RSI) of the last swing low
        self._last_high: tuple[int, float, float] | None = None
        self.count = 0
        self._value = 0
        self._vals = {"bullish": 0, "bearish": 0}

    @property
    def confirm_lag(self) -> int:
        return self.right

    def update_raw(self, high: float, low: float, close: float) -> None:
        self._rsi.update_raw(close)
        self._recent.append((float(high), float(low), self._rsi.value if self._rsi.initialized else None))
        self.count += 1
        bullish = bearish = 0
        if len(self._recent) == self._recent.maxlen:
            bars = list(self._recent)
            h, lo, rsi = bars[self.left]
            before, after = bars[: self.left], bars[self.left + 1:]
            at = self.count - 1 - self.right  # bar number of the swing candidate
            if rsi is not None and lo <= min(b[1] for b in before) and lo < min(b[1] for b in after):
                prev, self._last_low = self._last_low, (at, lo, rsi)
                if prev and at - prev[0] <= self.max_gap and lo < prev[1] and rsi > prev[2]:
                    bullish = 1
            if rsi is not None and h >= max(b[0] for b in before) and h > max(b[0] for b in after):
                prev, self._last_high = self._last_high, (at, h, rsi)
                if prev and at - prev[0] <= self.max_gap and h > prev[1] and rsi < prev[2]:
                    bearish = 1
        self._value = bullish - bearish
        self._vals = {"bullish": bullish, "bearish": bearish}

    def update_ohlcv(self, open_, high, low, close, volume, ts_ns=None) -> None:
        self.update_raw(high, low, close)

    @property
    def initialized(self) -> bool:
        # Not before its warm-up: until then the RSI at an earlier swing still leans on where it started (QA, F1).
        return self._rsi.initialized and self.count >= self.warmup_bars

    def _outputs(self) -> dict:
        return dict(self._vals)

    @warmup
    def warmup_bars(cls, s) -> int:
        # The RSI settled, then room for the previous swing a divergence compares with.
        return Rsi.warmup_bars(period=s["rsi_period"]) + s["left"] + s["right"] + s["max_gap"]
