"""Idea #1 from the plan: a long-only trend filter, with the lab's refinements.

Hold the instrument while its fast average is above its slow average at a bar's close, otherwise
hold cash. Options, all off by default so existing sleeves trade exactly as before:
- ema=1: exponential averages instead of simple ones,
- vol_target > 0: hold vol_target / recent realised volatility of the sleeve (capped at 100%, spot,
  no borrowing) instead of all of it, and only trade back to that weight when it drifts more than
  rebalance_band (default 25%) from the weight last traded to, to keep fees down.
"""

from __future__ import annotations

import math
from collections import deque

from nautilus_trader.model import Bar

from sleeve_fund.data import bar_minutes
from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.indicators import Sma, settle_bars

SPEC = IdeaSpec(
    summary="Long while the {fast}-bar average is above the {slow}-bar average, otherwise flat in cash.",
    name="trend_filter",
    family="trend",
    idea="Long only when the 50-day average is above the 200-day average.",
    rules=(
        "Long when the fast average (simple, or exponential with ema=1) is above the slow one at the "
        "close, flat (cash) otherwise. With vol_target, hold vol_target / realised volatility of the "
        "strategy's capital, at most all of it, rebalanced only past the band. Market orders, charged the taker fee."
    ),
    param_grid={"fast": [20, 50, 100], "slow": [100, 150, 200, 250]},
    default_params={"fast": 50, "slow": 200, "ema": 0, "vol_target": 0.0, "vol_lookback_days": 30},
    known_weaknesses=(
        "Whipsaws in sideways markets pay the 0.8% taker fee twice per round trip. "
        "Lags turns by weeks, so it gives back part of every rally."
    ),
)


class TrendFilterConfig(LongFlatConfig):
    def __init__(self, *, fast: int = 50, slow: int = 200, ema: int = 0, vol_target: float = 0.0,
                 vol_lookback_days: int = 30, **kwargs) -> None:
        # A converted perp (P2-1b, sizing="central") trades entries and exits, never rebalances.
        converted = kwargs.get("sizing") == "central" and kwargs.get("market", "spot") != "spot"
        if vol_target and kwargs.get("rebalance_band") is None and not converted:
            kwargs["rebalance_band"] = 0.25
        super().__init__(**kwargs)
        if not 1 < fast < slow:
            raise ValueError(f"need 1 < fast < slow, got fast={fast} slow={slow}")
        if ema not in (0, 1):
            raise ValueError("ema is 1 (exponential averages) or 0 (simple)")
        if not 0 <= vol_target <= 5:
            raise ValueError(f"vol_target {vol_target} outside [0, 5]; use a fraction, e.g. 0.4 for 40% a year")
        if vol_lookback_days < 10:
            raise ValueError("vol_lookback_days must be at least 10")
        self.fast = fast
        self.slow = slow
        self.ema = ema
        self.vol_target = vol_target
        self.vol_lookback_days = vol_lookback_days


class _Ema:
    """Exponential average, alpha = 2 / (span + 1), seeded with the first close and ready after
    `span` closes (pandas ewm(span, adjust=False, min_periods=span))."""

    def __init__(self, span: int) -> None:
        self.alpha, self.span, self.count, self.value = 2 / (span + 1), span, 0, 0.0

    def update_raw(self, x: float) -> None:
        self.value = x if self.count == 0 else self.alpha * x + (1 - self.alpha) * self.value
        self.count += 1

    @property
    def initialized(self) -> bool:
        return self.count >= self.span


class TrendFilter(LongFlatStrategy):
    def __init__(self, config: TrendFilterConfig) -> None:
        super().__init__(config)
        make = _Ema if config.ema else Sma
        self.fast, self.slow = make(config.fast), make(config.slow)
        per_day = 1440 / bar_minutes(config.bar_type)
        self._ann = math.sqrt(365 * per_day)
        self._rets: deque[float] = deque(maxlen=int(config.vol_lookback_days * per_day))
        self._min_rets = int(10 * per_day)
        self._prev_close: float | None = None
        self._vol: float | None = None
        self._sum = self._sumsq = 0.0  # running sums over self._rets, so each bar costs O(1)
        self._since_exact = 0

    @classmethod
    def warmup_needed(cls, params: dict, bar_minutes: int) -> int:
        # An exponential average settles over ten spans (indicators.settle_bars), a simple one over its own
        # length; volatility needs its whole window.
        slow = int(params.get("slow", 200))
        need = settle_bars(slow) if params.get("ema") else slow
        if params.get("vol_target"):
            need = max(need, int(int(params.get("vol_lookback_days", 30)) * 1440 / bar_minutes) + 1)
        return need

    def update_indicators(self, bar: Bar) -> None:
        c = bar.close.as_double()
        for avg in (self.fast, self.slow):
            avg.update_raw(c)
        prev, self._prev_close = self._prev_close, c
        if not self._cfg.vol_target or not prev:
            return  # volatility only sizes vol-targeted positions; skip it otherwise
        r = c / prev - 1
        if len(self._rets) == self._rets.maxlen:
            old = self._rets[0]
            self._sum -= old
            self._sumsq -= old * old
        self._rets.append(r)
        self._sum += r
        self._sumsq += r * r
        self._since_exact += 1
        if self._since_exact >= self._rets.maxlen:  # running sums drift; recompute them once a window
            self._sum, self._sumsq, self._since_exact = sum(self._rets), sum(x * x for x in self._rets), 0
        n = len(self._rets)
        if n >= max(self._min_rets, 2):
            var = max(self._sumsq - self._sum * self._sum / n, 0.0) / (n - 1)
            self._vol = math.sqrt(var) * self._ann
        else:
            self._vol = None

    def want_long(self, bar: Bar) -> bool | None:
        if not (self.fast.initialized and self.slow.initialized):
            return None
        return self.fast.value > self.slow.value

    @classmethod
    def weight_sized(cls, params: dict) -> bool:
        return bool(params.get("vol_target"))  # without a volatility target it is all or nothing

    def target_weight(self, bar: Bar) -> float | None:
        on = self.want_long(bar)
        if on is None or not self._cfg.vol_target:
            return None if on is None else float(on)
        if not on or self._vol is None:
            return 0.0  # no volatility estimate yet: stay out rather than guess a size
        return 1.0 if self._vol <= 0 else min(1.0, self._cfg.vol_target / self._vol)

    def explain(self, bar: Bar, target: bool) -> tuple[str, dict]:
        f, s, c = self.fast.value, self.slow.value, self._cfg
        kind = "exponential" if c.ema else ""
        rel = "above" if self.fast.value > self.slow.value else "below"
        gap = f / s - 1 if s else 0.0
        label = "ema" if c.ema else "sma"
        text = (f"{c.fast}-bar {kind + ' ' if kind else ''}average {f:,.6g} is {rel} the {c.slow}-bar average "
                f"{s:,.6g} ({gap:+.2%})")
        values = {f"{label}_{c.fast}": f, f"{label}_{c.slow}": s, "gap": gap}
        if c.vol_target and target and self._vol:
            w = min(1.0, c.vol_target / self._vol) if self._vol > 0 else 1.0
            text += (f"; volatility is {self._vol:.0%} a year against a {c.vol_target:.0%} target, "
                     f"so hold {w:.0%} of its capital")
            values.update(volatility=self._vol, vol_target=c.vol_target)
        return text, values
