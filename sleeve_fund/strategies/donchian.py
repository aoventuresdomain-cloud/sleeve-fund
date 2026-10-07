"""Donchian breakout ensemble with volatility targeting (strategy sprint item 4, A1 in the research catalogue,
5 Oct 2026): three daily breakout sub-models, each a third of the position, sized to a volatility target and
traded back to it only past a rebalance band. Long only."""

from __future__ import annotations

import math
from collections import deque

from nautilus_trader.model import Bar

from sleeve_fund.data import bar_minutes
from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.indicators import Donchian as DonchianChannel

SPEC = IdeaSpec(
    summary="Three daily breakouts ({lookbacks} days), each a third of the position, sized to {vol_target:.0%} "
            "volatility a year and exited at its half-length channel low.",
    name="donchian",
    family="trend",
    idea=(
        "Follow trends with several speeds at once: buy a third on a 20-day high, a third on a 55-day high and a "
        "third on a 100-day high, each sold when the close breaks its own half-length low. Size to a volatility "
        "target so quiet and wild markets carry the same risk."
    ),
    rules=(
        "On each day's close, for each lookback N in lookbacks: a sub-model goes long when the close is above the "
        "highest close of the N days before it, and goes flat when the close is below the lowest close of the N/2 "
        "days before it (rounded down). The target weight is the share of sub-models long times vol_target / the "
        "annualised volatility of daily returns over vol_lookback_days, at most all of the capital (spot, no "
        "borrowing). Traded back to that weight only when it drifts more than rebalance_band from the weight last "
        "traded to (20% by default on spot; a perpetual has no rebalancing yet, so it enters at the target and holds). "
        "Market orders, charged the taker fee."
    ),
    data_needs="daily OHLCV",
    default_risk_profile="balanced",
    # The research spec's two volatility targets.
    param_grid={"vol_target": [0.25, 0.40]},
    default_params={"lookbacks": "20,55,100", "vol_target": 0.25, "vol_lookback_days": 90},
    known_weaknesses=(
        "Long whipsaw stretches in sideways markets, where each third buys a high and sells a low. One published "
        "study (2015 to 2025) with no out-of-sample record yet; an independent replication found inconsistencies."
    ),
)


WARMUP_LOOKBACKS = 4  # warm-up length in multiples of the longest lookback (warmup_needed)


def _lookbacks(value) -> list[int]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    try:
        out = [int(str(x).strip()) for x in raw if str(x).strip()]
    except ValueError:
        raise ValueError("lookbacks is a list of whole numbers of days, e.g. 20,55,100") from None
    if not out or any(n < 2 or n > 1000 for n in out) or len(set(out)) != len(out):
        raise ValueError("lookbacks needs one or more different whole numbers of days from 2 to 1,000, e.g. 20,55,100")
    return sorted(out)


class DonchianConfig(LongFlatConfig):
    def __init__(self, *, lookbacks="20,55,100", vol_target: float = 0.25, vol_lookback_days: int = 90,
                 **kwargs) -> None:
        if kwargs.get("market", "spot") == "spot" and kwargs.get("rebalance_band") is None:
            kwargs["rebalance_band"] = 0.2
        super().__init__(**kwargs)
        if self.allow_short:
            raise ValueError("the Donchian ensemble is long only")
        if bar_minutes(self.bar_type) != 1440:
            raise ValueError("the Donchian ensemble decides on daily candles")
        if not 0 < vol_target <= 5:
            raise ValueError(f"vol_target {vol_target} outside (0, 5]; use a fraction, e.g. 0.25 for 25% a year")
        if int(vol_lookback_days) != vol_lookback_days or vol_lookback_days < 10:
            raise ValueError("vol_lookback_days must be a whole number of days, at least 10")
        self.lookbacks = _lookbacks(lookbacks)
        self.vol_target, self.vol_lookback_days = float(vol_target), int(vol_lookback_days)


class Donchian(LongFlatStrategy):
    def __init__(self, config: DonchianConfig) -> None:
        super().__init__(config)
        self.c = config
        self._closes: deque[float] = deque(maxlen=max(config.lookbacks) + 1)  # the day's close last
        self._rets: deque[float] = deque(maxlen=config.vol_lookback_days)
        self._on = {n: False for n in config.lookbacks}
        # The shared channel block (v2 P1-3): each third enters above the high close of the n days before and
        # leaves below the low close of the n // 2 days before, the bar just closed left out of both.
        self._entry = {n: DonchianChannel(n, source="close") for n in config.lookbacks}
        self._exit = {n: DonchianChannel(max(n // 2, 1), source="close") for n in config.lookbacks}
        self._vol: float | None = None
        self._why: tuple[str, dict] = ("", {})

    @classmethod
    def warmup_needed(cls, params: dict, bar_minutes: int) -> int:
        # A third stays on until the close breaks its half-length low, which can be any time after its breakout,
        # so one lookback of warm-up can miss a breakout that is still on (review round 13, E13-5). WARMUP_LOOKBACKS
        # of them reach back past the last time each third was out in all but very long unbroken trends.
        longest = max(_lookbacks(params.get("lookbacks", "20,55,100")))
        return max(WARMUP_LOOKBACKS * longest, int(params.get("vol_lookback_days", 90))) + 1

    def update_indicators(self, bar: Bar) -> None:
        close = bar.close.as_double()
        if self._closes:
            prev = self._closes[-1]
            if prev > 0:
                self._rets.append(math.log(close / prev))
        self._closes.append(close)
        for n in self.c.lookbacks:
            entry, exit_ = self._entry[n], self._exit[n]
            entry.update_raw(close, close, close)
            exit_.update_raw(close, close, close)
            if not entry.initialized:
                continue
            if not self._on[n] and close > entry.values["upper"]:
                self._on[n] = True
            elif self._on[n] and close < exit_.values["lower"]:
                self._on[n] = False
        if len(self._rets) >= max(10, self.c.vol_lookback_days // 2):
            m = sum(self._rets) / len(self._rets)
            var = sum((r - m) ** 2 for r in self._rets) / (len(self._rets) - 1)
            self._vol = math.sqrt(var) * math.sqrt(365)
        else:
            self._vol = None

    def _ready(self) -> bool:
        return len(self._closes) > max(self.c.lookbacks) and self._vol is not None

    def target_weight(self, bar: Bar) -> float | None:
        if not self._ready():
            self._why = ("The channels and the volatility estimate are still filling: no position", {})
            return None
        share = sum(self._on.values()) / len(self._on)
        scale = 1.0 if self._vol <= 0 else min(1.0, self.c.vol_target / self._vol)
        w = share * scale
        on = [str(n) for n in self.c.lookbacks if self._on[n]]
        values = {"close": self._closes[-1], "volatility": round(self._vol, 6), "vol_target": self.c.vol_target,
                  "share_long": round(share, 6), **{f"long_{n}d": int(self._on[n]) for n in self.c.lookbacks}}
        if on:
            text = (f"Long on the {', '.join(on)}-day breakout{'s' if len(on) > 1 else ''} ({share:.0%} of the "
                    f"ensemble); volatility {self._vol:.0%} a year against a {self.c.vol_target:.0%} target, so hold "
                    f"{w:.0%} of the capital")
        else:
            text = "No breakout is on: every third is out below its half-length low, so hold cash"
        self._why = (text, values)
        return w

    def want_long(self, bar: Bar) -> bool | None:
        w = self.target_weight(bar)
        return None if w is None else w > 0

    def explain(self, bar: Bar, target) -> tuple[str, dict]:
        return self._why

    def model_stop_level(self, side: int) -> tuple[float, str] | None:
        """P2-1b, Advisor 6 Oct 17:09 ruling 1: where the whole position would be off, as a long's resting level: the
        lowest of the active thirds' exit channels, each the lowest close of its n // 2 most recent closed bars, the
        bar just closed included."""
        closes = list(self._closes)
        active = [n for n in self.c.lookbacks if self._on[n]]
        if side <= 0 or not active or not closes:
            return None
        n = min(active, key=lambda k: min(closes[-max(k // 2, 1):]))
        level = min(closes[-max(n // 2, 1):])
        return level, (f"at the {n}-day third's exit, the lowest close of the last {max(n // 2, 1)} days "
                       f"({level:,.6g}), the lowest of the active thirds'")
