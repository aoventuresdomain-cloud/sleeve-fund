"""4-hour dip-buy inside a daily trend (strategy sprint item 3, M4 in the research round 3 list, 5 Oct 2026): the
one pre-registered test of whether holds of hours can pay. In a daily up-trend, buy a sharp 4-hour dip and sell
on the bounce to the short average, after six bars, or at the stop; the short leg mirrors it in a down-trend on
a perpetual."""

from __future__ import annotations

from collections import deque

from nautilus_trader.model import Bar

from sleeve_fund.data import bar_minutes
from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.indicators import AtrSma, Rsi, Sma, settle_bars

DAY_NS = 86_400_000_000_000

SPEC = IdeaSpec(
    summary="In a daily up-trend, buys when 4-hour RSI({rsi_period}) is at or below {rsi_entry:g} and the close is "
            "{dip_atr:g} simple ATR below the day's high; sells at the {exit_sma}-bar average or after {time_stop_bars} "
            "bars. Mirrored short in a down-trend (on a perpetual; held flat on spot).",
    name="dip_buy",
    family="pullback-in-trend",
    idea=(
        "Buy a sharp dip inside a daily up-trend and sell the bounce within a day. The one test of whether "
        "holds of hours can pay after costs: if it fails, shorter holds aren't tried again without new evidence."
    ),
    rules=(
        "Daily regime from the last closed day: up when the close is above its trend_sma_days simple average and "
        "the trend_ema_days exponential average rose that day; down when below and falling. On each decision bar "
        "close (4 hours), with Wilder's RSI(rsi_period) and the simple ATR(dip_atr_bars) of those bars. Flat, up-trend: long "
        "when RSI <= rsi_entry and the close is at least dip_atr simple ATR below the highest high of the last 24 hours. "
        "Flat, down-trend: short when RSI >= 100 - rsi_entry and the close is at least dip_atr simple ATR above the lowest "
        "low of the last 24 hours. A leg ends when the close reaches its exit_sma-bar simple average (at or above "
        "for a long, at or below for a short) or after time_stop_bars bars. The stop comes from the exits (the "
        "sprint's: stop_atr = 2 over atr_bars = 20). Market orders (the research spec's maker entry waits until "
        "maker orders are switched on)."
    ),
    data_needs="1-minute OHLCV, decided on 4-hour candles",
    default_risk_profile="balanced",
    # The research spec's two pre-registered thresholds.
    param_grid={"rsi_entry": [5.0, 10.0]},
    default_params={"rsi_period": 2, "rsi_entry": 10.0, "dip_atr": 1.0, "dip_atr_bars": 20, "exit_sma": 6,
                    "time_stop_bars": 6, "trend_sma_days": 100, "trend_ema_days": 20,
                    "stop_atr": 2.0, "atr_bars": 20},
    known_weaknesses=(
        "High win rate with small wins: a few stops wipe out many bounces, and every round trip pays the fee twice. "
        "Research confidence was 10% at the cheapest fee tier and 3% at a 0.8% taker fee."
    ),
)


class DipBuyConfig(LongFlatConfig):
    def __init__(self, *, rsi_period: int = 2, rsi_entry: float = 10.0, dip_atr: float = 1.0, dip_atr_bars: int = 20,
                 exit_sma: int = 6, time_stop_bars: int = 6, trend_sma_days: int = 100, trend_ema_days: int = 20,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        for label, v, low in (("rsi_period", rsi_period, 2), ("dip_atr_bars", dip_atr_bars, 2),
                              ("exit_sma", exit_sma, 2), ("trend_sma_days", trend_sma_days, 2),
                              ("trend_ema_days", trend_ema_days, 2)):
            if int(v) != v or not low <= v <= 1000:
                raise ValueError(f"{label} must be a whole number from {low} to 1,000")
        if not 0 < rsi_entry < 50:
            raise ValueError("rsi_entry is the oversold level, above 0 and below 50, e.g. 10; the short leg uses 100 minus it")
        if not 0 <= dip_atr <= 10:
            raise ValueError("dip_atr is how many ATRs below the day's high the close must be, from 0 to 10")
        if int(time_stop_bars) != time_stop_bars or time_stop_bars < 0:
            raise ValueError("time_stop_bars is a whole number of bars; 0 means no time stop")
        if bar_minutes(self.bar_type) > 1440 or 1440 % bar_minutes(self.bar_type):
            raise ValueError("the decision candle must divide a day, e.g. 4 hours")
        self.rsi_period, self.rsi_entry, self.dip_atr = int(rsi_period), float(rsi_entry), float(dip_atr)
        self.dip_atr_bars, self.exit_sma, self.time_stop_bars = int(dip_atr_bars), int(exit_sma), int(time_stop_bars)
        self.trend_sma_days, self.trend_ema_days = int(trend_sma_days), int(trend_ema_days)


class _Ema:
    """Exponential average seeded with the first value, as trend_filter's."""

    def __init__(self, span: int) -> None:
        self.alpha, self.span, self.count, self.value = 2 / (span + 1), span, 0, 0.0

    def update_raw(self, x: float) -> None:
        self.value = x if self.count == 0 else self.alpha * x + (1 - self.alpha) * self.value
        self.count += 1

    @property
    def initialized(self) -> bool:
        return self.count >= self.span


class DipBuy(LongFlatStrategy):
    def __init__(self, config: DipBuyConfig) -> None:
        super().__init__(config)
        self.c = config
        per_day = 1440 // bar_minutes(config.bar_type)
        self.rsi = Rsi(config.rsi_period)
        self.atr = AtrSma(config.dip_atr_bars)
        self.exit_avg = Sma(config.exit_sma)
        self._highs: deque[float] = deque(maxlen=per_day)  # the last 24 hours of bars
        self._lows: deque[float] = deque(maxlen=per_day)
        self.day_sma = Sma(config.trend_sma_days)
        self.day_ema = _Ema(config.trend_ema_days)
        self._ema_prev: float | None = None  # the daily EMA the day before
        self._day_close: float | None = None
        self._side = 0  # the leg the rules are on: +1 long, -1 short, 0 flat
        self._held = 0
        self._why: tuple[str, dict] = ("", {})

    @classmethod
    def warmup_needed(cls, params: dict, bar_minutes: int) -> int:
        per_day = 1440 // max(bar_minutes, 1)
        days = max(int(params.get("trend_sma_days", 100)), settle_bars(int(params.get("trend_ema_days", 20)))) + 1
        bars = max(settle_bars(int(params.get("rsi_period", 2))), settle_bars(int(params.get("dip_atr_bars", 20))),
                   int(params.get("exit_sma", 6)), per_day)
        return max(days * per_day, bars)

    def resume_leg(self, side: int, held: int) -> None:
        self._side, self._held = side, held  # after a restart: the leg, and its time stop, from the journal's entry

    def update_indicators(self, bar: Bar) -> None:
        close = bar.close.as_double()
        self.rsi.handle_bar(bar)
        self.atr.handle_bar(bar)
        self.exit_avg.update_raw(close)
        self._highs.append(bar.high.as_double())
        self._lows.append(bar.low.as_double())
        if bar.ts_event % DAY_NS == 0:  # bars are stamped at their close: this one closes a day
            self._ema_prev = self.day_ema.value if self.day_ema.count else None
            self.day_ema.update_raw(close)
            self.day_sma.update_raw(close)
            self._day_close = close

    def regime(self) -> int | None:
        """The daily trend: +1 up, -1 down, 0 neither; None while the daily averages are still filling."""
        if not (self.day_sma.initialized and self.day_ema.initialized) or self._ema_prev is None:
            return None
        rising = self.day_ema.value > self._ema_prev
        if self._day_close > self.day_sma.value and rising:
            return 1
        if self._day_close < self.day_sma.value and self.day_ema.value < self._ema_prev:
            return -1
        return 0

    def target_side(self, close: float, rsi: float, atr: float, high24: float, low24: float, exit_avg: float,
                    regime: int | None) -> int:
        c, ended = self.c, False
        if self._side:
            self._held += 1
            leg = "long" if self._side == 1 else "short"
            if (self._side == 1 and close >= exit_avg) or (self._side == -1 and close <= exit_avg):
                self._side, self._why = 0, (f"The close {close:,.6g} reached the {c.exit_sma}-bar average "
                                            f"{exit_avg:,.6g}: the {leg} ends", {})
            elif c.time_stop_bars and self._held >= c.time_stop_bars:
                self._side, self._why = 0, (f"{self._held} bars without reaching the {c.exit_sma}-bar average: "
                                            f"the {leg} ends on its time stop", {})
            if self._side:
                return self._side
            ended = True
        dip = high24 - close
        rip = close - low24
        if regime is None:
            self._why = ("The daily trend averages are still filling: no position", {})
        elif regime == 1 and rsi <= c.rsi_entry and dip >= c.dip_atr * atr:
            self._side, self._held = 1, 0
            self._why = (f"Daily up-trend, RSI {rsi:.1f} at or below {c.rsi_entry:g} and the close {dip / atr:.2f} simple ATR "
                         f"below the day's high {high24:,.6g}: long until the {c.exit_sma}-bar average", {})
        elif regime == -1 and rsi >= 100 - c.rsi_entry and rip >= c.dip_atr * atr:
            self._side, self._held = -1, 0
            self._why = (f"Daily down-trend, RSI {rsi:.1f} at or above {100 - c.rsi_entry:g} and the close "
                         f"{rip / atr:.2f} simple ATR above the day's low {low24:,.6g}: short until the "
                         f"{c.exit_sma}-bar average", {})
        elif not ended:
            trend = {1: "up-trend", -1: "down-trend", 0: "no daily trend"}[regime]
            self._why = (f"Daily {trend}, RSI {rsi:.1f}: no dip to buy or rally to sell, no position"
                         if regime else f"No daily trend (the close and the {c.trend_ema_days}-day average disagree): "
                         "no position", {})
        return self._side

    def want_side(self, bar: Bar) -> int | None:
        if not (self.rsi.initialized and self.atr.initialized and self.exit_avg.initialized):
            return None
        close, rsi, atr = bar.close.as_double(), self.rsi.value, self.atr.value
        high24, low24 = max(self._highs), min(self._lows)
        regime = self.regime()
        side = self.target_side(close, rsi, atr, high24, low24, self.exit_avg.value, regime)
        values = {"rsi": round(rsi, 4), "atr": round(atr, 8), "high_24h": high24, "low_24h": low24,
                  "exit_sma": round(self.exit_avg.value, 8)}
        if regime is not None:
            values.update(regime=regime, day_close=self._day_close, day_sma=round(self.day_sma.value, 8),
                          day_ema=round(self.day_ema.value, 8))
        self._why = (self._why[0], values)
        return side

    def want_long(self, bar: Bar) -> bool | None:
        side = self.want_side(bar)
        return None if side is None else side == 1  # spot: the short leg is held flat

    def explain(self, bar: Bar, target) -> tuple[str, dict]:
        return self._why
