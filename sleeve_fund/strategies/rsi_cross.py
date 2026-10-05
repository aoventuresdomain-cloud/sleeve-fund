"""RSI cross-back on 15-minute candles (strategy sprint item 1, 5 Oct 2026): buy when Wilder's RSI crosses back
above the low band after being below it, sell when it reaches the exit band or after a time stop; the short leg
mirrors it on a perpetual. An optional higher-timeframe trend filter only takes legs with the larger trend."""

from __future__ import annotations

from nautilus_trader.model import Bar

from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.indicators import Rsi, Sma, settle_bars

SPEC = IdeaSpec(
    summary="Buys when RSI({rsi_period}) crosses back above {long_entry:g} and sells at {long_exit:g} or after "
            "{time_stop_bars} bars; shorts when it crosses back below {short_entry:g} until {short_exit:g} (on a "
            "perpetual; held flat on spot).",
    name="rsi_cross",
    family="mean-reversion",
    idea=(
        "After an oversold dip, buy the first sign of recovery rather than the falling market: RSI crossing back "
        "above 30. Take a modest bounce (RSI 55), and give up after 12 hours on 15-minute candles."
    ),
    rules=(
        "On each bar close, with Wilder's RSI(rsi_period). Flat: RSI crossing back above long_entry (below it on "
        "the bar before, at or above it now) goes long; RSI crossing back below short_entry goes short. Long: "
        "until RSI >= long_exit or time_stop_bars bars have passed. Short: until RSI <= short_exit or the time "
        "stop. With trend_sma > 0, a long is taken only while the last closed trend_minutes candle closed above "
        "its trend_sma-candle simple average, a short only while it closed below. A leg that ends needs a fresh "
        "cross to start again. Stops come from the exits (the sprint's: stop_atr = 2 over atr_bars = 14)."
    ),
    data_needs="1-minute OHLCV, decided on 15-minute candles",
    default_risk_profile="balanced",
    # The sprint's four pre-registered variants: bands 30/55 or 25/50, with or without the 4-hour SMA(50) filter.
    param_grid={"long_entry": [25.0, 30.0], "trend_sma": [0, 50]},
    default_params={"rsi_period": 14, "long_entry": 30.0, "long_exit": 55.0, "short_entry": 70.0, "short_exit": 50.0,
                    "time_stop_bars": 48, "trend_minutes": 240, "trend_sma": 0},
    known_weaknesses=(
        "Prior lab work lost on every 15-minute idea even at 0.03% a side: many short trades, so fees dominate. "
        "The 25/50 grid point keeps the 55 exit unless long_exit is set too."
    ),
)


class RsiCrossConfig(LongFlatConfig):
    def __init__(self, *, rsi_period: int = 14, long_entry: float = 30.0, long_exit: float = 55.0,
                 short_entry: float = 70.0, short_exit: float = 50.0, time_stop_bars: int = 48,
                 trend_minutes: int = 240, trend_sma: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        if int(rsi_period) != rsi_period or not 2 <= rsi_period <= 500:
            raise ValueError("rsi_period must be a whole number of bars from 2 to 500")
        if not 0 < long_entry < long_exit < 100:
            raise ValueError("the long side needs 0 < entry < exit < 100, e.g. in at 30, out at 55")
        if not 0 < short_exit < short_entry < 100:
            raise ValueError("the short side needs 0 < exit < entry < 100, e.g. in at 70, out at 50")
        if int(time_stop_bars) != time_stop_bars or time_stop_bars < 0:
            raise ValueError("time_stop_bars is a whole number of bars; 0 means no time stop")
        if int(trend_minutes) != trend_minutes or trend_minutes <= 0 or 1440 % trend_minutes:
            raise ValueError("trend_minutes must divide a day, e.g. 240 for 4-hour candles")
        if int(trend_sma) != trend_sma or trend_sma < 0 or trend_sma == 1:
            raise ValueError("trend_sma is a number of candles, at least 2; 0 means no trend filter")
        self.rsi_period, self.time_stop_bars = int(rsi_period), int(time_stop_bars)
        self.long_entry, self.long_exit = long_entry, long_exit
        self.short_entry, self.short_exit = short_entry, short_exit
        self.trend_minutes, self.trend_sma = int(trend_minutes), int(trend_sma)


class RsiCross(LongFlatStrategy):
    def __init__(self, config: RsiCrossConfig) -> None:
        super().__init__(config)
        self.c = config
        self.rsi = Rsi(config.rsi_period)
        self.trend = Sma(config.trend_sma) if config.trend_sma else None
        self._trend_close: float | None = None  # the last closed trend candle's close
        self._prev: float | None = None  # RSI on the bar before
        self._side = 0  # the leg the rules are on: +1 long, -1 short, 0 flat
        self._held = 0  # decision bars the leg has run
        self._why: tuple[str, dict] = ("", {})

    @classmethod
    def warmup_needed(cls, params: dict, bar_minutes: int) -> int:
        need = settle_bars(int(params.get("rsi_period", 14))) + 1  # +1: a cross compares with the bar before
        sma = int(params.get("trend_sma", 0) or 0)
        if sma:
            need = max(need, sma * int(params.get("trend_minutes", 240)) // max(bar_minutes, 1))
        return need

    def resume_leg(self, side: int, held: int) -> None:
        self._side, self._held = side, held  # after a restart: the leg, and its time stop, from the journal's entry

    def update_indicators(self, bar: Bar) -> None:
        self._prev = self.rsi.value if self.rsi.initialized else None
        self.rsi.handle_bar(bar)
        if self.trend is not None and bar.ts_event % (self.c.trend_minutes * 60_000_000_000) == 0:
            # Bars are stamped at their close: this one closes a trend candle.
            close = bar.close.as_double()
            self.trend.update_raw(close)
            self._trend_close = close

    def _trend_allows(self, side: int) -> bool | None:
        """Whether the larger trend allows a leg on this side; None while the trend average is still filling."""
        if self.trend is None:
            return True
        if not self.trend.initialized or self._trend_close is None:
            return None
        return self._trend_close > self.trend.value if side > 0 else self._trend_close < self.trend.value

    def target_side(self, rsi: float, prev: float | None) -> int:
        """The side the rules want from this bar's RSI and the bar before's: +1 long, -1 short, 0 flat."""
        c, ended = self.c, False
        if self._side:
            self._held += 1
            if self._side == 1 and rsi >= c.long_exit:
                self._side, self._why = 0, (f"RSI {rsi:.1f} reached {c.long_exit:g}: the long ends", {})
            elif self._side == -1 and rsi <= c.short_exit:
                self._side, self._why = 0, (f"RSI {rsi:.1f} fell to {c.short_exit:g}: the short ends", {})
            elif c.time_stop_bars and self._held >= c.time_stop_bars:
                leg = "long" if self._side == 1 else "short"
                self._side, self._why = 0, (f"{self._held} bars without reaching the exit: the {leg} ends on its "
                                            "time stop", {})
            if self._side:
                return self._side
            ended = True
        cross = 1 if prev is not None and prev < c.long_entry <= rsi else (
            -1 if prev is not None and prev > c.short_entry >= rsi else 0)
        if cross:
            allowed = self._trend_allows(cross)
            words = (f"RSI crossed back above {c.long_entry:g} ({prev:.1f} to {rsi:.1f})" if cross > 0 else
                     f"RSI crossed back below {c.short_entry:g} ({prev:.1f} to {rsi:.1f})")
            if allowed:
                self._side, self._held = cross, 0
                exit_at = c.long_exit if cross > 0 else c.short_exit
                self._why = (f"{words}: {'long' if cross > 0 else 'short'} until it reaches {exit_at:g}", {})
            else:
                why = "the trend average is still filling" if allowed is None else (
                    f"the {c.trend_minutes // 60}-hour close is {'below' if cross > 0 else 'above'} its "
                    f"{c.trend_sma}-candle average")
                self._why = (f"{words}, but {why}: no position", {})
        elif not ended:
            self._why = (f"RSI {rsi:.1f}: no cross of {c.long_entry:g} or {c.short_entry:g}, no position", {})
        return self._side

    def want_side(self, bar: Bar) -> int | None:
        if not self.rsi.initialized:
            return None
        rsi = self.rsi.value
        side = self.target_side(rsi, self._prev)
        values = {"rsi": round(rsi, 4)}
        if self._prev is not None:
            values["rsi_prev"] = round(self._prev, 4)
        if self.trend is not None and self.trend.initialized:
            values.update(trend_close=self._trend_close, trend_sma=round(self.trend.value, 8))
        self._why = (self._why[0], values)
        return side  # a short is taken only on a perpetual with allow_short

    def want_long(self, bar: Bar) -> bool | None:
        side = self.want_side(bar)
        return None if side is None else side == 1  # spot: the short leg is held flat

    def explain(self, bar: Bar, target) -> tuple[str, dict]:
        return self._why
