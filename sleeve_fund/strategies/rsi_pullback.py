"""Worked example of a multi-indicator idea: buy oversold dips in an uptrend on a volume spike,
exit on an ATR trailing stop. Shows how a plain-English idea maps onto the template."""

from __future__ import annotations

from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model import Bar

from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.indicators import AtrSma, Rsi, Sma, settle_bars

SPEC = IdeaSpec(
    summary="Buys when RSI is below {rsi_entry}, price is above its {ema_period}-bar EMA and volume is over {vol_mult}x normal; exits on a {atr_mult} ATR trailing stop.",
    name="rsi_pullback",
    family="mean-reversion-in-trend",
    idea=(
        "Buy when RSI is below 30, price is above the 200 EMA and volume spikes; "
        "get out on an ATR trailing stop."
    ),
    rules=(
        "Entry (all must hold at the bar close): RSI(rsi_period) < rsi_entry; close > EMA(ema_period); "
        "volume > vol_mult x SMA(volume, 20). Exit: close falls below the highest close since entry "
        "minus atr_mult x ATR(atr_period). All of its capital in or out; market orders at the taker fee."
    ),
    param_grid={"rsi_entry": [25, 30, 35], "atr_mult": [2.0, 3.0, 4.0]},
    default_params={"rsi_entry": 30, "atr_mult": 3.0},
    known_weaknesses=(
        "Few signals on daily bars (oversold inside an uptrend is rare), so it may not reach the trade-count bar. "
        "The trailing stop is checked at the close, so gaps go through it."
    ),
)


class RsiPullbackConfig(LongFlatConfig):
    def __init__(self, *, rsi_period: int = 14, rsi_entry: float = 30, ema_period: int = 200,
                 vol_period: int = 20, vol_mult: float = 1.5, atr_period: int = 14, atr_mult: float = 3.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0 < rsi_entry < 100:
            raise ValueError("rsi_entry is on the usual 0 to 100 scale")
        if min(rsi_period, ema_period, vol_period, atr_period) < 2 or vol_mult <= 0 or atr_mult <= 0:
            raise ValueError("periods must be >= 2 and multipliers positive")
        self.rsi_period, self.rsi_entry, self.ema_period = rsi_period, rsi_entry, ema_period
        self.vol_period, self.vol_mult = vol_period, vol_mult
        self.atr_period, self.atr_mult = atr_period, atr_mult


class RsiPullback(LongFlatStrategy):
    @classmethod
    def warmup_needed(cls, params: dict, bar_minutes: int) -> int:
        # Wilder's RSI and the EMA settle over ten lengths; the ATR is a simple average of true ranges, so it
        # needs only its length and one bar more, but ten lengths are asked for it too (harmless, never short).
        p = {"rsi_period": 14, "ema_period": 200, "vol_period": 20, "atr_period": 14, **params}
        return max(settle_bars(int(p["rsi_period"])), settle_bars(int(p["ema_period"])),
                   settle_bars(int(p["atr_period"])), int(p["vol_period"]) + 1)

    def __init__(self, config: RsiPullbackConfig) -> None:
        super().__init__(config)
        self.c = config
        self.rsi = Rsi(config.rsi_period)  # the standard RSI, as the chart draws it
        self.ema = ExponentialMovingAverage(config.ema_period)
        self.vol = Sma(config.vol_period)
        self.atr = AtrSma(config.atr_period)
        self._prev_vol_avg = None
        self._peak = None

    def update_indicators(self, bar: Bar) -> None:
        # The volume spike compares this bar with the average of the bars before it.
        self._prev_vol_avg = self.vol.value if self.vol.initialized else None
        self.rsi.handle_bar(bar)
        self.ema.handle_bar(bar)
        self.vol.update_raw(bar.volume.as_double())
        self.atr.handle_bar(bar)

    def want_long(self, bar: Bar) -> bool | None:
        if not (self.rsi.initialized and self.ema.initialized and self.atr.initialized and self._prev_vol_avg):
            return None
        close = bar.close.as_double()
        if self._is_long():
            self._peak = max(self._peak or close, close)
            return close >= self._peak - self.c.atr_mult * self.atr.value  # False = trailing stop hit
        self._peak = None
        entry = (
            self.rsi.value < self.c.rsi_entry
            and close > self.ema.value
            and bar.volume.as_double() > self.c.vol_mult * self._prev_vol_avg
        )
        if entry:
            self._peak = close
        return entry

    def explain(self, bar: Bar, target: bool) -> tuple[str, dict]:
        c, close = self.c, bar.close.as_double()
        rsi, vol, avg = self.rsi.value, bar.volume.as_double(), self._prev_vol_avg or 0.0
        values = {"rsi": rsi, f"ema_{c.ema_period}": self.ema.value, "volume_x": vol / avg if avg else None,
                  "atr": self.atr.value}
        if target:
            return (f"RSI {rsi:.1f} below {c.rsi_entry:g}, close {close:,.6g} above the {c.ema_period}-bar EMA "
                    f"{self.ema.value:,.6g}, volume {vol / avg:.2f}x normal (needs {c.vol_mult:g}x)", values)
        stop = (self._peak or close) - c.atr_mult * self.atr.value
        values.update(peak=self._peak, trail_stop=stop)
        return (f"Trailing stop: close {close:,.6g} fell below {stop:,.6g} (peak {self._peak or close:,.6g} minus "
                f"{c.atr_mult:g} x ATR {self.atr.value:,.4g})", values)
