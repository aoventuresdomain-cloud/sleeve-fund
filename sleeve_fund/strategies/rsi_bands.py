"""Test strategy: RSI bands. Long when RSI reaches the low band until it recovers to the long exit;
short when it reaches the high band until it falls back to the short exit."""

from __future__ import annotations

from nautilus_trader.model import Bar

from sleeve_fund.strategies.base import Condition, IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.indicators import Rsi, settle_bars

SPEC = IdeaSpec(
    summary="Buys when RSI({rsi_period}) closes at or below {long_entry:g} and sells when it reaches {long_exit:g}; "
            "short at or above {short_entry:g} until it falls to {short_exit:g} (on a perpetual; held flat on spot).",
    name="rsi_bands",
    family="test",
    idea=(
        "If RSI hits 30, long the asset until it reaches 55; short the asset when RSI hits 70 until it comes "
        "back down to 50."
    ),
    rules=(
        "On each bar close, with RSI(rsi_period) on the usual 0 to 100 scale. Flat: RSI <= long_entry goes "
        "long; RSI >= short_entry goes short. Long: until RSI >= long_exit. Short: until RSI <= short_exit. "
        "A leg that ends checks the flat rules on the same bar. The short leg is a short position on a "
        "perpetual with shorts allowed, and held flat on spot. All of its capital in or out; no stop-loss unless "
        "one is set."
    ),
    data_needs="1-minute OHLCV",
    default_risk_profile="balanced",
    param_grid={"long_entry": [25.0, 30.0], "long_exit": [50.0, 55.0]},
    default_params={"rsi_period": 14, "long_entry": 30.0, "long_exit": 55.0, "short_entry": 70.0, "short_exit": 50.0},
    known_weaknesses=(
        "A test strategy. No stop-loss by default, so a long that never sees RSI recover is held. On 1-minute "
        "bars RSI crosses its bands often, so fees dominate."
    ),
)


class RsiBandsConfig(LongFlatConfig):
    def __init__(self, *, rsi_period: int = 14, long_entry: float = 30.0, long_exit: float = 55.0,
                 short_entry: float = 70.0, short_exit: float = 50.0, **kwargs) -> None:
        super().__init__(**kwargs)
        if int(rsi_period) != rsi_period or not 2 <= rsi_period <= 500:
            raise ValueError("rsi_period must be a whole number of bars from 2 to 500")
        if not 0 < long_entry < long_exit < 100:
            raise ValueError("the long side needs 0 < entry < exit < 100, e.g. in at 30, out at 55")
        if not 0 < short_exit < short_entry < 100:
            raise ValueError("the short side needs 0 < exit < entry < 100, e.g. in at 70, out at 50")
        self.rsi_period = int(rsi_period)
        self.long_entry, self.long_exit = long_entry, long_exit
        self.short_entry, self.short_exit = short_entry, short_exit


class RsiBands(LongFlatStrategy):
    def __init__(self, config: RsiBandsConfig) -> None:
        super().__init__(config)
        self.c = config
        self.rsi = Rsi(config.rsi_period)  # the standard RSI, as the chart draws it
        self._side = 0  # the leg the rules are on: +1 long, -1 short, 0 flat
        self._why = ("", {})

    @classmethod
    def warmup_needed(cls, params: dict, bar_minutes: int) -> int:
        return settle_bars(int(params.get("rsi_period", 14)))

    def resume_leg(self, side: int, held: int) -> None:
        self._side = side  # after a restart: the leg the journal's last entry opened

    def update_indicators(self, bar: Bar) -> None:
        self.rsi.handle_bar(bar)

    def _entry_rule(self, side: int, rsi: float) -> Condition:
        """The band that opens `side` from flat: RSI at or below long_entry, or at or above short_entry."""
        c = self.c
        if side == 1:
            return Condition.check(f"RSI({c.rsi_period}) at or below {c.long_entry:g}", rsi, "<=", c.long_entry,
                                   note="opens the long leg")
        return Condition.check(f"RSI({c.rsi_period}) at or above {c.short_entry:g}", rsi, ">=", c.short_entry,
                               note="opens the short leg")

    def _exit_rule(self, leg: int, rsi: float) -> Condition:
        """The band that ends the leg the rules are on: RSI back up to long_exit, or down to short_exit."""
        c = self.c
        if leg == 1:
            return Condition.check(f"RSI({c.rsi_period}) at or above {c.long_exit:g}", rsi, ">=", c.long_exit,
                                   exit=True, note="ends the long leg")
        return Condition.check(f"RSI({c.rsi_period}) at or below {c.short_exit:g}", rsi, "<=", c.short_exit,
                               exit=True, note="ends the short leg")

    def target_side(self, rsi: float) -> int:
        """The side the rules want from this RSI: +1 long, -1 short, 0 flat. Decided by the same rules
        conditions() shows on the Signals tab."""
        c, was = self.c, self._side
        if self._side == 1 and self._exit_rule(1, rsi).met:
            self._side, self._why = 0, (f"RSI {rsi:.1f} reached {c.long_exit:g}: the long leg ends", {})
        elif self._side == -1 and self._exit_rule(-1, rsi).met:
            self._side, self._why = 0, (f"RSI {rsi:.1f} fell to {c.short_exit:g}: the short leg ends", {})
        if self._side == 0:
            if self._entry_rule(1, rsi).met:
                self._side, self._why = 1, (f"RSI {rsi:.1f} at or below {c.long_entry:g}: long until it reaches "
                                            f"{c.long_exit:g}", {})
            elif self._entry_rule(-1, rsi).met:
                self._side, self._why = -1, (f"RSI {rsi:.1f} at or above {c.short_entry:g}: short until it falls to "
                                             f"{c.short_exit:g}", {})
            elif was == 0:
                self._why = (f"RSI {rsi:.1f} between {c.long_entry:g} and {c.short_entry:g}: no position", {})
        return self._side

    def conditions(self, side: int, price: float | None = None) -> list[Condition]:
        """The rules target_side() applies for `side` on this bar: on that leg, the band that ends it;
        otherwise the band that opens it, after the band that ends the other leg when one is on. With
        `price`, on the forming candle's RSI, worked out on a copy (Rsi.peek)."""
        if price is None:
            rsi = self.rsi.value if self.rsi.initialized else None
        else:
            rsi = self.rsi.peek(price)
        if rsi is None:
            return []
        leg = self._side
        if leg == side:
            return [self._exit_rule(side, rsi)]
        rows = []
        if leg == -side:
            end = self._exit_rule(leg, rsi)
            rows.append(Condition(end.name, end.value, end.threshold, end.op, end.met, note=end.note))
        rows.append(self._entry_rule(side, rsi))
        if side == -1 and leg == 0 and self.c.long_entry >= self.c.short_entry:
            # Overlapping bands: target_side checks the long band first, so a short needs RSI above it.
            rows.append(Condition.check(f"RSI({self.c.rsi_period}) above {self.c.long_entry:g}", rsi, ">",
                                        self.c.long_entry, note="the long band is checked first"))
        return rows

    def want_side(self, bar: Bar) -> int | None:
        if not self.rsi.initialized:
            return None
        rsi = self.rsi.value
        side = self.target_side(rsi)
        self._why = (self._why[0], {"rsi": rsi})
        return side  # a short is taken only on a perpetual with allow_short

    def want_long(self, bar: Bar) -> bool | None:
        side = self.want_side(bar)
        return None if side is None else side == 1  # spot: the short leg is held flat

    def explain(self, bar: Bar, target: bool) -> tuple[str, dict]:
        return self._why
