"""Test strategy for checking entries, exits and fees: buy at the start, sell after a rise, wait for a
dip, buy again. It decides on bar closes only, so a backtest and paper on the same bars make the same
decisions and any difference between them is in the fills and fees, which is what it is there to show."""

from __future__ import annotations

from nautilus_trader.model import Bar

from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy

SPEC = IdeaSpec(
    summary="Buys at the start, sells once the close rises {rise} above the close it bought on (0.01 = 1%), "
            "then buys again once the close falls {dip} below the close it sold on.",
    name="ping_pong",
    family="test",
    idea=(
        "When it starts it buys the underlying and sells after a 1% rise; then it shorts the same instrument "
        "with a 0.5% take-profit, and buys again."
    ),
    rules=(
        "A cycle on bar closes. Long leg: buy on the first bar; sell when the close is rise above the close of "
        "the bar it bought on. Short leg: from the bar it sold on until the close is dip below that bar's close, "
        "then buy again. On a perpetual with shorts allowed the short leg is a short position (the long is sold "
        "and the short opened on the same close); on spot it is held flat: same timing, no position. Measured from bar closes rather than fill prices, so backtest and paper "
        "make the same decisions. All of its capital in or out; no stop-loss unless one is set."
    ),
    data_needs="1-minute OHLCV",
    default_risk_profile="balanced",
    default_params={"rise": 0.01, "dip": 0.005},
    known_weaknesses=(
        "A test of the plumbing, not an idea: at a 0.80% taker fee a 1% rise doesn't pay for the round "
        "trip. No stop-loss by default, so a falling market holds the long leg until it recovers."
    ),
)


class PingPongConfig(LongFlatConfig):
    def __init__(self, *, rise: float = 0.01, dip: float = 0.005, **kwargs) -> None:
        super().__init__(**kwargs)
        for label, v in (("rise", rise), ("dip", dip)):
            if not 0 < v < 0.5:
                raise ValueError(f"{label} {v} outside (0, 0.5); use a fraction, e.g. 0.01 for 1%")
        self.rise, self.dip = rise, dip


class PingPong(LongFlatStrategy):
    def __init__(self, config: PingPongConfig) -> None:
        super().__init__(config)
        self.c = config
        # The leg the cycle is on (+1 long, -1 short, None before the first bar) and the close it started from.
        self._side: int | None = None
        self._ref: float | None = None
        self._why = ("", {})

    def on_start(self) -> None:
        super().on_start()
        if self.runtime is None or self.runtime.backtest:
            return
        # After a restart, pick the cycle up from the journal: long from the average entry, or waiting to buy
        # again from the last sell. With no fills yet it starts with a buy, as on the first start.
        # On a perpetual a short is picked up from its entry the same way.
        qty, entry = self.runtime.book["qty"], self.runtime.book["entry_px"]
        if qty and entry:
            self._side, self._ref = (1 if qty > 0 else -1), float(entry)
            return
        last = self.runtime.store.fills(self.runtime.name, limit=1)
        if last:
            self._side, self._ref = (-1 if last[0]["side"] == "SELL" else 1), float(last[0]["price"])

    def target_side(self, close: float) -> int:
        """The side the cycle wants from this close: +1 long, -1 short. Moves the cycle on when a leg ends."""
        c = self.c
        if self._side is None:
            self._side, self._ref = 1, close
            self._why = ("Start of the cycle: buys on the first bar", {})
        elif self._side == 1 and close >= self._ref * (1 + c.rise):
            move = close / self._ref - 1
            self._why = (f"Close {close:,.6g} is {move:+.2%} from the {self._ref:,.6g} close it bought on, past the "
                         f"{c.rise:.1%} rise", {"entry_close": self._ref, "move": move})
            self._side, self._ref = -1, close
        elif self._side == -1 and close <= self._ref * (1 - c.dip):
            move = close / self._ref - 1
            self._why = (f"Close {close:,.6g} is {move:+.2%} from the {self._ref:,.6g} close it sold on, past the "
                         f"{c.dip:.1%} dip (the short leg's take-profit)", {"exit_close": self._ref, "move": move})
            self._side, self._ref = 1, close
        return self._side

    def want_side(self, bar: Bar) -> int | None:
        # A perpetual: the short leg is a short (held flat unless allow_short).
        return self.target_side(bar.close.as_double())

    def want_long(self, bar: Bar) -> bool | None:
        # Spot: the short leg is held flat.
        return self.target_side(bar.close.as_double()) == 1

    def explain(self, bar: Bar, target: bool) -> tuple[str, dict]:
        return self._why
