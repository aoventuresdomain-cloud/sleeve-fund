"""Benchmark: buy on the first bar and hold. Pays the same entry fee as any strategy."""

from __future__ import annotations

from nautilus_trader.model import Bar

from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy

SPEC = IdeaSpec(
    name="buy_and_hold",
    family="benchmark",
    idea="Buy and hold.",
    rules="Buy with the whole sleeve on the first bar; never sell.",
)


class BuyAndHoldConfig(LongFlatConfig):
    pass


class BuyAndHold(LongFlatStrategy):
    def want_long(self, bar: Bar) -> bool | None:
        return True
