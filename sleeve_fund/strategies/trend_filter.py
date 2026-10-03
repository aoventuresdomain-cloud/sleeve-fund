"""Idea #1 from the plan: 50/200-day trend filter on BTC and ETH."""

from __future__ import annotations

from nautilus_trader.indicators import SimpleMovingAverage
from nautilus_trader.model import Bar

from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy

SPEC = IdeaSpec(
    summary="Long while the {fast}-bar average is above the {slow}-bar average, otherwise flat in cash.",
    name="trend_filter",
    family="trend",
    idea="Long only when the 50-day average is above the 200-day average.",
    rules=(
        "Daily bars. Long the whole sleeve when SMA(fast) > SMA(slow) at the close, "
        "flat (cash) otherwise. Market orders, charged the taker fee."
    ),
    param_grid={"fast": [20, 50, 100], "slow": [100, 150, 200, 250]},
    default_params={"fast": 50, "slow": 200},
    known_weaknesses=(
        "Whipsaws in sideways markets pay the 0.8% taker fee twice per round trip. "
        "Lags turns by weeks, so it gives back part of every rally."
    ),
)


class TrendFilterConfig(LongFlatConfig):
    def __init__(self, *, fast: int = 50, slow: int = 200, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 1 < fast < slow:
            raise ValueError(f"need 1 < fast < slow, got fast={fast} slow={slow}")
        self.fast = fast
        self.slow = slow


class TrendFilter(LongFlatStrategy):
    def __init__(self, config: TrendFilterConfig) -> None:
        super().__init__(config)
        self.fast = SimpleMovingAverage(config.fast)
        self.slow = SimpleMovingAverage(config.slow)

    def update_indicators(self, bar: Bar) -> None:
        self.fast.handle_bar(bar)
        self.slow.handle_bar(bar)

    def want_long(self, bar: Bar) -> bool | None:
        if not (self.fast.initialized and self.slow.initialized):
            return None
        return self.fast.value > self.slow.value

    def explain(self, bar: Bar, target: bool) -> tuple[str, dict]:
        f, s, c = self.fast.value, self.slow.value, self._cfg
        rel = "above" if target else "below"
        gap = f / s - 1 if s else 0.0
        return (f"{c.fast}-bar average {f:,.6g} is {rel} the {c.slow}-bar average {s:,.6g} ({gap:+.2%})",
                {f"sma_{c.fast}": f, f"sma_{c.slow}": s, "gap": gap})
