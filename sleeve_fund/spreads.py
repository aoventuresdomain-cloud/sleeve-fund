"""Which bid-ask spread a backtest charges, and why.

Backtest bars carry trade prices, so a market order there fills at the bar's price, while a real
one pays the ask to buy and takes the bid to sell. Backtests therefore charge half the spread on
every order that takes liquidity. Paper sleeves subscribe to the venue's quotes, fill on the bid
or ask themselves, and record each instrument's typical spread; backtests use the latest of those
measurements, else the venue profile's cautious assumption, and every result says which.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sleeve_fund.venues import venue as venue_profile


@dataclass(frozen=True)
class SpreadQuote:
    half_spread: float  # a fraction of the mid price
    source: str  # "measured" or "assumed"
    samples: int
    measured_at: datetime | None

    @property
    def text(self) -> str:
        full = f"{2 * self.half_spread:.3%} bid-ask spread"
        if self.source == "measured":
            return f"{full}, the median of {self.samples:,} live quotes on {self.measured_at:%d %b %Y}"
        return f"{full}, assumed; no paper strategy has measured this instrument yet"

    @property
    def short(self) -> str:
        """One line for a result's header: "0.10% spread (assumed)"."""
        how = f"measured {self.measured_at.day} {self.measured_at:%b %Y}" if self.source == "measured" else "assumed"
        return f"{2 * self.half_spread:.2%} spread ({how})"


def resolve(venue: str | None, instrument: str, store=None) -> SpreadQuote:
    """The spread to charge for `instrument` (e.g. "BTC/USD") on `venue` now."""
    profile = venue_profile(venue)
    row = None
    if store is not None:
        try:
            row = store.latest_spread(profile.name, instrument)
        except Exception:  # noqa: BLE001 - an old database without the table: assume
            row = None
    if row:
        return SpreadQuote(row["half_spread"], "measured", row["samples"], row["measured_at"])
    return SpreadQuote(profile.assumed_half_spread, "assumed", 0, None)
