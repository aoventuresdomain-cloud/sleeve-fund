"""A position as the portfolio book sees it (v2 P2-2; day-0 interface note v4): holding_for() is the one seam from a
strategy's position to a book Holding, and its open risk is open_risk.position_risk, the interim per-strategy
check's own formula (Advisor 17:20 UK), so the gate, the interim check and the dashboard tile never differ.

Trailing stops (Advisor 18:55 and 19:00 UK): a trail enforced as a resting or every-tick stop is passed as that stop's
current level; a trail checked only at the candle close is no stop at all (stop_price None: the stopless measure),
unless a hard stop rests too, which is then the stop that bounds it."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sleeve_fund.money import money
from sleeve_fund.open_risk import position_risk
from sleeve_fund.portfolio.limits import Holding

# A venue's own code for an asset, mapped to the common one: the same map as research.holdout's, kept here so the core
# imports nothing heavy (Q199-1); a test keeps the two equal.
ALIASES = {"XBT": "BTC", "XDG": "DOGE"}


def underlying(instrument: str) -> str:
    """The underlying an instrument's net counts in, across venues and contract types: the base of BASE/QUOTE, with
    a venue's own code for it mapped to the common one."""
    base = instrument.split("/")[0].strip().upper()
    return ALIASES.get(base, base)


@dataclass(frozen=True)
class Position:
    """One strategy's open position. side: +1 long, -1 short; qty: unsigned. leverage: None for spot (its margin is
    its full notional). stop_price: the stop resting for it now (a trail enforced as a stop at its current level), or
    None when nothing rests."""

    strategy: str
    instrument: str
    venue: str
    side: int
    qty: Decimal
    leverage: float | None = None
    stop_price: Decimal | None = None

    def __post_init__(self):
        object.__setattr__(self, "qty", money(self.qty, "qty"))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", money(self.stop_price, "stop_price"))
        if self.side not in (1, -1):
            raise ValueError(f"side is +1 or -1, not {self.side!r}")


def holding_for(position: Position, mark, atr_pct: float | None) -> Holding:
    """The book's Holding for `position` at `mark`: signed notional, margin (notional / leverage on a perp, the full
    notional on spot) and open risk from the mark (open_risk.position_risk: to the resting stop, or the stopless
    measure when none rests or the price is through it). Costs and slippage stay in sizing, never in the open-risk
    measure (HoE 20:12 UK, note v4 section 2). Spot counts too (Advisor GATE-SPOT, 20:20 UK). Raises ValueError when
    a stopless position's daily ATR isn't known."""
    mark = money(mark, "mark")
    notional = position.side * position.qty * mark
    margin = abs(notional) / Decimal(repr(float(position.leverage))) if position.leverage else abs(notional)
    risk = position_risk(position.side * position.qty, mark, position.stop_price, atr_pct)
    return Holding(underlying(position.instrument), notional, margin, risk)
