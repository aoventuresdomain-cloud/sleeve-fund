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
    """One strategy's open position, in the seam's shape (day-0 note v4 section 2; QA's P2-1a cell G2). qty: signed
    (long > 0). underlying: what its net counts in (underlying(instrument)). stop: the stop resting for it now (a trail
    enforced as a stop at its current level), or None when nothing rests. leverage: None for spot (its margin is its
    full notional)."""

    strategy: str
    underlying: str
    qty: Decimal
    stop: Decimal | None = None
    leverage: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "qty", money(self.qty, "qty"))
        if self.stop is not None:
            object.__setattr__(self, "stop", money(self.stop, "stop"))


def holding_for(position, mark, atr_pct: float | None) -> Holding:
    """The book's Holding for `position` (a Position, or anything with its strategy, underlying, signed qty and stop;
    leverage defaults to None) at `mark`: signed notional, margin (notional / leverage on a perp, the full notional on
    spot) and open risk from the mark (open_risk.position_risk: to the resting stop, or the stopless measure when none
    rests or the price is through it). Costs and slippage stay in sizing, never in the open-risk measure (HoE 20:12
    UK, note v4 section 2). Spot counts too (Advisor GATE-SPOT, 20:20 UK). Raises ValueError when a stopless
    position's daily ATR isn't known."""
    mark, qty = money(mark, "mark"), money(position.qty, "qty")
    stop = None if position.stop is None else money(position.stop, "stop")
    leverage = getattr(position, "leverage", None)
    notional = qty * mark
    margin = abs(notional) / Decimal(repr(float(leverage))) if leverage else abs(notional)
    return Holding(position.underlying, notional, margin, position_risk(qty, mark, stop, atr_pct))
