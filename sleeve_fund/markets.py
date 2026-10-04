"""Markets a strategy can trade an instrument on: spot (the default, long or flat) or a perpetual future
(long or short, on margin). The engine stays one engine; the market sets the account it trades in, the
costs it pays and the perp's own cash flows (funding) and limits (liquidation).

A perp here is simulated on the venue's live spot prices: research and paper only, with no venue account
(the venue is chosen at G2). Its costs are a low-fee perp venue's published taker and maker rates; the
"perp-venue-fees" market charges the venue's own spot schedule instead, as the stress case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sleeve_fund.instruments import FeeSchedule

SPOT = "spot"
PERP = "perp"
PERP_VENUE_FEES = "perp-venue-fees"
MARKETS = (SPOT, PERP, PERP_VENUE_FEES)
# The simulated venue's leverage for a perp account: above every risk profile's cap (sleeve_fund.risk), so
# our own leverage and liquidation guards, not the simulated venue's margin check, decide.
VENUE_LEVERAGE = 10


@dataclass(frozen=True)
class PerpTerms:
    """What holding a linear (quote-settled) perpetual costs and risks."""

    label: str
    fees: FeeSchedule | None  # None: the venue's own schedule (the stress case)
    half_spread: float | None  # None: the venue's assumption
    funding_rate: float  # per funding interval, as a share of the position's value: longs pay, shorts receive
    funding_hours: tuple[int, ...]  # UTC hours funding is exchanged at
    maintenance_margin: float  # share of the position's value the account must keep, or it is liquidated


# Published rates of the large perp venues' entry tiers (0.02% maker, 0.05% taker), a tight BTC book
# (0.01% half spread), funding at the usual baseline of 0.01% every 8 hours (about 11% a year, paid by
# longs) and a 0.5% maintenance margin, as the long/short verdict (4 Oct 2026) assumed.
LOW_FEE_PERP = PerpTerms("Low-fee perpetual (simulated)", FeeSchedule(Decimal("0.0002"), Decimal("0.0005")),
                         0.0001, 0.0001, (0, 8, 16), 0.005)
VENUE_FEE_PERP = PerpTerms("Perpetual at the venue's spot fees (stress)", None, None, 0.0001, (0, 8, 16), 0.005)


def market_of(params: dict | None) -> str:
    m = (params or {}).get("market") or SPOT
    if m not in MARKETS:
        raise ValueError(f"unknown market {m!r}; choose one of {', '.join(MARKETS)}")
    return m


def is_perp(params: dict | None) -> bool:
    return market_of(params) != SPOT


def terms(params: dict | None) -> PerpTerms | None:
    m = market_of(params)
    return None if m == SPOT else LOW_FEE_PERP if m == PERP else VENUE_FEE_PERP


def fees_for(params: dict | None, venue_fees: FeeSchedule) -> FeeSchedule:
    """The fee schedule a strategy pays: its market's, or the venue's."""
    t = terms(params)
    return t.fees if t is not None and t.fees is not None else venue_fees


def half_spread_for(params: dict | None, venue_half_spread: float) -> float:
    t = terms(params)
    return t.half_spread if t is not None and t.half_spread is not None else venue_half_spread


def funding_times(after: datetime, until: datetime, hours: tuple[int, ...]) -> list[datetime]:
    """The funding times in (after, until], oldest first."""
    out, day = [], datetime(after.year, after.month, after.day, tzinfo=timezone.utc)
    while day <= until:
        for h in hours:
            t = day + timedelta(hours=h)
            if after < t <= until:
                out.append(t)
        day += timedelta(days=1)
    return out


def liquidation_price(cash: float, qty: float, maintenance: float) -> float | None:
    """The price at which a position's equity (cash + qty x price) falls to the maintenance margin on
    its value, with the strategy's whole equity as the position's isolated margin. None when flat, or
    when no positive price liquidates it (a long fully paid for in cash)."""
    if qty == 0:
        return None
    # cash + qty * p = maintenance * |qty| * p  =>  p = cash / (maintenance * |qty| - qty)
    denom = maintenance * abs(qty) - qty
    if denom == 0:
        return None
    p = cash / denom
    return p if p > 0 else None
