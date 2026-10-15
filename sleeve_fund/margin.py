"""Isolated-margin arithmetic with no engine imports: the margin a perpetual position puts up, where the venue
liquidates it, and where an entry would be liquidated once it fills. One copy, used by the engine (strategies.base,
markets) and by the pure portfolio core (portfolio.sizing), so the sizing check measures liquidation exactly as the
engine does (Q199-1)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple


class Tier(NamedTuple):
    """One band of a venue's maintenance table (VS-1): positions whose notional is below `ceiling` (and at or above
    the band before's) keep notional x rate - amount. The amount makes the table continuous at every ceiling."""

    ceiling: float
    rate: float
    amount: float


Maintenance = float | Sequence[Tier]  # one rate (a simulated perp's), or a venue's table


def flat(rate: float) -> tuple[Tier, ...]:
    """One rate at every size: the simulated perp's stated terms."""
    return (Tier(math.inf, rate, 0.0),)


def tiers(table: Sequence[Tier]) -> tuple[Tier, ...]:
    """A venue's table, checked: ceilings rising, the last unbounded, rates in (0, 1), and continuous at every ceiling
    (the venue's maintenance amounts make it so; a typo in one is caught here, not at a liquidation)."""
    table = tuple(Tier(float(t.ceiling), float(t.rate), float(t.amount)) for t in table)
    if not table or table[-1].ceiling != math.inf:
        raise ValueError("a maintenance table ends with an unbounded tier")
    for a, b in zip(table, table[1:]):
        if not a.ceiling < b.ceiling:
            raise ValueError(f"maintenance tiers must rise: {a.ceiling:,.0f} then {b.ceiling:,.0f}")
        if not math.isclose(a.ceiling * a.rate - a.amount, a.ceiling * b.rate - b.amount, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"maintenance tiers aren't continuous at {a.ceiling:,.0f}: check the maintenance amounts")
    if any(not 0 < t.rate < 1 for t in table):
        raise ValueError("a maintenance rate is a share of the notional, between 0 and 1")
    return table


def _table(maintenance: Maintenance) -> tuple[Tier, ...]:
    return flat(float(maintenance)) if isinstance(maintenance, int | float) else tuple(maintenance)


def tier_at(maintenance: Maintenance, notional: float) -> Tier:
    """The tier a notional falls in: the first whose ceiling is above it, so a notional exactly on a ceiling takes the
    higher tier (Advisor, VS-1 (b)(1))."""
    table = _table(maintenance)
    return next((t for t in table if abs(notional) < t.ceiling), table[-1])


def isolated_margin(qty: float, entry: float, leverage: float, balance: float | None = None) -> float:
    """The margin an isolated perpetual position puts up: its notional at entry over the leverage it is
    opened at (the risk profile's cap), never more than the balance there is to put up. The rest of the
    strategy's equity is not at risk to the venue's liquidation. The one margin figure for paper,
    backtest, the dashboard and the demo copy (set to isolated at the same leverage)."""
    margin = abs(qty) * entry / max(leverage, 1e-9)
    return min(margin, max(balance, 0.0)) if balance is not None else margin


def liquidation_price(cash: float, qty: float, maintenance: Maintenance) -> float | None:
    """The price at which a position's equity (cash + qty x price) falls to the maintenance margin on
    its value, cash being the margin backing it less qty x entry (isolated_liquidation). None when flat,
    or when no positive price liquidates it (a long fully paid for in cash).

    maintenance is one rate or a venue's tiers (VS-1). The notional moves with the price, so each tier's candidate is
    kept only where its own notional falls in that tier (Advisor, VS-1 (b)(1)); the table is continuous, so one does."""
    if qty == 0:
        return None
    floor = 0.0
    for t in _table(maintenance):
        # cash + qty * p = rate * |qty| * p - amount  =>  p = (cash + amount) / (rate * |qty| - qty)
        denom = t.rate * abs(qty) - qty
        p = (cash + t.amount) / denom if denom else None
        # an open-ended last tier isn't compared: a journal Money can't be compared with infinity, and keeps its type
        if p is not None and p > 0 and floor <= abs(qty) * p and (math.isinf(t.ceiling) or abs(qty) * p < t.ceiling):
            return p
        floor = t.ceiling
    return None


def shifted_liquidation(liq: float | None, qty: float, funded: float, maintenance: Maintenance) -> float | None:
    """The liquidation price once `funded` more (a credit) or less (a payment) backs the position, solved afresh
    from the cash its current price implies, never shifted at the tier it was at (Advisor, VS-1 (b)(2)): a payment
    can carry it into another tier."""
    if liq is None or not qty:
        return liq
    t = tier_at(maintenance, abs(qty) * liq)
    cash = liq * (t.rate * abs(qty) - qty) - t.amount
    return liquidation_price(cash + funded, qty, maintenance)


def isolated_liquidation(cash: float, qty: float, entry: float, leverage: float, maintenance: Maintenance) -> float | None:
    """The liquidation price of a position of qty opened at entry on isolated margin at this leverage.
    cash is the strategy's spot-style cash (its balance less qty x entry). None when flat, or when no
    positive price liquidates it (a long at 1x or less is fully paid for)."""
    if qty == 0 or entry <= 0:
        return None
    margin = isolated_margin(qty, entry, leverage, cash + qty * entry)
    return liquidation_price(margin - qty * entry, qty, maintenance)


def entry_liquidation(cash: float, qty: float, close: float, side: int, fee: float, maintenance: Maintenance,
                      leverage: float) -> tuple[float | None, float]:
    """The liquidation price of an entry of qty at close once it fills (the fee paid, its notional over the
    leverage as its isolated margin: isolated_margin), and how far that is from close as a share of
    it (inf when none)."""
    notional = qty * close  # cash afterwards is spot-style, as the journal keeps it: the fee paid, the notional taken out
    liq = isolated_liquidation(cash - side * notional * (1 + side * fee), side * qty, close, leverage, maintenance)
    return liq, (abs(liq / close - 1) if liq is not None else float("inf"))
