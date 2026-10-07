"""Isolated-margin arithmetic with no engine imports: the margin a perpetual position puts up, where the venue
liquidates it, and where an entry would be liquidated once it fills. One copy, used by the engine (strategies.base,
markets) and by the pure portfolio core (portfolio.sizing), so the sizing check measures liquidation exactly as the
engine does (Q199-1)."""

from __future__ import annotations


def isolated_margin(qty: float, entry: float, leverage: float, balance: float | None = None) -> float:
    """The margin an isolated perpetual position puts up: its notional at entry over the leverage it is
    opened at (the risk profile's cap), never more than the balance there is to put up. The rest of the
    strategy's equity is not at risk to the venue's liquidation. The one margin figure for paper,
    backtest, the dashboard and the demo copy (set to isolated at the same leverage)."""
    margin = abs(qty) * entry / max(leverage, 1e-9)
    return min(margin, max(balance, 0.0)) if balance is not None else margin


def liquidation_price(cash: float, qty: float, maintenance: float) -> float | None:
    """The price at which a position's equity (cash + qty x price) falls to the maintenance margin on
    its value, cash being the margin backing it less qty x entry (isolated_liquidation). None when flat,
    or when no positive price liquidates it (a long fully paid for in cash)."""
    if qty == 0:
        return None
    # cash + qty * p = maintenance * |qty| * p  =>  p = cash / (maintenance * |qty| - qty)
    denom = maintenance * abs(qty) - qty
    if denom == 0:
        return None
    p = cash / denom
    return p if p > 0 else None


def isolated_liquidation(cash: float, qty: float, entry: float, leverage: float, maintenance: float) -> float | None:
    """The liquidation price of a position of qty opened at entry on isolated margin at this leverage.
    cash is the strategy's spot-style cash (its balance less qty x entry). None when flat, or when no
    positive price liquidates it (a long at 1x or less is fully paid for)."""
    if qty == 0 or entry <= 0:
        return None
    margin = isolated_margin(qty, entry, leverage, cash + qty * entry)
    return liquidation_price(margin - qty * entry, qty, maintenance)


def entry_liquidation(cash: float, qty: float, close: float, side: int, fee: float, maintenance: float,
                      leverage: float) -> tuple[float | None, float]:
    """The liquidation price of an entry of qty at close once it fills (the fee paid, its notional over the
    leverage as its isolated margin: isolated_margin), and how far that is from close as a share of
    it (inf when none)."""
    notional = qty * close  # cash afterwards is spot-style, as the journal keeps it: the fee paid, the notional taken out
    liq = isolated_liquidation(cash - side * notional * (1 + side * fee), side * qty, close, leverage, maintenance)
    return liq, (abs(liq / close - 1) if liq is not None else float("inf"))
