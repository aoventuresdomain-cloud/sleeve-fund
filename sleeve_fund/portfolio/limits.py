"""The portfolio limits across the whole book (v2 P2-2, step A), as pure functions: the PM's accepted limits (6 Oct
14:17) and the Independent Quant Advisor's ruling on how they act (6 Oct ~21:54, quant-review/advisor-rulings.md).

decide() answers one order that raises a strategy's position (an entry, an add, a reversal's open leg): the largest
quantity, rounded down to the venue's step, that keeps the book within every limit. Below the venue minimum it is
refused. It never sees an exit, a stop, a reduce-only order, a close-now or a liquidation: those never go to it.
book_breach() is the book-wide drawdown halt and daily pause. Paper and backtest call the same functions with the
same inputs, so they give the same answer (QA's parity pins)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal

from sleeve_fund.risk import PortfolioProfile

# The order limits are checked in, so a tie names the same limit every time.
LIMITS = ("gross", "net", "margin", "open_risk")
WORDS = {"gross": "gross exposure", "net": "net exposure in {u}", "margin": "margin used", "open_risk": "open risk"}


@dataclass(frozen=True)
class Holding:
    """One position, or a resting entry not filled yet (a reservation, counted as if filled at its limit), as the
    book sees it. notional: signed, in the quote currency at the latest mark (long > 0). margin: posted isolated margin
    for a perp, the full notional for spot. risk: its open risk (loss to its stop from the mark with the exit fee and
    the stop's slippage, or the stopless measure)."""

    underlying: str
    notional: float
    margin: float
    risk: float


@dataclass(frozen=True)
class Book:
    """The whole fund's marked equity (every strategy's, running or not, plus unallocated cash; Advisor 3) and what it
    holds, resting entries included."""

    equity: float
    holdings: tuple[Holding, ...] = ()

    def gross(self) -> float:
        return sum(abs(h.notional) for h in self.holdings)

    def net(self, underlying: str) -> float:
        return sum(h.notional for h in self.holdings if h.underlying == underlying)

    def margin(self) -> float:
        return sum(h.margin for h in self.holdings)

    def open_risk(self) -> float:
        return sum(h.risk for h in self.holdings)


@dataclass(frozen=True)
class Intent:
    """An order that raises a position. side: +1 buys, -1 sells. price: its expected fill (mid plus or minus half the
    spread). Per unit of quantity: margin_per_unit (price / leverage on a perp, price on spot) and risk_per_unit (the
    distance from the expected fill to its planned stop plus the entry and exit fees and the stop's slippage, or the
    stopless measure; Advisor 5). step and min_qty: the venue's quantity step and minimum."""

    underlying: str
    side: int
    qty: float
    price: float
    margin_per_unit: float
    risk_per_unit: float
    step: float
    min_qty: float


@dataclass(frozen=True)
class Decision:
    outcome: str  # "approved" | "trimmed" | "refused"
    qty: float  # what may be sent: the request, less, or 0
    requested: float
    limit: str | None  # the limit that bound it, if any (LIMITS)
    why: str  # one sentence for the journal and the strategy page
    figures: dict = field(default_factory=dict)  # each limit: book before, after (at qty), the cap; all as x book


def _floor(qty: float, step: float) -> float:
    if qty <= 0:
        return 0.0
    s = Decimal(str(step))
    return float((Decimal(repr(qty)) / s).to_integral_value(rounding=ROUND_DOWN) * s)


def _room(used: float, cap: float, per_unit: float) -> float:
    """Most quantity a figure that only rises with it can take before passing its cap; inf when it doesn't rise."""
    if per_unit <= 0:
        return math.inf
    return max(cap - used, 0.0) / per_unit


def _net_room(net: float, cap: float, signed_per_unit: float) -> float:
    """Most quantity before |net| passes the cap, where the order only binds if it raises |net| past it (Advisor 2):
    it may take |net| to the larger of the cap and where it stood. An order against the net first brings it down,
    through zero, then up the other side."""
    if signed_per_unit == 0:
        return math.inf
    bound = max(cap, abs(net))
    if net * signed_per_unit < 0:  # against the net
        return (abs(net) + bound) / abs(signed_per_unit)
    return max(bound - abs(net), 0.0) / abs(signed_per_unit)


def decide(book: Book, intent: Intent, profile: PortfolioProfile) -> Decision:
    """The largest quantity of `intent`, at most what it asks and rounded down to the step, that keeps gross, net in
    its underlying, margin and open risk within the profile; refused if that is below the venue minimum. A figure
    already over its limit from marks alone forces nothing down: only what raises it is held back (Advisor 2). Bad
    inputs fail closed."""
    nums = (book.equity, intent.qty, intent.price, intent.margin_per_unit, intent.risk_per_unit, intent.step,
            intent.min_qty, *(x for h in book.holdings for x in (h.notional, h.margin, h.risk)))
    if (not all(isinstance(x, (int, float)) and math.isfinite(x) for x in nums) or book.equity <= 0
            or intent.qty <= 0 or intent.price <= 0 or intent.step <= 0 or intent.side not in (1, -1)
            or intent.margin_per_unit < 0 or intent.risk_per_unit < 0):
        return Decision("refused", 0.0, intent.qty, None, "Entry refused: the book or the order had a figure that "
                        f"isn't a usable number (book {book.equity!r}, order {intent!r}), so nothing is sent")
    e, u = book.equity, intent.underlying
    used = {"gross": book.gross(), "net": book.net(u), "margin": book.margin(), "open_risk": book.open_risk()}
    caps = {"gross": profile.gross * e, "net": profile.net_per_instrument * e, "margin": profile.margin * e,
            "open_risk": profile.open_risk * e}
    room = {"gross": _room(used["gross"], caps["gross"], intent.price),
            "net": _net_room(used["net"], caps["net"], intent.side * intent.price),
            "margin": _room(used["margin"], caps["margin"], intent.margin_per_unit),
            "open_risk": _room(used["open_risk"], caps["open_risk"], intent.risk_per_unit)}
    binding = min(LIMITS, key=lambda k: room[k])  # first in LIMITS on a tie
    limit = binding if room[binding] < intent.qty else None
    qty = intent.qty if limit is None else _floor(room[binding], intent.step)
    after = {"gross": used["gross"] + qty * intent.price, "net": used["net"] + intent.side * qty * intent.price,
             "margin": used["margin"] + qty * intent.margin_per_unit,
             "open_risk": used["open_risk"] + qty * intent.risk_per_unit}
    figures = {k: {"before": used[k] / e, "after": after[k] / e, "cap": caps[k] / e} for k in LIMITS}
    if limit is None:
        return Decision("approved", qty, intent.qty, None, "Entry within every portfolio limit", figures)
    per_unit = {"gross": intent.price, "net": intent.side * intent.price, "margin": intent.margin_per_unit,
                "open_risk": intent.risk_per_unit}[limit]
    whole, now, cap = abs(used[limit] + intent.qty * per_unit) / e, abs(used[limit]) / e, caps[limit] / e
    show = (lambda x: f"{x:.2f}x") if limit in ("gross", "net") else (lambda x: f"{x:.1%}")
    what = (f"{WORDS[limit].format(u=u)} would be {show(whole)} of the book with the whole order, above the "
            f"{show(cap)} limit (now {show(now)})")
    if qty < intent.min_qty or qty <= 0:
        return Decision("refused", 0.0, intent.qty, limit,
                        f"Entry refused: {what}; the room left, {qty:g}, is below the venue minimum of "
                        f"{intent.min_qty:g}", figures)
    return Decision("trimmed", qty, intent.qty, limit,
                    f"Entry trimmed from {intent.qty:g} to {qty:g}: {what}", figures)


@dataclass(frozen=True)
class BookBreach:
    action: str  # "halt" | "pause"
    reason: str


def book_breach(profile: PortfolioProfile, equity: float, reference: float, day_start: float) -> BookBreach | None:
    """The book-wide drawdown halt and daily pause, worst first, or None (Advisor 9). reference: the high-water mark
    since the last book reset, or the book at the PM's last Resume after a portfolio halt (it re-bases there, so the
    halt doesn't fire again at once); day_start: the book at 00:00 UTC. Capital added or withdrawn moves both by the
    flow, which the caller applies. Bad inputs fail closed."""
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (equity, reference, day_start)):
        return BookBreach("halt", f"the book's figures aren't usable (book {equity!r}, reference {reference!r}, "
                                  f"day start {day_start!r})")
    drawdown = 1 - equity / max(reference, equity)
    if drawdown >= profile.drawdown_halt:
        return BookBreach("halt", f"the book is {drawdown:.1%} below its reference {reference:,.2f}, at the "
                                  f"{profile.drawdown_halt:.0%} portfolio drawdown limit")
    day_loss = 1 - equity / day_start
    if day_loss >= profile.daily_pause:
        return BookBreach("pause", f"the book has lost {day_loss:.1%} since 00:00 UTC, at the "
                                   f"{profile.daily_pause:.0%} daily limit")
    return None
