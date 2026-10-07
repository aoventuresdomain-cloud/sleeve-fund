"""The portfolio limits across the whole book (v2 P2-2, step A), as pure functions: the PM's accepted limits (6 Oct
14:17) and the Independent Quant Advisor's ruling on how they act (6 Oct ~21:54, quant-review/advisor-rulings.md).

decide() answers one order that raises a strategy's position (an entry, an add, a reversal's open leg): the largest
quantity, rounded down to the venue's step, that keeps the book within every limit. Below the venue minimum it is
rejected. It never sees an exit, a stop, a reduce-only order, a close-now or a liquidation: those never go to it.
book_breach() is the book-wide drawdown halt and daily pause. Paper and backtest call the same functions with the
same inputs, so they give the same answer (QA's parity pins).

Money and quantities are exact Decimals (sleeve_fund.money, PE1's #196; day-0 interface note v4): a float for money is a
TypeError, NaN or Infinity a ValueError, both raised when the dataclass is built. Limits compare exactly; the
figures in a decision are "x book" ratios and stay float."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal

from sleeve_fund.money import money, scale
from sleeve_fund.risk import PortfolioProfile

# The order limits are checked in, so a tie names the same limit every time (QA's limit_hit names).
LIMITS = ("gross", "net_instrument", "margin", "open_risk")
WORDS = {"gross": "gross exposure", "net_instrument": "net exposure in {u}", "margin": "margin used",
         "open_risk": "open risk"}
ZERO = Decimal(0)


def _money_fields(obj, *names: str) -> None:
    for n in names:
        object.__setattr__(obj, n, money(getattr(obj, n), n))


@dataclass(frozen=True)
class Holding:
    """One position, or a resting entry not filled yet (a reservation, counted as if filled at its expected price),
    as the book sees it. notional: signed, in the quote currency at the latest mark (long > 0). margin: posted
    isolated margin for a perp, the full notional for spot. risk: its open risk, from portfolio.holding_for (the one
    formula, open_risk.position_risk)."""

    underlying: str
    notional: Decimal
    margin: Decimal
    risk: Decimal

    def __post_init__(self):
        _money_fields(self, "notional", "margin", "risk")


@dataclass(frozen=True)
class Book:
    """The whole fund's marked equity (every strategy's, running or not, plus unallocated cash; Advisor 3) and what it
    holds, resting entries included."""

    equity: Decimal
    holdings: tuple[Holding, ...] = ()

    def __post_init__(self):
        _money_fields(self, "equity")

    def gross(self) -> Decimal:
        return sum((abs(h.notional) for h in self.holdings), ZERO)

    def net(self, underlying: str) -> Decimal:
        return sum((h.notional for h in self.holdings if h.underlying == underlying), ZERO)

    def margin(self) -> Decimal:
        return sum((h.margin for h in self.holdings), ZERO)

    def open_risk(self) -> Decimal:
        return sum((h.risk for h in self.holdings), ZERO)


@dataclass(frozen=True)
class Intent:
    """An order that raises a position. side: +1 buys, -1 sells. price: its expected fill, close x (1 ± half spread)
    for the side, the same number sizing used (Advisor 17:20 UK). Per unit of quantity: margin_per_unit (price /
    leverage on a perp, price on spot) and risk_per_unit (from the expected fill to its planned stop, or the stopless
    measure; open_risk.position_risk, as holding_for uses it), from sizing's intent_for. Both are above 0 for any order
    that raises a position: a zero would switch its limit off, so it is refused (CR208-1). Fees and slippage stay in
    sizing, out of the open-risk measure (HoE 20:12 UK). step and min_qty: the venue's quantity step and minimum."""

    underlying: str
    side: int
    qty: Decimal
    price: Decimal
    margin_per_unit: Decimal
    risk_per_unit: Decimal
    step: Decimal
    min_qty: Decimal

    def __post_init__(self):
        _money_fields(self, "qty", "price", "margin_per_unit", "risk_per_unit", "step", "min_qty")


@dataclass(frozen=True)
class Decision:
    outcome: str  # "approved" | "trimmed" | "rejected"
    approved_qty: Decimal  # what may be sent: the request, less, or 0
    requested_qty: Decimal
    limit_hit: str | None  # the limit that bound it, if any (LIMITS, or a portfolio block)
    reason: str  # one sentence for the journal and the strategy page
    figures: dict = field(default_factory=dict)  # each limit: book before, after (at approved_qty), cap; x book


def floor_to(qty: Decimal, step: Decimal) -> Decimal:
    """`qty` rounded down to a whole number of `step`s (quantities round down to the lot); 0 when not above 0."""
    if qty <= 0:
        return ZERO
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


def rejected(requested: Decimal, limit_hit: str | None, reason: str, figures: dict | None = None) -> Decision:
    return Decision("rejected", ZERO, requested, limit_hit, reason, figures or {})


def _room(used: Decimal, cap: Decimal, per_unit: Decimal) -> Decimal | None:
    """Most quantity a figure that only rises with it can take before passing its cap; None when it doesn't rise."""
    if per_unit <= 0:
        return None
    return max(cap - used, ZERO) / per_unit


def _net_room(net: Decimal, cap: Decimal, signed_per_unit: Decimal) -> Decimal | None:
    """Most quantity before |net| passes the cap, where the order only binds if it raises |net| past it (Advisor 2):
    it may take |net| to the larger of the cap and where it stood. An order against the net first brings it down,
    through zero, then up the other side."""
    if signed_per_unit == 0:
        return None
    bound = max(cap, abs(net))
    if net * signed_per_unit < 0:  # against the net
        return (abs(net) + bound) / abs(signed_per_unit)
    return max(bound - abs(net), ZERO) / abs(signed_per_unit)


def decide(book: Book, intent: Intent, profile: PortfolioProfile) -> Decision:
    """The largest quantity of `intent`, at most what it asks and rounded down to the step, that keeps gross, net in
    its underlying, margin and open risk within the profile; rejected if that is below the venue minimum. A figure
    already over its limit from marks alone forces nothing down: only what raises it is held back (Advisor 2). A
    landing exactly on a limit passes. Inputs that can't be used fail closed."""
    if (book.equity <= 0 or intent.qty <= 0 or intent.price <= 0 or intent.step <= 0 or intent.side not in (1, -1)
            or intent.margin_per_unit <= 0 or intent.risk_per_unit <= 0 or intent.min_qty < 0):
        return rejected(intent.qty, None, "Entry rejected: the book or the order had a figure that can't be used "
                                          f"(book {book.equity}, order {intent!r}), so nothing is sent")
    e, u = book.equity, intent.underlying
    used = {"gross": book.gross(), "net_instrument": book.net(u), "margin": book.margin(),
            "open_risk": book.open_risk()}
    caps = {"gross": scale(e, profile.gross), "net_instrument": scale(e, profile.net_instrument),
            "margin": scale(e, profile.margin), "open_risk": scale(e, profile.open_risk)}
    per_unit = {"gross": intent.price, "net_instrument": intent.side * intent.price,
                "margin": intent.margin_per_unit, "open_risk": intent.risk_per_unit}
    room = {k: (_net_room(used[k], caps[k], per_unit[k]) if k == "net_instrument"
                else _room(used[k], caps[k], per_unit[k])) for k in LIMITS}
    bound = [k for k in LIMITS if room[k] is not None and room[k] < intent.qty]
    limit = min(bound, key=lambda k: room[k]) if bound else None  # min() keeps the first of LIMITS on a tie
    qty = intent.qty if limit is None else floor_to(room[limit], intent.step)

    def figures(at: Decimal) -> dict:
        after = {k: used[k] + at * per_unit[k] for k in LIMITS}
        return {k: {"before": float(used[k] / e), "after": float(after[k] / e), "cap": float(caps[k] / e)}
                for k in LIMITS}

    if limit is None:
        return Decision("approved", qty, intent.qty, None, "Entry within every portfolio limit", figures(qty))
    whole, now, cap = abs(used[limit] + intent.qty * per_unit[limit]) / e, abs(used[limit]) / e, caps[limit] / e
    show = (lambda x: f"{x:.2f}x") if limit in ("gross", "net_instrument") else (lambda x: f"{x:.1%}")
    what = (f"{WORDS[limit].format(u=u)} would be {show(whole)} of the book with the whole order, above the "
            f"{show(cap)} limit (now {show(now)})")
    if qty <= 0 or qty < intent.min_qty:
        return rejected(intent.qty, limit, f"Entry rejected: {what}; the room left, {qty.normalize():f}, is below "
                                           f"the venue minimum of {intent.min_qty.normalize():f}", figures(ZERO))
    return Decision("trimmed", qty, intent.qty, limit,
                    f"Entry trimmed from {intent.qty.normalize():f} to {qty.normalize():f}: {what}", figures(qty))


@dataclass(frozen=True)
class BookBreach:
    action: str  # "halt" | "pause"
    reason: str


def book_breach(profile: PortfolioProfile, equity: Decimal, reference: Decimal, day_start: Decimal) -> BookBreach | None:
    """The book-wide drawdown halt and daily pause, worst first, or None (Advisor 9). reference: the high-water mark
    since the last book reset, or the book at the PM's last Resume after a portfolio halt (it re-bases there, so the
    halt doesn't fire again at once); day_start: the book at 00:00 UTC. Capital added or withdrawn moves both by the
    flow, which the caller applies. At the limit counts. Figures that can't be used fail closed (a halt)."""
    try:
        equity, reference, day_start = (money(equity, "equity"), money(reference, "reference"),
                                        money(day_start, "day_start"))
    except (TypeError, ValueError) as e:  # a float for money (TypeError) fails closed like NaN
        return BookBreach("halt", f"the book's figures aren't usable ({e})")
    if equity <= 0 or reference <= 0 or day_start <= 0:
        return BookBreach("halt", f"the book's figures aren't usable (book {equity}, reference {reference}, "
                                  f"day start {day_start})")
    top = max(reference, equity)
    if equity <= top - scale(top, profile.drawdown):
        return BookBreach("halt", f"the book is {1 - equity / top:.1%} below its reference {top:,.2f}, at the "
                                  f"{profile.drawdown:.0%} portfolio drawdown limit")
    if equity <= day_start - scale(day_start, profile.daily_loss):
        return BookBreach("pause", f"the book has lost {1 - equity / day_start:.1%} since 00:00 UTC, at the "
                                   f"{profile.daily_loss:.0%} daily limit")
    return None
