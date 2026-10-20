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
# Bridge Drop A (Advisor COMPLIANCE-5): loss if every stop is hit per asset (T1) and per asset group (T2), checked
# after the four above when the profile sets them, so a tie still names the older limit first.
STOP_TESTS = ("asset_open_risk", "group_open_risk")
WORDS = {"gross": "gross exposure", "net_instrument": "net exposure in {u}", "margin": "margin used",
         "open_risk": "open risk", "asset_open_risk": "loss at the stops in {u}",
         "group_open_risk": "loss at the stops in the {g} group"}
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
    that raises a position: a zero would switch its limit off, so it is refused (CR208-1). risk_per_unit includes the
    taker fee on the stop's notional (Advisor FEE-LIVE 5); slippage stays in sizing. step and min_qty: the venue's
    quantity step and minimum."""

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
    outcome: str  # "approved" | "trimmed" | "rejected"; "error" only in the journal, for a check that couldn't run
    approved_qty: Decimal  # what may be sent: the request, less, or 0
    requested_qty: Decimal
    limit_hit: str | None  # what bound it, if any: LIMITS, "below_min", or a portfolio block (gate.entry_block)
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


def _caps_on(e: Decimal, profile: PortfolioProfile) -> dict[str, Decimal]:
    """The caps of the order limits the PM has on, keyed as LIMITS. A test switched off (None) is off: it has no cap,
    binds nothing and has no figure (B-32, PM 15 Oct), the way _stop_tests already treats T1 and T2. The same holds
    for T3, which can't be switched off today (compliance_rules.LOCKED refuses the save): B-31's locked half is then
    only that switch (HoE, 20 Oct)."""
    return {k: scale(e, getattr(profile, k)) for k in LIMITS if getattr(profile, k) is not None}


class Unmapped(Exception):
    """An entry on an asset the PM's group map leaves out, while T2 is on (Advisor BRIDGE-A-FE-2)."""


def _stop_tests(book: Book, underlying: str, profile: PortfolioProfile) -> tuple[dict, dict, str]:
    """T1 and T2 for an order in `underlying`, as ({test: used}, {test: cap}, its group): the loss at the stops already
    in the asset, and in its group. With no group map at all T2 has nothing to judge. Raises Unmapped for an asset the
    map leaves out (it never borrows another group's room), and whatever a map that can't be read raises: the caller
    refuses both (BRIDGE-3, fail closed)."""
    e = book.equity
    used, caps, group = {}, {}, ""
    if profile.asset_open_risk is not None:
        used["asset_open_risk"] = sum((h.risk for h in book.holdings if h.underlying == underlying), ZERO)
        caps["asset_open_risk"] = scale(e, profile.asset_open_risk)
    groups = profile.groups
    if profile.group_open_risk is not None and groups is not None and len(groups):
        group = groups.get(underlying)
        if group is None:
            raise Unmapped(underlying)
        used["group_open_risk"] = sum((h.risk for h in book.holdings if groups.get(h.underlying) == group), ZERO)
        caps["group_open_risk"] = scale(e, profile.group_open_risk)
    return used, caps, group


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
    try:
        stop_used, stop_caps, group = _stop_tests(book, u, profile)
    except Unmapped:
        return rejected(intent.qty, "group_open_risk", f"Entry rejected: {u} is in no asset group, so the per-group "
                                                       "loss-at-stops test can't judge it; the PM maps it to a group "
                                                       "in Portfolio Settings")
    except Exception as exc:  # noqa: BLE001 - a test that can't be evaluated refuses (BRIDGE-3, fail closed)
        return rejected(intent.qty, "group_open_risk", f"Entry rejected: the loss-at-stops tests couldn't be checked "
                                                       f"({type(exc).__name__}), so nothing is sent")
    caps = {**_caps_on(e, profile), **stop_caps}
    tests = tuple(k for k in LIMITS if k in caps) + tuple(k for k in STOP_TESTS if k in stop_used)
    used = {"gross": book.gross(), "net_instrument": book.net(u), "margin": book.margin(),
            "open_risk": book.open_risk(), **stop_used}
    per_unit = {"gross": intent.price, "net_instrument": intent.side * intent.price,
                "margin": intent.margin_per_unit, "open_risk": intent.risk_per_unit,
                **{k: intent.risk_per_unit for k in stop_used}}
    room = {k: (_net_room(used[k], caps[k], per_unit[k]) if k == "net_instrument"
                else _room(used[k], caps[k], per_unit[k])) for k in tests}
    bound = [k for k in tests if room[k] is not None and room[k] < intent.qty]
    limit = min(bound, key=lambda k: room[k]) if bound else None  # min() keeps the first of LIMITS on a tie
    qty = floor_to(intent.qty if limit is None else room[limit], intent.step)  # both paths on the step (QA F211)

    def figures(at: Decimal) -> dict:
        after = {k: used[k] + at * per_unit[k] for k in tests}
        return {k: {"before": float(used[k] / e), "after": float(after[k] / e), "cap": float(caps[k] / e)}
                for k in tests}

    if limit is None and (qty <= 0 or qty < intent.min_qty):  # nothing binds, but the request is under the minimum
        return rejected(intent.qty, "below_min", f"Entry rejected: {qty.normalize():f} is below the venue minimum "
                                                 f"of {intent.min_qty.normalize():f}", figures(ZERO))
    if limit is None:
        return Decision("approved", qty, intent.qty, None, "Entry within every portfolio limit", figures(qty))
    whole, now, cap = abs(used[limit] + intent.qty * per_unit[limit]) / e, abs(used[limit]) / e, caps[limit] / e
    show = (lambda x: f"{x:.2f}x") if limit in ("gross", "net_instrument") else (lambda x: f"{x:.1%}")
    what = (f"{WORDS[limit].format(u=u, g=group)} would be {show(whole)} of the book with the whole order, above the "
            f"{show(cap)} limit (now {show(now)})")
    if qty <= 0 or qty < intent.min_qty:
        return rejected(intent.qty, limit, f"Entry rejected: {what}; the room left, {qty.normalize():f}, is below "
                                           f"the venue minimum of {intent.min_qty.normalize():f}", figures(ZERO))
    return Decision("trimmed", qty, intent.qty, limit,
                    f"Entry trimmed from {intent.qty.normalize():f} to {qty.normalize():f}: {what}", figures(qty))


def decide_pair(book: Book, first: Intent, second: Intent, profile: PortfolioProfile) -> tuple[Decision, Decision]:
    """A two-leg entry (P2-8) as one position: the same share of both legs, or neither (Advisor, 13 Oct 23:55 UK).
    The second leg is held at its ratio to the first (second.qty / first.qty). Gross and margin count both legs; open
    risk is the sum of the legs' own risk, with no hedge credit (ruling A); net exposure is per underlying, so two legs
    on one underlying net against each other. The share is the largest that keeps every limit; the first leg is
    rounded down to its step and the second sized from it at the ratio and rounded down to its own. Either leg below
    its venue minimum rejects the pair whole. Inputs that can't be used fail closed."""
    legs = (first, second)
    if book.equity <= 0 or any(
            i.qty <= 0 or i.price <= 0 or i.step <= 0 or i.side not in (1, -1) or i.margin_per_unit <= 0
            or i.risk_per_unit <= 0 or i.min_qty < 0 for i in legs):
        why = (f"Entry rejected: the book or a leg had a figure that can't be used (book {book.equity}, legs "
               f"{first!r}, {second!r}), so neither leg is sent")
        return rejected(first.qty, None, why), rejected(second.qty, None, why)
    e, ratio = book.equity, second.qty / first.qty
    unders = list(dict.fromkeys(i.underlying for i in legs))
    on = _caps_on(e, profile)
    # Only the tests the PM has on are figures here: everything below loops over `used` (B-32).
    used = {k: v for k, v in (("gross", book.gross()), ("margin", book.margin()), ("open_risk", book.open_risk()))
            if k in on}
    caps = {k: on[k] for k in used}
    if "net_instrument" in on:
        used.update({f"net:{u}": book.net(u) for u in unders})
        caps.update({f"net:{u}": on["net_instrument"] for u in unders})
    group_of: dict[str, str] = {}  # T1 per asset and T2 per group, both legs' risk counted (no hedge credit)
    try:
        for u in unders:
            stop_used, stop_caps, g = _stop_tests(book, u, profile)
            if "asset_open_risk" in stop_used:
                used[f"asset:{u}"], caps[f"asset:{u}"] = stop_used["asset_open_risk"], stop_caps["asset_open_risk"]
            if "group_open_risk" in stop_used:
                group_of[u] = g
                used[f"group:{g}"], caps[f"group:{g}"] = stop_used["group_open_risk"], stop_caps["group_open_risk"]
    except Unmapped as exc:
        why = (f"Entry rejected: {exc.args[0]} is in no asset group, so the per-group loss-at-stops test can't judge "
               "it; neither leg is sent")
        return rejected(first.qty, "group_open_risk", why), rejected(second.qty, "group_open_risk", why)
    except Exception as exc:  # noqa: BLE001 - fail closed (BRIDGE-3)
        why = f"Entry rejected: the loss-at-stops tests couldn't be checked ({type(exc).__name__}); neither leg is sent"
        return rejected(first.qty, "group_open_risk", why), rejected(second.qty, "group_open_risk", why)

    def added(q1: Decimal, q2: Decimal) -> dict:
        out = {"gross": q1 * first.price + q2 * second.price,
               "margin": q1 * first.margin_per_unit + q2 * second.margin_per_unit,
               "open_risk": q1 * first.risk_per_unit + q2 * second.risk_per_unit}
        for u in unders:
            out[f"net:{u}"] = sum((q * i.side * i.price for q, i in ((q1, first), (q2, second)) if i.underlying == u),
                                  ZERO)
            risk = sum((q * i.risk_per_unit for q, i in ((q1, first), (q2, second)) if i.underlying == u), ZERO)
            if f"asset:{u}" in used:
                out[f"asset:{u}"] = risk
            if u in group_of:
                out[f"group:{group_of[u]}"] = out.get(f"group:{group_of[u]}", ZERO) + risk
        return out

    whole = added(first.qty, second.qty)  # per unit of share: the share f adds f x whole to each figure
    room = {k: (_net_room(used[k], caps[k], whole[k]) if k.startswith("net:") else _room(used[k], caps[k], whole[k]))
            for k in used}
    bound = [k for k in used if room[k] is not None and room[k] < 1]
    limit = min(bound, key=lambda k: room[k]) if bound else None
    share = Decimal(1) if limit is None else room[limit]
    q1 = floor_to(first.qty * share, first.step)
    q2 = floor_to(q1 * ratio, second.step)

    def name(k: str) -> str:
        return {"net": "net_instrument", "asset": "asset_open_risk", "group": "group_open_risk"}.get(k.split(":")[0], k)

    def word(k: str) -> str:
        kind, _, what = k.partition(":")
        return WORDS[name(k)].format(u=what, g=what) if what else WORDS[k]

    def figures(a1: Decimal, a2: Decimal) -> dict:
        at = added(a1, a2)
        return {k: {"before": float(used[k] / e), "after": float((used[k] + at[k]) / e), "cap": float(caps[k] / e)}
                for k in used}

    def both(outcome, a1, a2, limit_hit, reason, figs):
        return (Decision(outcome, a1, first.qty, limit_hit, reason, figs),
                Decision(outcome, a2, second.qty, limit_hit, reason, figs))

    after = added(q1, q2)
    over = [k for k in used if (abs(used[k] + after[k]) > max(caps[k], abs(used[k])) if k.startswith("net:")
                                else after[k] > 0 and used[k] + after[k] > caps[k])]
    small = [n for n, q, i in (("first", q1, first), ("second", q2, second)) if q <= 0 or q < i.min_qty]
    if limit is None and not small and not over:
        return both("approved", q1, q2, None, "Both legs within every portfolio limit", figures(q1, q2))
    hit = name(limit) if limit is not None else ("below_min" if small else name(over[0]))
    if small or over:
        what = (f"the {' and '.join(small)} leg would be below its venue minimum" if small
                else f"{name(over[0])} would pass its limit after rounding")
        if limit is not None:
            what = f"{word(limit)} binds, " + what
        return both("rejected", ZERO, ZERO, hit, f"Entry rejected: {what}, so neither leg is sent",
                    figures(ZERO, ZERO))
    return both("trimmed", q1, q2, hit, f"Both legs trimmed to {share:.1%} of the request: {word(limit)} would pass its "
                                        f"limit with the whole pair", figures(q1, q2))


@dataclass(frozen=True)
class BookBreach:
    action: str  # "halt" | "pause"
    reason: str


def day_return(equity, day_start) -> Decimal | None:
    """Today's return (Advisor DAY-RET (B), 20 Oct): equity / day_start - 1, day_start the book at 00:00 UTC scaled by
    each flow since (gate.apply_flow, Advisor MUST 14 Oct): the time-weighted return, so a deposit or a withdrawal is
    never a gain or a loss, and a loss counts at its share of the book it was made on. The one reader: the daily pause
    judges it and the pages show it (FE never works it out again). None when either figure is missing or the day's start
    is 0 or below: nothing to measure from."""
    if equity is None or day_start is None:
        return None
    equity, day_start = money(equity, "equity"), money(day_start, "day_start")
    return None if day_start <= 0 else equity / day_start - 1


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
    # A brake switched off (None) never fires (B-32; locked today, B-31's locked half is the switch).
    if profile.drawdown is not None and equity <= top - scale(top, profile.drawdown):
        return BookBreach("halt", f"the book is {1 - equity / top:.1%} below its reference {top:,.2f}, at the "
                                  f"{profile.drawdown:.0%} portfolio drawdown limit")
    if profile.daily_loss is not None and equity <= day_start - scale(day_start, profile.daily_loss):
        return BookBreach("pause", f"the book has lost {-day_return(equity, day_start):.1%} since 00:00 UTC, at the "
                                   f"{profile.daily_loss:.0%} daily limit")
    return None


def validate_tests(profile: PortfolioProfile) -> None:
    """A PM's edited test set, checked before it is saved (portfolio.compliance.validate_tests)."""
    from sleeve_fund.portfolio.compliance import validate_tests as check

    check(profile)
