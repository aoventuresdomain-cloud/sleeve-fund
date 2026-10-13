"""Netting across Strategies on one account (v2 P2-3, step B), shadow first, as pure functions: what one account per
venue would have traded and paid had each bar's opening and closing intents in an instrument gone out as one net order,
beside step A, which is unchanged (Independent Quant Advisor 13 Oct 19:10 UK, quant-review/specs-p2-3-p2-8-p2-5.md §2).

Netting is execution only (§2.1): it never changes a Strategy's decisions, its evidence or any limit. So each Strategy's
sub-ledger is its own step A fills, fees and funding, booked as it traded alone (§2.3), and whatever one net order
saves goes to book-level lines, never to a Strategy:
- crossed: the price the crossing intents didn't pay (the spread both sides of a cross would have crossed alone);
- fee: the fees on volume that crossed inside the account instead of going to the venue;
- residual: what is left when the sub-ledgers and those lines don't sum to the account (§2.5). Above RESIDUAL_BP of the
  account's equity it is alerted and blocks new entries on that venue (reconcile()).

Funding has no saving line (Advisor 13 Oct 20:25 UK): it is linear in the position, so the net position pays what the
virtual ones do but for each account's rounding (a baseline settlement for a missing rate, which both sides pay, is
not linear: there the account pays less than its Strategies when they are opposed, and that shows in the residual). The account pays the net position's funding, the sub-ledgers carry
their own, and any difference is part of the residual, under the same 5 bp rule. Netting saves fees, spread and margin;
the margin is shown as information only, never as P&L.

A bar whose intents all go one way crosses nothing: the account sends what step A sent, at step A's own prices and fees,
and nothing is saved. Where intents cross, the net order is priced with step A's own fill model for a market order
(ScheduleFeeModel: the decision mark moved by the half spread against the net side, the venue's taker fee). A bar whose
intents fully cross sends nothing.

Money and quantities are exact Decimals (sleeve_fund.money): a float is refused, as in portfolio.limits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations

from sleeve_fund.money import money, scale

RESIDUAL_BP = Decimal(5)  # §2.5: a residual above this many basis points of the account's equity alerts and blocks
ZERO = Decimal(0)


@dataclass(frozen=True)
class Fill:
    """One step A fill: a Strategy's own trade in an instrument at a decision time `ts` (the bar it was decided on), as
    it filled trading alone. qty is signed (a buy is positive)."""

    strategy: str
    instrument: str
    ts: int
    qty: Decimal
    price: Decimal
    fee: Decimal

    def __post_init__(self) -> None:
        for name in ("qty", "price", "fee"):
            money(getattr(self, name), name)

    @property
    def cash(self) -> Decimal:
        """What the fill does to the Strategy's cash: a buy pays qty x price, a sell receives it, the fee always paid."""
        return -self.qty * self.price - self.fee


@dataclass(frozen=True)
class Funding:
    """One step A funding line: what a Strategy's virtual position in `instrument` paid (negative) or was credited at
    the settlement `ts`, at `rate` on `mark`. both_pay: a settlement charged at the baseline for a missing rate, which
    every side pays and none receives (Advisor 6 Oct, D9): the account then pays it on its net position's size."""

    strategy: str
    instrument: str
    ts: int
    rate: Decimal
    mark: Decimal
    amount: Decimal
    both_pay: bool = False

    def __post_init__(self) -> None:
        for name in ("rate", "mark", "amount"):
            money(getattr(self, name), name)


@dataclass(frozen=True)
class NetOrder:
    """What one account would have sent for an instrument at `ts`: `qty` signed (zero when the intents fully cross,
    and then nothing goes to the venue), the volume matched inside the account (`crossed`, one side of it), and its fill
    at `price` (the decision mark when nothing is sent) for `fee`."""

    instrument: str
    ts: int
    qty: Decimal
    crossed: Decimal
    price: Decimal
    fee: Decimal
    strategies: tuple[str, ...]

    @property
    def cash(self) -> Decimal:
        return -self.qty * self.price - self.fee


@dataclass(frozen=True)
class Shadow:
    """Step B beside step A: the net orders, each Strategy's sub-ledger rows (its step A lines, kind fill, fee or
    funding) and the book-level saving lines (kind crossed or fee), all as cash amounts, plus the account's cash move
    those orders and the net position's funding make (account_funding, its funding part)."""

    orders: tuple[NetOrder, ...]
    sub_ledger: tuple[dict, ...]
    book: tuple[dict, ...]
    account: Decimal
    account_funding: Decimal = ZERO

    def strategy_cash(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = defaultdict(lambda: ZERO)
        for row in self.sub_ledger:
            out[row["strategy"]] += row["amount"]
        return dict(out)

    def saving(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = defaultdict(lambda: ZERO)
        for row in self.book:
            out[row["kind"]] += row["amount"]
        return dict(out)


def net_orders(fills: Iterable[Fill], mark: Callable[[str, int], Decimal], half_spread: Callable[[str, int], object],
               taker: Decimal) -> tuple[NetOrder, ...]:
    """The bar's intents per instrument as one net order each (§2.2), in time then instrument order. mark(instrument,
    ts): the decision mark step A's market orders fill from; half_spread(instrument, ts): the half spread its fill model
    charged a market order there. Only a bar whose intents cross is repriced; one whose intents all go one way is what
    step A sent (its volume-weighted price, its fees: a stop's or a target's own booking stays as it was)."""
    taker = money(taker, "taker")
    groups: dict[tuple[int, str], list[Fill]] = defaultdict(list)
    for f in fills:
        groups[(f.ts, f.instrument)].append(f)
    out = []
    for (ts, instrument), group in sorted(groups.items()):
        net = sum((f.qty for f in group), ZERO)
        crossed = (sum((abs(f.qty) for f in group), ZERO) - abs(net)) / 2
        if not crossed:
            price, fee = sum((f.qty * f.price for f in group), ZERO) / net, sum((f.fee for f in group), ZERO)
        elif net:
            m, slip = money(mark(instrument, ts), "mark"), _ratio(half_spread(instrument, ts))
            price = m * (1 + slip) if net > 0 else m * (1 - slip)
            fee = abs(net) * price * taker
        else:
            price, fee = money(mark(instrument, ts), "mark"), ZERO
        out.append(NetOrder(instrument, ts, net, crossed, price, fee,
                            tuple(sorted({f.strategy for f in group}))))
    return tuple(out)


def _ratio(r) -> Decimal:
    """A half spread as a Decimal: as given when exact, else through money.scale (the one float-to-Decimal door)."""
    return r if isinstance(r, Decimal) else scale(1, r)


def shadow(fills: Iterable[Fill], funding: Iterable[Funding], mark: Callable[[str, int], Decimal],
           half_spread: Callable[[str, int], object], taker: Decimal,
           funding_quantum: Decimal = Decimal("0.00000001")) -> Shadow:
    """Step B in shadow over a step A run's fills and funding (§2.3, §2.6). Each Strategy's sub-ledger is its own step A
    lines; the account trades the net orders and pays funding on the net position (rate x position x mark, rounded to
    `funding_quantum` as the venue books it). What the net orders save is the book's lines; what the net position's
    funding differs from the sub-ledgers' is left to reconcile()'s residual."""
    fills, funding = list(fills), list(funding)
    orders = net_orders(fills, mark, half_spread, taker)
    sub: list[dict] = []
    for f in fills:
        sub.append({"strategy": f.strategy, "instrument": f.instrument, "ts": f.ts, "kind": "fill",
                    "qty": f.qty, "price": f.price, "amount": -f.qty * f.price})
        sub.append({"strategy": f.strategy, "instrument": f.instrument, "ts": f.ts, "kind": "fee", "amount": -f.fee})
    for r in funding:
        sub.append({"strategy": r.strategy, "instrument": r.instrument, "ts": r.ts, "kind": "funding",
                    "amount": r.amount})
    book: list[dict] = []
    by_bar: dict[tuple[int, str], list[Fill]] = defaultdict(list)
    for f in fills:
        by_bar[(f.ts, f.instrument)].append(f)
    for o in orders:
        alone = by_bar[(o.ts, o.instrument)]
        price_alone = sum((-f.qty * f.price for f in alone), ZERO)
        fee_alone = sum((f.fee for f in alone), ZERO)
        if o.crossed or price_alone != -o.qty * o.price:
            book.append({"instrument": o.instrument, "ts": o.ts, "kind": "crossed",
                         "amount": -o.qty * o.price - price_alone})
        if fee_alone != o.fee:
            book.append({"instrument": o.instrument, "ts": o.ts, "kind": "fee", "amount": fee_alone - o.fee})
    net_funding = ZERO
    settlements: dict[tuple[int, str], list[Funding]] = defaultdict(list)
    for r in funding:
        settlements[(r.ts, r.instrument)].append(r)
    positions = _positions(fills)
    for (ts, instrument), rows in sorted(settlements.items()):
        rate, m, pos = rows[0].rate, rows[0].mark, positions(instrument, ts)
        owed = -abs(rate * pos * m) if rows[0].both_pay else -rate * pos * m
        net_funding += owed.quantize(funding_quantum)
    account = sum((o.cash for o in orders), ZERO) + net_funding
    return Shadow(orders, tuple(sub), tuple(book), account, net_funding)


def _positions(fills: list[Fill]) -> Callable[[str, int], Decimal]:
    """The account's net position in an instrument just before `ts` (a settlement sees the fills decided before it)."""
    by_inst: dict[str, list[Fill]] = defaultdict(list)
    for f in sorted(fills, key=lambda f: f.ts):
        by_inst[f.instrument].append(f)

    def at(instrument: str, ts: int) -> Decimal:
        return sum((f.qty for f in by_inst[instrument] if f.ts < ts), ZERO)

    return at


def reconcile(sub_ledger_total: Decimal, book_total: Decimal, account: Decimal, equity: Decimal) -> dict:
    """§2.5: the sub-ledgers plus the book's lines against the account's own cash move. The difference is the residual,
    booked at book level (kind residual), never to a Strategy; above RESIDUAL_BP of `equity` it alerts and blocks new
    entries on that venue until resolved."""
    for name, v in (("sub_ledger_total", sub_ledger_total), ("book_total", book_total), ("account", account),
                    ("equity", equity)):
        money(v, name)
    residual = account - sub_ledger_total - book_total
    bp = abs(residual) / equity * 10_000 if equity > 0 else (Decimal("Infinity") if residual else ZERO)
    return {"kind": "residual", "amount": residual, "bp": bp, "block": bp > RESIDUAL_BP}


def opposed_exposure(fills: Iterable[Fill], marks: Callable[[str, int], Decimal],
                     times: Iterable[int]) -> dict[tuple[str, str], float]:
    """§2.2: per Strategy pair, the share of their combined gross notional that cancels out, over the sample `times`
    (positions just after each), across instruments: 2 x sum(min(|a|, |b|) where they are opposite) / sum(|a| + |b|).
    1.0: they always fully cancel; 0.0: never opposed. A PM allocation question, shown on the portfolio page."""
    fills = sorted(fills, key=lambda f: f.ts)
    names = sorted({f.strategy for f in fills})
    instruments = sorted({f.instrument for f in fills})
    cancel: dict[tuple[str, str], Decimal] = defaultdict(lambda: ZERO)
    gross: dict[tuple[str, str], Decimal] = defaultdict(lambda: ZERO)
    for t in sorted(times):
        for inst in instruments:
            m = money(marks(inst, t), "mark")
            pos = {n: sum((f.qty for f in fills if f.strategy == n and f.instrument == inst and f.ts <= t), ZERO)
                   for n in names}
            for a, b in combinations(names, 2):
                gross[(a, b)] += (abs(pos[a]) + abs(pos[b])) * m
                if pos[a] * pos[b] < 0:
                    cancel[(a, b)] += min(abs(pos[a]), abs(pos[b])) * m
    return {pair: float(2 * cancel[pair] / gross[pair]) if gross[pair] else 0.0 for pair in combinations(names, 2)}
