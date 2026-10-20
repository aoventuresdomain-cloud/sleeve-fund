"""B-13: netting across Strategies in shadow on paper (v2 P2-3; PM 13 Oct "shadow first"; Advisor 13 Oct 19:10 UK,
quant-review/specs-p2-3-p2-8-p2-5.md §2). What one account per venue would have traded and paid had each decision bar's
intents in an instrument gone out as one net order, read from the paper journal as it stands: the same pure core the
backtest shadow uses (portfolio.netting), over the fund's own fills.

Read only. It sends nothing, writes nothing and changes no order, fill, limit or Strategy figure (§2.1, §2.6): every
Strategy's sub-ledger is its own journal, and the figures here are never Strategy evidence (G1/G2).

- Accounts: one per venue, holding the fund's Strategies on it (a test Strategy outside the fund's book is its own fund
  and is left out, as the book leaves it out).
- Grouping: an order decided on a bar is grouped by its venue, instrument and decision bar's close (order_timings),
  so two Strategies deciding on one close net together. An order with no decision bar (a stop or target watched on a
  trade, the risk guard's) is its own group: nothing else was decided with it.
- Crossing prices: the decision mark is the order's signal close; the half spread is what that group's market fills
  paid against it, as the backtest shadow measures it. The taker fee is the venue's, as its fills were charged.
- Funding: each settlement charged (settled, or the baseline for a missing rate); a later true-up or reversal is a
  correction of a rate already charged and is counted, not re-netted (corrections).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select

from sleeve_fund.portfolio import netting as nt

ZERO = Decimal(0)


def _dec(x) -> Decimal:
    """A journal figure as an exact Decimal: as stored, or a float read through FloatView by the decimal it prints as."""
    return x if isinstance(x, Decimal) else Decimal(repr(float(x)))


def _ns(t: datetime) -> int:
    from sleeve_fund.store import _aware

    return int(_aware(t).timestamp() * 1_000_000) * 1_000


def journal_rows(store, members: Callable[[str], bool]) -> tuple[list[dict], list[dict]]:
    """The fills and funding charges of the Strategies `members` keeps (by name), from the paper journal: fills with
    their order's venue, intent, signal close and decision bar's close; funding rows as booked."""
    from sleeve_fund.store import (FUNDING_CHARGE_KINDS, _not_backtest, _rows, fills_t, funding_t, order_timings_t,
                                   orders_t)

    f, o, t = fills_t, orders_t, order_timings_t
    q = (select(f.c.sleeve, f.c.ts, f.c.side, f.c.qty, f.c.price, f.c.fee, f.c.instrument, f.c.order_id,
                o.c.venue, o.c.intent, o.c.signal, t.c.bar_close)
         .join(o, o.c.order_id == f.c.order_id)
         .outerjoin(t, t.c.order_id == f.c.order_id)
         .where(_not_backtest(f.c.sleeve))
         .order_by(f.c.ts, f.c.id))
    fq = select(funding_t).where(_not_backtest(funding_t.c.sleeve)).order_by(funding_t.c.ts, funding_t.c.id)
    with store.engine.connect() as c:
        fills = [r for r in _rows(c.execute(q)) if members(r["sleeve"])]
        funding = [r for r in _rows(c.execute(fq)) if members(r["sleeve"])]
    venue_of = {}
    for r in fills:
        venue_of.setdefault((r["sleeve"], r["instrument"]), r["venue"])
    for r in funding:
        r["venue"] = venue_of.get((r["sleeve"], r["instrument"]))
        r["charge"] = r["kind"] in FUNDING_CHARGE_KINDS
    return fills, funding


def shadow(fills: list[dict], funding: list[dict], taker: Callable[[str], Decimal], equity: Decimal) -> dict:
    """Step B in shadow per venue account over journal rows (journal_rows). taker(strategy): the taker fee that
    Strategy's fills are charged (its market's, from the venue's live terms), which must be one per account. equity: the fund's equity, the residual's yardstick (§2.5).
    Returns {venue: figures}, each as the backtest shadow reports it (research.portfolio_run.netting_shadow)."""
    out = {}
    venues = sorted({r["venue"] for r in fills if r["venue"]})
    for venue in venues:
        out[venue] = _account([r for r in fills if r["venue"] == venue],
                              [r for r in funding if r.get("venue") == venue], venue, taker, equity)
    return out


def _account(rows: list[dict], funding_rows: list[dict], venue: str, taker: Callable[[str], Decimal],
             equity: Decimal) -> dict:
    fills: list[nt.Fill] = []
    marks: dict[tuple[str, int], Decimal] = {}
    halves: dict[tuple[str, int], Decimal] = {}
    for r in rows:
        qty = _dec(r["qty"])
        if not qty:
            continue
        # A bar's decision groups by its close; anything else is its own, at its own fill's time (a microsecond apart
        # from any bar's close, which are whole minutes)
        ts = _ns(r["bar_close"]) if r["bar_close"] is not None else _ns(r["ts"]) + 1
        side = 1 if str(r["side"]).upper() == "BUY" else -1
        px = _dec(r["price"])
        fills.append(nt.Fill(r["sleeve"], r["instrument"], ts, side * qty, px, _dec(r["fee"])))
        close = (r["signal"] or {}).get("close") if r["bar_close"] is not None else None
        if close and (r["instrument"], ts) not in marks:
            marks[(r["instrument"], ts)] = _dec(close)
            halves[(r["instrument"], ts)] = abs(px / marks[(r["instrument"], ts)] - 1)
    first = {}
    for f in fills:
        first.setdefault((f.instrument, f.ts), f)

    def mark(inst: str, ts: int) -> Decimal:
        return marks.get((inst, ts), first[(inst, ts)].price)

    def half(inst: str, ts: int) -> Decimal:
        return halves.get((inst, ts), ZERO)

    rates = {taker(name) for name in {f.strategy for f in fills}}
    if len(rates) > 1:
        raise ValueError(f"netting shadow on {venue}: the Strategies on one account must pay the same taker fee")
    rate = next(iter(rates)) if rates else ZERO
    charged = [r for r in funding_rows if r["charge"]]
    funding = [nt.Funding(r["sleeve"], r["instrument"], _ns(r["ts"]), _dec(r["rate"]),
                          _dec(r["price"]), _dec(r["amount"]), r["kind"] == "baseline") for r in charged]
    sh = nt.shadow(fills, funding, mark, half, rate)
    saving, sub = sh.saving(), sh.strategy_cash()
    sub_total = sum(sub.values(), ZERO)
    residual = nt.reconcile(sub_total, sum(saving.values(), ZERO) + sh.baseline_total(), sh.account, equity)
    by_group: dict[tuple[str, int], set[str]] = defaultdict(set)
    for f in fills:
        by_group[(f.instrument, f.ts)].add(f.strategy)
    return {
        "saving": saving,
        "saving_total": sum(saving.values(), ZERO),
        "account": sh.account,
        "sub_ledger_total": sub_total,
        "sub_ledgers": sub,
        "conservative_funding_baseline": sh.baseline_total(),
        "funding": {"account": sh.account_funding, "sub_ledgers": sum((r.amount for r in funding), ZERO)},
        "corrections": sum(1 for r in funding_rows if not r["charge"]),
        "crossed_notional": sum((o.crossed * mark(o.instrument, o.ts) for o in sh.orders), ZERO),
        "orders_sent": sum(1 for o in sh.orders if o.qty),
        "orders_alone": len({r["order_id"] for r in rows if _dec(r["qty"])}),
        "bars_shared": sum(1 for names in by_group.values() if len(names) > 1),
        "residual": residual,
        "opposed_exposure": nt.opposed_exposure(fills, _last_price(fills), sorted({f.ts for f in fills})),
    }


def _last_price(fills: list[nt.Fill]) -> Callable[[str, int], Decimal]:
    """An instrument's last fill price at or before `ts` (its first before any): the mark opposed exposure weighs
    positions by at each group's time."""
    by: dict[str, list[nt.Fill]] = defaultdict(list)
    for f in sorted(fills, key=lambda f: f.ts):
        by[f.instrument].append(f)

    def at(inst: str, ts: int) -> Decimal:
        seen = [f.price for f in by[inst] if f.ts <= ts]
        return seen[-1] if seen else by[inst][0].price

    return at
