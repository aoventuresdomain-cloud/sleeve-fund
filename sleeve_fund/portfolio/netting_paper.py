"""B-13: netting across Strategies in shadow on paper (v2 P2-3; PM 13 Oct "shadow first"; Advisor 13 Oct 19:10 UK,
quant-review/specs-p2-3-p2-8-p2-5.md §2, and HEAD6-B13-B14, 20 Oct). What one account would have traded and paid had
each decision bar's intents in an instrument gone out as one net order, read from the paper journal as it stands: the
same pure core the backtest shadow uses (portfolio.netting), over the fund's own fills.

Read only. It sends nothing, writes nothing and changes no order, fill, limit or Strategy figure (§2.1, §2.6): every
Strategy's sub-ledger is its own journal, and the figures here are never Strategy evidence (G1/G2). Its 5 bp residual
is reported and alerted, never blocking (HEAD6 4); none over the shadow period is a precondition for netting going live.

Every input is what the journal recorded at the time, never a value read today (HEAD6 1), so the same journal gives the
same figures on any later read:
- Accounts: one per venue and account (HEAD6 2), holding the fund's Strategies on it. A test Strategy outside the fund's
  book is its own fund and is left out, as the book leaves it out.
- Grouping: an order decided on a bar is grouped by its account, instrument and decision bar's close (order_timings),
  so two Strategies deciding on one close net together. An order with no decision bar (a stop or target watched on a
  trade, the risk guard's) is its own group: nothing else was decided with it (this understates the saving).
- Crossing prices: the decision mark is the order's journaled signal close; the half spread is what that group's
  market fills paid against it. The taker rate is the one those market fills were charged (fees / notional, to 1e-7,
  within their fees' rounding: _one_rate), so a fee change mid-period prices each bar at the rate then in force; fills
  in one group at two rates are refused.
- Funding: no saving line (netting saves none by construction, Advisor 13 Oct 20:25 UK). Settlement charges are
  netted on the account's position; a later true-up or reversal is linear in the position, so the account books the
  sum of the Strategies' corrections, and any difference falls to the residual. The conservative baseline line stays.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select

from sleeve_fund.portfolio import netting as nt

ZERO = Decimal(0)
SLACK = Decimal("1e-9")  # beyond one place of the fee's currency (_one_rate)
RATE_PLACES = Decimal("1e-7")  # a fee rate read back from a fill: 0.055% is 0.00055 to this, whatever the rounding


def _dec(x) -> Decimal:
    """A journal figure as an exact Decimal: as stored, or a float read through FloatView by the decimal it prints as."""
    return x if isinstance(x, Decimal) else Decimal(repr(float(x)))


def _ns(t: datetime) -> int:
    from sleeve_fund.store import _aware

    return int(_aware(t).timestamp() * 1_000_000) * 1_000


def journal_rows(store, members: Callable[[str], bool]) -> tuple[list[dict], list[dict]]:
    """The fills and funding rows of the Strategies `members` keeps (by name), from the paper journal: fills with their
    order's venue, type, intent, signal close, decision bar's close and the Strategy's account; funding as booked."""
    from sleeve_fund.store import (FUNDING_CHARGE_KINDS, _not_backtest, _rows, fills_t, funding_t, order_timings_t,
                                   orders_t)

    f, o, t = fills_t, orders_t, order_timings_t
    q = (select(f.c.sleeve, f.c.ts, f.c.side, f.c.qty, f.c.price, f.c.fee, f.c.instrument, f.c.order_id,
                o.c.venue, o.c.order_type, o.c.intent, o.c.signal, t.c.bar_close)
         .join(o, o.c.order_id == f.c.order_id)
         .outerjoin(t, t.c.order_id == f.c.order_id)
         .where(_not_backtest(f.c.sleeve))
         .order_by(f.c.ts, f.c.id))
    fq = select(funding_t).where(_not_backtest(funding_t.c.sleeve)).order_by(funding_t.c.ts, funding_t.c.id)
    with store.engine.connect() as c:
        fills = [r for r in _rows(c.execute(q)) if members(r["sleeve"])]
        funding = [r for r in _rows(c.execute(fq)) if members(r["sleeve"])]
    accounts = {n: store.account_of(n) for n in {r["sleeve"] for r in fills + funding}}
    venue_of = {}
    for r in fills:
        r["account"] = (r["venue"], accounts[r["sleeve"]])
        venue_of.setdefault((r["sleeve"], r["instrument"]), r["venue"])
    for r in funding:
        venue = venue_of.get((r["sleeve"], r["instrument"]))
        r["account"] = (venue, accounts[r["sleeve"]]) if venue else None
        r["charge"] = r["kind"] in FUNDING_CHARGE_KINDS
    return fills, funding


def shadow(fills: list[dict], funding: list[dict], equity: Decimal) -> dict:
    """Step B in shadow per account over journal rows (journal_rows). equity: the fund's equity, the residual's
    yardstick (§2.5). Returns {(venue, account): figures}, each as the backtest shadow reports it
    (research.portfolio_run.netting_shadow)."""
    out = {}
    for key in sorted({r["account"] for r in fills}):
        out[key] = _account([r for r in fills if r["account"] == key],
                            [r for r in funding if r["account"] == key], key, equity)
    return out


def _charged(r: dict) -> tuple[Decimal, Decimal] | None:
    """(fee, notional) of a market fill as journaled, the taker rate it was charged read from the two; None for anything
    else (a resting order's maker fee is not what a net market order would pay)."""
    notional = _dec(r["qty"]) * _dec(r["price"])
    if str(r["order_type"]).upper() != "MARKET" or not notional:
        return None
    return _dec(r["fee"]), notional


def fee_places(inst: str) -> int:
    """The decimal places paper books a fill's fee to in `inst` ("BASE/QUOTE"): its quote currency's, the currency the
    paper account settles in (instruments: settlement_currency=quote). Refused in words when the currency is unknown."""
    from nautilus_trader.model import Currency

    try:
        return Currency.from_str(inst.split("/")[1]).precision
    except (IndexError, ValueError, KeyError) as e:
        raise ValueError(f"netting shadow: can't tell the places a fee in {inst} is booked to") from e


def _one_rate(inst: str, charged: list[tuple[Decimal, Decimal]]) -> Decimal:
    """The taker rate one candle's market fills paid in all (fees over notional, to RATE_PLACES; Advisor: pooled, as
    journaled). Paper rounds each fill's fee to its currency's places (fee_places: the cent in USD), so slices of one
    order read back a little apart one by one. They are one rate when a single rate explains every fee to within one
    place (plus 1e-9): the rates each fee allows overlap. Otherwise it is a second rate, refused. Not half a place (the
    rounding bound): paper's journaled fees sit up to 0.008 from their fill's price x qty x rate (1.81 for 1.818), so half
    a place refuses one order's own slices (b13y). Shortcut: on a fill too small for one place to tell two rates apart
    (under about 30 of notional in USD at maker vs taker), a second rate can pass as the first; immaterial to a shadow.
    Netting go-live MUST (Advisor): journal each fill's charged rate and use it, never read it back."""
    place = Decimal(1).scaleb(-fee_places(inst)) + SLACK
    low = max((f - place) / n for f, n in charged)
    high = min((f + place) / n for f, n in charged)
    if low > high:
        seen = sorted({(f / n).quantize(RATE_PLACES) for f, n in charged})
        raise ValueError(f"netting shadow: fills of one candle in {inst} were charged different taker rates "
                         f"({', '.join(f'{x:%}' for x in seen)}); the Strategies on one account must pay the same "
                         "taker fee")
    return (sum(f for f, _ in charged) / sum(n for _, n in charged)).quantize(RATE_PLACES)


def _account(rows: list[dict], funding_rows: list[dict], key, equity: Decimal) -> dict:
    fills: list[nt.Fill] = []
    marks: dict[tuple[str, int], Decimal] = {}
    halves: dict[tuple[str, int], Decimal] = {}
    paid_by: dict[tuple[str, int], list[tuple[Decimal, Decimal]]] = defaultdict(list)  # candle: its market fills
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
        if (paid := _charged(r)) is not None:
            paid_by[(r["instrument"], ts)].append(paid)
        close = (r["signal"] or {}).get("close") if r["bar_close"] is not None else None
        if close and (r["instrument"], ts) not in marks:
            marks[(r["instrument"], ts)] = _dec(close)
            halves[(r["instrument"], ts)] = abs(px / marks[(r["instrument"], ts)] - 1)
    first = {}
    for f in fills:
        first.setdefault((f.instrument, f.ts), f)
    rates = {(inst, ts): _one_rate(inst, paid) for (inst, ts), paid in paid_by.items()}

    def mark(inst: str, ts: int) -> Decimal:
        return marks.get((inst, ts), first[(inst, ts)].price)

    def half(inst: str, ts: int) -> Decimal:
        return halves.get((inst, ts), ZERO)

    def taker(inst: str, ts: int) -> Decimal:
        if (rate := rates.get((inst, ts))) is None:
            raise ValueError(f"netting shadow: no market fill in {inst} at a crossing candle to read the taker rate from")
        return rate

    charged = [r for r in funding_rows if r["charge"]]
    funding = [nt.Funding(r["sleeve"], r["instrument"], _ns(r["ts"]), _dec(r["rate"]), _dec(r["price"]),
                          _dec(r["amount"]), r["kind"] == "baseline") for r in charged]
    corrections = [r for r in funding_rows if not r["charge"]]
    corrected = sum((_dec(r["amount"]) for r in corrections), ZERO)
    sh = nt.shadow(fills, funding, mark, half, taker)
    saving, sub = sh.saving(), sh.strategy_cash()
    for r in corrections:  # each Strategy's own correction, booked to it as its journal did
        sub[r["sleeve"]] = sub.get(r["sleeve"], ZERO) + _dec(r["amount"])
    sub_total = sum(sub.values(), ZERO)
    account = sh.account + corrected  # linear in the position: the account's correction is the sum of theirs
    residual = nt.reconcile(sub_total, sum(saving.values(), ZERO) + sh.baseline_total(), account, equity)
    by_group: dict[tuple[str, int], set[str]] = defaultdict(set)
    for f in fills:
        by_group[(f.instrument, f.ts)].add(f.strategy)
    sub_funding = sum((r.amount for r in funding), ZERO) + corrected
    return {
        "venue": key[0], "account": key[1],
        "saving": saving,
        "saving_total": sum(saving.values(), ZERO),
        "account_cash": account,
        "sub_ledger_total": sub_total,
        "sub_ledgers": sub,
        "conservative_funding_baseline": sh.baseline_total(),
        # Reconciled, not saved: the account's funding against the Strategies', the baseline line apart
        "funding": {"account": sh.account_funding + corrected, "sub_ledgers": sub_funding,
                    "gap": sh.account_funding + corrected + sh.baseline_total() - sub_funding},
        "corrections": len(corrections),
        "crossed_notional": sum((o.crossed * mark(o.instrument, o.ts) for o in sh.orders), ZERO),
        "orders_sent": sum(1 for o in sh.orders if o.qty),
        "orders_alone": len({r["order_id"] for r in rows if _dec(r["qty"])}),
        "bars_shared": sum(1 for names in by_group.values() if len(names) > 1),
        "residual": residual,
        # Advisor (B-13 fee fix): each crossing candle's taker rate as read (pooled), with the fill slices it came from
        "taker_rates": [{"instrument": inst, "ts": ts, "rate": rates[(inst, ts)], "slices": len(paid)}
                        for (inst, ts), paid in sorted(paid_by.items())],
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


def over_line(store, members: Callable[[str], bool], equity: Decimal) -> list[str]:
    """The accounts whose residual is over the 5 bp line, in words, for the nightly alert (HEAD6 4: reported and
    alerted, never blocking). Names no venue: an account is named by its Strategies."""
    fills, funding = journal_rows(store, members)
    out = []
    for s in shadow(fills, funding, equity).values():
        if s["residual"]["block"]:
            out.append(f"netting shadow (trading unaffected): the account of {', '.join(sorted(s['sub_ledgers']))} has "
                       f"an unexplained difference of {s['residual']['amount']:,.2f} ({s['residual']['bp']:.1f} bp), "
                       f"over the {nt.RESIDUAL_BP} bp line")
    return out
