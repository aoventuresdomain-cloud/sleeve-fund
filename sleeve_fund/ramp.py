"""A9: the minimum-size ramp and its track-length gate (pre-live set; Independent Quant Advisor A9 ruling, 15 Oct 2026
09:13 UK). MUST for the Drop 4 build; the cost band (c) is MUST before the ramp is switched on.

A Strategy inside the fund trades at its MINIMUM SIZE, the venue's smallest order (the larger of its smallest quantity
and its smallest notional, rounded up to the lot), until it passes the gate, all three of:
(a) at least MIN_TRADES closed trades,
(b) its minimum track record length met (R5, research.mintrl.may_scale, 95%, fat tails counted) on its own daily track,
(c) its per-trade costs inside the band of the model (fills-vs-model, A8). There is no cost reader yet, so (c) counts as
    NOT passed: nobody leaves the ramp on (a) and (b) alone.
Until then a new entry takes one minimum order from flat and a held position takes no add, so for a Strategy that adds
the track describes its no-add version (Advisor FOLLOW-UP: before an add-heavy Strategy goes live, weight the track by
the notional the Model would have held, the skipped add included, virtually). It only ever shrinks a size:
exits, stops, take-profits and reductions are never capped, and a Strategy dropping back to the ramp is never cut.
Impact at minimum size understates impact at full size: fills-vs-model carries that caveat.

The track counts only fills in the fund's own mode (live fund, live fills: a Strategy's paper weeks never pass it on
arrival) in the current book: from the later of the latest re-base (supervisor.bridge_start) and the Strategy's
assignment to its current account. A day's return is each position's return on its own entry notional, weighted by its
UN-RAMPED notional (the size before the ramp cap, journaled with the entry) over the fund's equity at 00:00 UTC, marked
to market at the Strategy's own marks (the books' marks); a day with no trade counts as 0. So the same trades at ramp
size and at normal size give the same track. Only whole UTC days count, and the gate is re-evaluated once a day.

Switched OFF by default (SLEEVE_MIN_SIZE_RAMP=1, read once when a Strategy starts): sizing is then exactly as before.
Never in a backtest or research run. Anything unreadable fails closed: the Strategy stays at minimum size."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_UP, Decimal

import pandas as pd

from sleeve_fund.research import mintrl

ENV = "SLEEVE_MIN_SIZE_RAMP"
MIN_TRADES = 30
NOTIONAL_HEADROOM = Decimal("0.01")  # above the venue's smallest notional, for the move between decision and fill
LIMIT = "minimum-size ramp"  # the sizing limit's name in the journal's "sized by"
UNRAMPED = "unramped_notional"  # the entry's size before the ramp cap, journaled in its signal


def enabled() -> bool:
    """Off unless SLEEVE_MIN_SIZE_RAMP=1, the maker-orders switch's convention."""
    return os.environ.get(ENV, "") == "1"


@dataclass(frozen=True)
class Gate:
    passed: bool
    words: str  # why it stays at minimum size, or why it may scale, in the PM's words


def _signed(f: dict) -> Decimal:
    q = Decimal(str(f["qty"]))
    return q if str(f["side"]).upper() == "BUY" else -q


def positions(fills: list[dict]) -> list[tuple[datetime, datetime | None, str]]:
    """Each position in these fills, oldest first, from flat at the first fill: (opened, closed or None while open, the
    order that opened it). A flip closes one and opens the next at the same fill."""
    out, held, opened = [], Decimal(0), None
    for f in fills:
        after = held + _signed(f)
        if held != 0 and (after == 0 or (after > 0) != (held > 0)):
            out.append((opened[0], f["ts"], opened[1]))
            opened = None
        if after != 0 and opened is None:
            opened = (f["ts"], f["order_id"])
        held = after
    if opened is not None:
        out.append((opened[0], None, opened[1]))
    return out


def closed_trades(fills: list[dict]) -> int:
    """Trades closed: each time the position returns to flat, or flips through it. A partial close is not one."""
    return sum(1 for _, closed, _ in positions(fills) if closed is not None)


def weights(fills: list[dict], orders: dict[str, dict]) -> list[tuple[datetime, datetime | None, Decimal]]:
    """Each position's span and its weight k = un-ramped notional / the notional it entered with, from its opening
    order's signal; 1 when the entry wasn't capped by the ramp (it entered at its un-ramped size)."""
    first = {}
    for f in fills:
        first.setdefault(f["order_id"], f)
    out = []
    for opened, closed, oid in positions(fills):
        f, signal = first[oid], (orders.get(oid) or {}).get("signal") or {}
        entered = Decimal(str(f["qty"])) * Decimal(str(f["price"]))
        un = signal.get(UNRAMPED)
        out.append((opened, closed, Decimal(str(un)) / entered if un is not None and entered > 0 else Decimal(1)))
    return out


def daily_track(marks: list[dict], spans, fund_at, since: datetime, now: datetime) -> pd.Series:
    """The Strategy's daily returns at its un-ramped size, from the UTC day of `since` to the last whole day before
    `now`. marks: its own equity marks, oldest first (inside the fund its P&L, from 0; marked to market, fees and
    funding in). spans: weights(). Each move between two marks is the position's of that interval, times its k, booked
    on the day of the later mark; over the fund's equity at that day's 00:00 UTC mark (fund_at, at or before). A day
    with no move is 0. Raises when a day has no fund equity above zero: the gate then fails closed."""
    first = since.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    days = pd.date_range(first, today, freq="D", inclusive="left")
    pnl = {d: Decimal(0) for d in days}

    def k_at(a: datetime, b: datetime) -> Decimal:
        for opened, closed, k in spans:  # the position open in (a, b]: the one that closes in it, else the open one
            if opened <= b and (closed is None or closed > a):
                return k
        return Decimal(1)  # flat: nothing moves but funding or a late fee, booked as it is

    for a, b in zip(marks, marks[1:]):
        ta, tb = pd.Timestamp(a["ts"]).to_pydatetime(), pd.Timestamp(b["ts"]).to_pydatetime()
        day = pd.Timestamp(tb.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0))
        if tb < since or day not in pnl:
            continue
        move = Decimal(str(b["equity"])) - Decimal(str(a["equity"]))
        if move:
            pnl[day] += move * k_at(max(ta, since), tb)
    out = {}
    for day, p in pnl.items():
        base = fund_at(day.to_pydatetime())
        if base is None or Decimal(str(base)) <= 0:
            raise ValueError(f"no fund equity at {day:%Y-%m-%d} 00:00 UTC to measure the day against")
        out[day] = float(p / Decimal(str(base)))
    return pd.Series(out, dtype=float)


def cost_band(store, sleeve: str) -> Gate:
    """(c): per-trade costs inside the band of the model. No reader exists yet (fills-vs-model, A8, ships the report
    only), so it is NOT passed while the ramp is on (Advisor: MUST before the ramp is switched on)."""
    return Gate(False, "its per-trade cost band isn't measured yet, so it counts as not passed")


def gate(trades: int, track: pd.Series, cost: Gate, min_trades: int = MIN_TRADES) -> Gate:
    """All three must hold to leave the ramp: enough closed trades, a track at least its minimum track record, and its
    costs inside the band."""
    ok, why = mintrl.may_scale(track)
    passed = trades >= min_trades and ok and cost.passed
    head = "may scale" if passed else "stays at minimum size"
    track_words = why.split(": ", 1)[1] if ": " in why else why
    return Gate(passed, f"{head}: {trades} of the {min_trades} closed trades it needs; track (its no-add version: the "
                        f"ramp takes no adds): {track_words}; costs: {cost.words}")


def track_start(store, sleeve: str) -> datetime | None:
    """When its track may begin: the later of the latest re-base and its assignment to its current account (a move
    from paper to a live account starts it again, so only fills in that account's mode count)."""
    from sqlalchemy import select

    from sleeve_fund.store import _aware, sleeve_accounts_t
    from sleeve_fund.supervisor import bridge_start

    with store.engine.connect() as c:
        assigned = c.execute(select(sleeve_accounts_t.c.assigned_at)
                             .where(sleeve_accounts_t.c.sleeve == sleeve)).scalar()
    at = [t for t in (bridge_start(store), _aware(assigned) if assigned is not None else None) if t is not None]
    return max(at) if at else None


def read_gate(store, sleeve: str, now: datetime) -> Gate:
    """The gate for one Strategy, from the journal. Fails closed: anything unreadable keeps it at minimum size."""
    try:
        start = track_start(store, sleeve)
        fills = [f for f in store.fills(sleeve, limit=1_000_000)[::-1] if start is None or f["ts"] >= start]
        if not fills:
            return Gate(False, f"stays at minimum size: no trades yet in this book and mode (0 of the {MIN_TRADES} "
                               "closed trades it needs)")
        since = fills[0]["ts"]
        orders = {o["order_id"]: o for o in store.orders(sleeve, limit=1_000_000, intents=("entry",))}
        marks = store.equity_since(sleeve, since - timedelta(days=1))
        track = daily_track(marks, weights(fills, orders), store.book_equity_at_or_before, since, now)
        return gate(closed_trades(fills), track, cost_band(store, sleeve))
    except Exception as exc:  # noqa: BLE001 - fail closed: an unread track never scales a Strategy up
        return Gate(False, f"stays at minimum size: its track couldn't be read ({type(exc).__name__}: {exc})")


def minimum_qty(min_qty: Decimal, lot: Decimal, min_notional: Decimal = Decimal(0),
                price: Decimal = Decimal(0)) -> Decimal:
    """The smallest order the venue takes, rounded up to the lot: its smallest quantity, and its smallest notional at
    `price` with NOTIONAL_HEADROOM, so a small move between the decision and the fill can't take it below."""
    least = lot if min_qty <= 0 else (min_qty / lot).to_integral_value(rounding=ROUND_UP) * lot
    if min_notional > 0 and price > 0:
        by_value = min_notional * (1 + NOTIONAL_HEADROOM) / price
        least = max(least, (by_value / lot).to_integral_value(rounding=ROUND_UP) * lot)
    return least


def cap(least: Decimal, held: Decimal, side: int, price: Decimal) -> Decimal:
    """The notional a new entry or add may carry under the ramp: one order of the minimum size from flat, and no add
    while a position is held on that side, since any add would either take it past the minimum or be under the
    smallest order the venue takes. A position on the other side is closed first (that part is an exit), so the open
    part may carry the whole minimum. Never a reduction of what is held."""
    return least * price if held * side <= 0 else Decimal(0)
