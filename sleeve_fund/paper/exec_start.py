"""The order client's start and the fills it can't leave to the Strategy (B-05 M5; drill-prep MUST 5, 11, 19, 20, 21).
Built OFF: nothing calls it while exec_client.assert_off refuses ON at the node. Cells run it on recorded responses
(tests/exec_stub.py); nothing here reaches the exchange, the demo or the server.

- start_reconcile: at a start, every open order on the account is sorted by whose it is (venue_stops.classify). One
  of ours that no journal owns is cancelled by its id; one outside the fund's namespace is foreign: CRITICAL once
  fund-wide and new entries on its instrument held fund-wide, never cancelled or booked (MUST 20). The rest, with the
  venue's position, is set against the journal per Strategy (exec_reconcile, PE2): a gap holds entries.
- protect_late_fill: a late fill gets a venue-resting reduce-only stop at once, from its owner, so it survives the
  owner dying (MUST 11); it is read back, a failed read retried once after a back-off, then CRITICAL (MUST 19).
- book_settlements: funding from the venue's settlement records, once per settlement, split to the Strategies on the
  account by their journal quantity with the cents adding up exactly to the venue's; a split that can't be made
  from the journal is a gap, never a guess (MUST 5).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

from sleeve_fund.paper import exec_reconcile
from sleeve_fund.paper.exec_client import order_id, stop_drift
from sleeve_fund.paper.venue_stops import OwnershipGate, classify

FOREIGN_HOLD, FOREIGN_CLEARED = "foreign_order_hold", "foreign_order_cleared"  # fund-wide events (sleeve None)
STOP_UNVERIFIED = "stop_unverified"
FUNDING_STEP = Decimal("0.00000001")  # the journal's exact-amount step


# start ----------------------------------------------------------------------------------------------------------------
def journal_view(store, tags) -> tuple[dict[str, Decimal], dict[str, Decimal], set[str]]:
    """Each tag's net from its journal, its open orders by id to what's left, and every order id it journaled."""
    from sleeve_fund.store import OPEN_ORDER_STATUSES

    net, open_, ids = {}, {}, set()
    for tag in tags:
        net[tag] = Decimal(str(store.journal_book(tag, 0)["qty"]))
        for o in store.orders(tag, limit=100_000):
            ids.add(o["order_id"])
            if o["status"] in OPEN_ORDER_STATUSES:
                open_[o["order_id"]] = Decimal(str(o["qty"])) - Decimal(str(o["filled_qty"] or 0))
    return net, open_, ids


def start_reconcile(store, reads, stops, *, account: str, symbol: str, instrument: str, tags, fund_ids: set,
                    now: datetime) -> dict:
    """The start, on what the venue answers now. `tags`: the Strategies trading `instrument` on this account;
    `fund_ids`: every order id any Strategy of the fund journaled. Returns what it did, for the cells and the log."""
    tags = tuple(tags)
    net, open_, own = journal_view(store, tags)
    positions, orders = reads.positions(symbol), reads.open_orders(symbol)
    if positions is None or orders is None:
        result = exec_reconcile.Result(instrument, tags, ("the venue didn't answer for its position and open orders",))
        exec_reconcile.record(store, account, instrument, result, now)
        return {"result": result, "cancelled": [], "foreign": []}
    mine, cancelled, foreign = [], [], []
    for row in orders:
        link = (row.get("orderLinkId") or "").strip()
        kind = "foreign" if not link else classify(link, own=own, fund=fund_ids | own)
        if kind == "foreign":
            foreign.append(link or str(row.get("orderId")))
        elif kind == "orphan":  # ours, in no journal: cancelled by its id (orphan_guard og3); the venue's answer
            cancelled.append((link, stops.cancel(link)))  # is read back by the next start, never assumed
        elif kind == "book":
            mine.append(row)
    for oid in foreign:
        hold_foreign(store, oid, instrument, now)
    result = exec_reconcile.reconcile(instrument, net, open_, positions, mine)
    exec_reconcile.record(store, account, instrument, result, now)
    if not foreign and result.clean:
        clear_foreign(store, instrument, now)
    return {"result": result, "cancelled": cancelled, "foreign": foreign}


def hold_foreign(store, oid: str, instrument: str, now: datetime) -> None:
    """A foreign order: CRITICAL (an error alert) once fund-wide per order (the journal dedupes it across processes), and new entries
    on its instrument held fund-wide."""
    seen = store.sleeve_events_since(None, (FOREIGN_HOLD,))
    if any(f"[{oid}]" in e["message"] for e in seen):
        return
    store.event(None, "error", FOREIGN_HOLD,
                f"CRITICAL: order [{oid}] on {instrument} is on the account and in no journal of the fund. New entries on "
                f"{{{instrument}}} are held for every Strategy until it is gone and a start check ties; the fund never "
                "cancels or books it. Exits and stops still go out", ts=now)


def clear_foreign(store, instrument: str, now: datetime) -> None:
    if foreign_hold(store, instrument) is not None:
        store.event(None, "info", FOREIGN_CLEARED, f"No foreign order on {{{instrument}}} and the start check ties: "
                    "entries are allowed again", ts=now)


def foreign_hold(store, instrument: str) -> str | None:
    """Why nothing may open on `instrument` for a foreign order, or None. Unreadable: held (fail closed)."""
    tag = f"{{{instrument}}}"
    try:
        events = store.sleeve_events_since(None, (FOREIGN_HOLD, FOREIGN_CLEARED))
    except Exception:  # noqa: BLE001 - fail closed, never an error out of the entry gate
        return "whether a foreign order is on the account couldn't be read from the journal."
    last = next((e for e in reversed(events) if tag in e["message"]), None)
    return last["message"] if last is not None and last["kind"] == FOREIGN_HOLD else None


# orders the client hands a Strategy (MUST 21) ---------------------------------------------------------------------
def journal_owner(store, oid: str) -> str | None:
    """The Strategy whose journal holds the order, or None."""
    from sqlalchemy import select

    from sleeve_fund.store import orders_t

    with store.engine.connect() as c:
        return c.execute(select(orders_t.c.sleeve).where(orders_t.c.order_id == oid)).scalar()


class Journaled:
    """The journal's order ids as a set for venue_stops.classify: one Strategy's (name) or the fund's (None), read
    when asked, so a row another process just wrote is seen."""

    def __init__(self, store, name: str | None = None) -> None:
        self.store, self.name = store, name

    def __contains__(self, oid) -> bool:
        owner = journal_owner(self.store, str(oid))
        return owner is not None and (self.name is None or owner == self.name)


def gate(store, name: str) -> OwnershipGate:
    return OwnershipGate(Journaled(store, name))


def claim(ownership: OwnershipGate, store, name: str, oid: str, now: datetime) -> str:
    """An order event the client hands `name` for an order it didn't send in this process: "book" (its journal's),
    "ignore" (a sibling's), "hold" (ours, row not visible yet: re-checked by recheck) or "foreign" (held fund-wide on
    the Strategy's instrument, CRITICAL once). Only "book" is booked."""
    verdict = ownership.on_claimed(oid, Journaled(store), now)
    if verdict == "foreign":
        hold_foreign(store, oid, store.sleeve(name).instrument, now)
    return verdict


def recheck(ownership: OwnershipGate, store, name: str, now: datetime) -> list[str]:
    """The held orders past their grace: one still in no journal is an orphan, returned for its owner to cancel by
    id. One that turned out to be this Strategy's had events skipped while held, so its book is not trusted until a
    start check ties: entries held (exec_reconcile), never booked by guess."""
    orphans = []
    for oid, verdict in ownership.recheck(Journaled(store), now):
        if verdict == "orphan":
            orphans.append(oid)
        elif verdict == "book":
            sleeve = store.sleeve(name)
            exec_reconcile.record(store, "this account", sleeve.instrument, exec_reconcile.Result(
                sleeve.instrument, (name,), (f"order {oid} was handed over before its journal row was visible",)), now)
    return orphans


# a late fill's resting stop -------------------------------------------------------------------------------------------
def protect_late_fill(store, name: str, stops, *, side: str, qty, level, now: datetime, backoff_s: float = 1.0,
                      sleep=time.sleep) -> dict:
    """MUST 11: the owner places a reduce-only stop resting at the venue for the late fill's quantity at once (a long's
    fill a SELL stop, a short's a BUY), then reads it back (MUST 19). Returns the stop's id, the venue's answer and
    what the read-back found."""
    oid = order_id()
    answer = stops.place(oid, side=side, qty=qty, level=level)
    if answer[0] == "rejected":
        store.event(name, "error", "cannot_protect", f"CRITICAL: cannot protect: {name}: the venue refused the resting stop for a late fill "
                    f"({side} {qty} at {level}: {answer[1]} {answer[2]}). The position is unprotected at the venue: "
                    "close it or place a stop now", ts=now)
        return {"id": oid, "answer": answer, "read": None}
    row = read_back(store, name, stops, oid, now=now, backoff_s=backoff_s, sleep=sleep)
    if row is not False:
        drift = stop_drift(row, side=side, qty=qty, level=level)
        if drift:
            store.event(name, "error", STOP_UNVERIFIED, f"CRITICAL: {name}: the late fill's resting stop {oid} is not as sent: "
                        + "; ".join(drift), ts=now)
    return {"id": oid, "answer": answer, "read": row}


def read_back(store, name: str, stops, oid: str, *, now: datetime, backoff_s: float = 1.0, sleep=time.sleep):
    """The venue's row for a stop just sent (None: it reports none). No answer is retried once after the back-off;
    no answer again is CRITICAL "stop unverified on the venue" and False (MUST 19, OWN-STOP-SPEC Q2)."""
    for attempt in (1, 2):
        try:
            return stops.lookup(oid)
        except ConnectionError:
            if attempt == 1:
                sleep(backoff_s)
    store.event(name, "error", STOP_UNVERIFIED, f"CRITICAL: {name}: stop unverified on the venue: {oid} could not be read "
                "back twice. It may or may not be resting; check the account now", ts=now)
    return False


# funding from settlement records --------------------------------------------------------------------------------------
def _dec(raw, what: str) -> Decimal:
    try:
        d = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        raise exec_reconcile.Unreadable(f"settlement {what} {raw!r} isn't a number") from None
    if not d.is_finite():
        raise exec_reconcile.Unreadable(f"settlement {what} {raw!r} isn't a number")
    return d


def settlement(row: dict) -> tuple[str, datetime, Decimal, Decimal, Decimal, Decimal]:
    """One SETTLEMENT row of the account's transaction log: (id, minute of settlement, signed size, mark price,
    funding rate, cash change). The cash change is what the account moved (received positive); the venue's own
    funding field is its negative, which is checked. Field names and signs are as recorded, to be confirmed with
    the docs check (MUST 15/17): any mismatch is Unreadable, never booked."""
    sid = str(row.get("id") or "").strip()
    if not sid:
        raise exec_reconcile.Unreadable("a settlement row has no id")
    side = row.get("side")
    if side not in ("Buy", "Sell"):
        raise exec_reconcile.Unreadable(f"settlement {sid} side {side!r}")
    size = _dec(row.get("size"), "size") * (1 if side == "Buy" else -1)
    change = _dec(row.get("change"), "cash change")
    if _dec(row.get("funding"), "funding") != -change:
        raise exec_reconcile.Unreadable(f"settlement {sid}: funding {row.get('funding')} isn't the cash change's negative")
    ms = int(_dec(row.get("transactionTime"), "time"))
    at = datetime.fromtimestamp(ms / 1000, UTC).replace(second=0, microsecond=0)
    return sid, at, size, _dec(row.get("tradePrice"), "price"), _dec(row.get("feeRate"), "rate"), change


def split(amount: Decimal, size: Decimal, held: dict[str, Decimal]) -> dict[str, Decimal]:
    """The venue's amount over the Strategies holding the position, by quantity, to the journal's step; the largest
    holder takes the rounding so the parts add up to the venue's amount exactly. Holders must add up to `size`."""
    parts = {t: (amount * q / size).quantize(FUNDING_STEP, ROUND_HALF_EVEN) for t, q in held.items()}
    big = max(held, key=lambda t: (abs(held[t]), t))
    parts[big] += amount - sum(parts.values(), Decimal(0))
    return parts


def book_settlements(store, rows: list[dict], held_at, *, account: str, instrument: str, tags, now: datetime) -> dict:
    """Books each settlement once: `held_at(at)` gives each tag's journal quantity at that settlement (signed). A row
    that can't be read, or holders that don't add up to the venue's size, is a gap: nothing booked for it, entries held
    on every tag (exec_reconcile), never split by guess. Returns per settlement id what was booked."""
    tags, done, gaps = tuple(tags), {}, []
    for row in rows:
        try:
            sid, at, size, px, rate, change = settlement(row)
        except exec_reconcile.Unreadable as e:
            gaps.append(str(e))
            continue
        if sid in done:
            continue
        held = {t: q for t, q in held_at(at).items() if q}
        if not size or sum(held.values(), Decimal(0)) != size:
            gaps.append(f"the {at:%d %b %H:%M} UTC settlement {sid} was on {size} {instrument}; the journal's "
                        f"Strategies held {sum(held.values(), Decimal(0))} between them")
            continue
        done[sid] = {}
        for tag, amount in split(change, size, held).items():
            done[sid][tag] = (store.record_funding(tag, qty=float(held[tag]), price=float(px), rate=float(rate),
                                                   amount=amount, ts=at, kind="settled"), amount)
    if gaps:
        exec_reconcile.record(store, account, instrument, exec_reconcile.Result(instrument, tags, tuple(gaps)), now)
    return {"booked": done, "gaps": gaps}
