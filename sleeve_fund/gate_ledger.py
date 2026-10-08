"""The portfolio gate's Ledger on the journal (v2 P2-2; the DA's tables, migrations 0010 and 0011): what paper's
processes share, where sleeve_fund.portfolio.gate.MemoryLedger is the in-memory one. The rules stay in the gate; this
only keeps its state, reservations and decisions.

- lock(): one transaction. On Postgres it starts with pg_advisory_xact_lock, held until commit or rollback, so a dead
  process can't keep it. On SQLite it is a lock within one process: for the lab and tests only, never two processes. Every ledger call made inside it uses that
  transaction, so check_order's read, decision, reservation and record commit together or not at all.
- seq: the decision's place in the first-come order, taken inside the lock (gate_decision_seq on Postgres, max + 1 on
  SQLite), so seq order is decision order. A rollback leaves a gap in the sequence; only the order matters (P2-7).
- A reservation's id is its decision's id (gate_reservations.decision_id). reserve() runs before record() in
  check_order, so it allocates that id and record() writes the decision, then the reservation under it.
- using(conn): a fill, reject or cancel is written by its own transaction; the release (or reduce) joins it, so the
  order's end and its reservation's end commit together (p2-2-tables.md).
- record() needs the journal's tags (TaggedIntent): the strategy's intent id, its kind, the bar and the stage. A plain
  Intent is refused, which check_order turns into no entry.
- Nothing here is wired into paper or backtest yet (P2-1b, after CASH-ONE-PATH): positions() comes from the caller."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import func, insert, select, text, update
from sqlalchemy.engine import Connection

from sleeve_fund.money import money, stored
from sleeve_fund.portfolio.gate import NEVER, RELEASE_REASONS, RESERVATION_TTL, PortfolioState, Reservation
from sleeve_fund.portfolio.limits import Book, Decision, Holding, Intent
from sleeve_fund.store import (
    GATE_STAGES,
    Store,
    _aware,
    book_marks_t,
    events_t,
    gate_decisions_t,
    gate_reservations_t,
    orders_t,
    portfolio_profile_t,
    portfolio_state_t,
    sleeves_t,
    utcnow,
)

LOCK_SQL = text("SELECT pg_advisory_xact_lock(hashtext('portfolio_gate'))")
# A stalled holder (a slow read, a hung connection) must not hold every strategy's entry checks without bound: past
# this the lock raises and check_order fails closed as lock_error (HoQA F219-Q1). Exits never take this lock.
LOCK_TIMEOUT = text("SET LOCAL lock_timeout = '2s'")


@dataclass(frozen=True)
class TaggedIntent(Intent):
    """An Intent with what its journal row needs: the strategy's id for it (unique per strategy, bar and stage), its
    kind (entry, add, band add, rebalance increase, reversal open leg), the bar it was decided on, the stage ('submit',
    or 'fill' for the backstop) and the regime weight and state it was sized under. decide() reads only the Intent."""

    intent_id: str = ""
    kind: str = "entry"
    bar_ts: datetime | None = None
    stage: str = "submit"
    regime_weight: Decimal | None = None
    regime_state: str | None = None


@dataclass(frozen=True)
class Journaled:
    """One decision as the journal holds it, for a replay in first-come order (P2-7 Done-when 1)."""

    seq: int
    strategy: str
    intent_id: str
    kind: str
    bar_ts: datetime
    stage: str
    decided_at: datetime
    underlying: str
    price: Decimal
    decision: Decision  # its reason is not journaled: ""
    profile_version: int


def _dec(x) -> Decimal | None:
    return None if x is None else stored(x)


class DbLedger:
    """The Ledger on the journal. positions: () -> (fund equity, every strategy's positions as Holdings), read inside
    the lock; the caller builds it from the store (P2-1b). Alerts go to the events table (the alerts inbox)."""

    def __init__(self, store: Store, positions: Callable[[], tuple[Decimal, tuple[Holding, ...]]]):
        self.store = store
        self.engine = store.engine
        self._positions = positions
        self._pg = self.engine.dialect.name == "postgresql"
        self._process_lock = threading.RLock()  # SQLite's lock; on Postgres the advisory lock serialises
        self._local = threading.local()

    # --- transactions --------------------------------------------------------------------------------------------

    @contextmanager
    def lock(self) -> Iterator[None]:
        loc = self._local
        if getattr(loc, "conn", None) is not None:
            if not getattr(loc, "locked", None):  # inside using(): no gate lock to re-enter (CR F219-4)
                raise RuntimeError("lock() inside using(): an entry check never runs in a fill's transaction")
            yield  # already inside (re-entrant, as MemoryLedger's RLock)
            return
        with nullcontext() if self._pg else self._process_lock, self.engine.begin() as conn:
            if self._pg:
                conn.execute(LOCK_TIMEOUT)
                conn.execute(LOCK_SQL)
            loc.conn, loc.locked, loc.book, loc.reserved, loc.pending = conn, True, None, None, None
            try:
                yield
            finally:
                loc.conn = loc.locked = loc.book = loc.reserved = loc.pending = None

    @contextmanager
    def using(self, conn: Connection) -> Iterator[None]:
        """Ledger calls inside join `conn`'s transaction: a fill, reject or cancel and its release commit together."""
        loc = self._local
        if getattr(loc, "conn", None) is not None:
            raise RuntimeError("using() inside the ledger's lock or another using(): one transaction at a time")
        loc.conn = conn
        try:
            yield
        finally:
            loc.conn = None

    @contextmanager
    def _tx(self, audit: bool = False) -> Iterator[Connection]:
        """The lock's or the fill's transaction when inside one, else a transaction of its own. audit: a failed entry
        check's error row or alert. Inside a fill's transaction (using(), never the lock) only a failed check writes
        those, so on Postgres they take their own transaction and outlive a fill that rolls back (CR F219-5). SQLite
        shares one connection, so there they join it. That own transaction can't see rows the fill's hasn't committed:
        a strategy created inside it would fail the audit row's lookup (CR F228-1; strategies are created at setup)."""
        loc = self._local
        conn = getattr(loc, "conn", None)
        if conn is not None and not (audit and self._pg and not getattr(loc, "locked", None)):
            yield conn
            return
        with nullcontext() if self._pg else self._process_lock, self.engine.begin() as conn:
            yield conn

    def _sleeve_id(self, conn: Connection, strategy: str) -> int:
        sid = conn.execute(select(sleeves_t.c.id).where(sleeves_t.c.name == strategy)).scalar()
        if sid is None:
            raise LookupError(f"no strategy named {strategy!r}")
        return sid

    # --- the book --------------------------------------------------------------------------------------------------

    def positions(self):
        equity, held = self._positions()
        held = tuple(held)
        self._local.book = (equity, held)  # for the decision's figures, if record() follows in this lock
        return equity, held

    def reservations(self) -> list[Reservation]:
        r, d, s = gate_reservations_t, gate_decisions_t, sleeves_t
        q = (select(r, s.c.name, d.c.order_id).join(s, s.c.id == r.c.sleeve_id).join(d, d.c.id == r.c.decision_id)
             .where(r.c.released_at.is_(None)).order_by(r.c.decision_id))
        with self._tx() as conn:
            out = [Reservation(row.decision_id, row.name,
                               Holding(row.underlying, money(row.notional), money(row.margin), money(row.open_risk)),
                               _aware(row.created_at), money(row.remaining_qty), row.order_id,
                               _aware(row.cancel_sent_at)) for row in conn.execute(q)]
        self._local.reserved = out
        return out

    def reserve(self, strategy: str, holding: Holding, at: datetime, qty: Decimal | None = None) -> int:
        loc = self._local
        if not getattr(loc, "locked", None):
            raise RuntimeError("reserve() runs inside the ledger's lock, with the decision it belongs to")
        if qty is None:
            raise ValueError("a reservation in the journal needs its quantity (remaining_qty)")
        if loc.pending is not None:
            raise RuntimeError("one reservation per decision: record() the last one first")
        conn = loc.conn
        if self._pg:
            rid = conn.execute(text("SELECT nextval(pg_get_serial_sequence('gate_decisions', 'id'))")).scalar()
        else:
            rid = (conn.execute(select(func.max(gate_decisions_t.c.id))).scalar() or 0) + 1
        loc.pending = (rid, strategy, holding, at, abs(money(qty, "qty")))
        return rid

    def record(self, strategy: str, at: datetime, intent: Intent, decision: Decision, version: int) -> int:
        tags = self._tags(strategy, at, intent)
        loc = self._local
        pending = getattr(loc, "pending", None)
        with self._tx(audit=decision.outcome == "error") as conn:
            if self._pg:
                seq = conn.execute(text("SELECT nextval('gate_decision_seq')")).scalar()
            else:
                seq = (conn.execute(select(func.max(gate_decisions_t.c.seq))).scalar() or 0) + 1
            sid = self._sleeve_id(conn, strategy)
            row = {**tags, "seq": seq, "sleeve_id": sid, "underlying": intent.underlying, "profile_version": version,
                   "outcome": decision.outcome, "limit_hit": decision.limit_hit,
                   "requested_qty": stored(decision.requested_qty), "approved_qty": stored(decision.approved_qty),
                   "price": stored(intent.price), "decided_at": at, **self._figures(intent.underlying)}
            if pending is not None:
                rid, who, holding, made, qty = pending
                if who != strategy or decision.approved_qty <= 0:
                    raise RuntimeError("the reservation made in this lock isn't this decision's")
                row["id"] = rid
            conn.execute(insert(gate_decisions_t), [row])
            if pending is not None:
                conn.execute(insert(gate_reservations_t), [{
                    "decision_id": rid, "sleeve_id": sid, "underlying": holding.underlying, "remaining_qty": qty,
                    "notional": stored(holding.notional), "margin": stored(holding.margin),
                    "open_risk": stored(holding.risk), "created_at": made, "expires_at": made + RESERVATION_TTL}])
                loc.pending = None
        return seq

    def _tags(self, strategy: str, at: datetime, intent: Intent) -> dict:
        """The journal's columns the core Intent doesn't carry. Refuses an untagged intent: a row that can't be tied
        to its strategy's intent is never written."""
        if not isinstance(intent, TaggedIntent) or not intent.intent_id or intent.bar_ts is None:
            raise ValueError("the gate's journal needs a TaggedIntent with its intent_id and bar_ts")
        if intent.stage not in GATE_STAGES:
            raise ValueError(f"a decision's stage is one of {GATE_STAGES}, not {intent.stage!r}")
        return {"intent_id": intent.intent_id, "intent": intent.kind, "bar_ts": intent.bar_ts, "stage": intent.stage,
                "regime_weight": intent.regime_weight, "regime_state": intent.regime_state}

    def _figures(self, underlying: str) -> dict:
        """The book before this entry, reservations included, as check_order read it in this lock; null when it
        didn't (a blocked portfolio, or a check that couldn't run)."""
        book, reserved = getattr(self._local, "book", None), getattr(self._local, "reserved", None)
        if book is None or reserved is None:
            return {}
        b = Book(book[0], (*book[1], *(r.holding for r in reserved)))
        return {"book_equity": stored(b.equity), "gross": stored(b.gross()), "net_underlying": stored(b.net(underlying)),
                "margin_used": stored(b.margin()), "open_risk": stored(b.open_risk())}

    def replayable(self) -> list[Journaled]:
        """Every decision a replay follows, in seq order. 'error' rows are left out: they take their seq outside the
        lock, so they hold no place in the first-come order and decided nothing (CR F219-2)."""
        d, s = gate_decisions_t, sleeves_t
        q = (select(d, s.c.name).join(s, s.c.id == d.c.sleeve_id).where(d.c.outcome != "error").order_by(d.c.seq))
        with self._tx() as conn:
            return [Journaled(row.seq, row.name, row.intent_id, row.intent, _aware(row.bar_ts), row.stage,
                              _aware(row.decided_at), row.underlying, money(row.price),
                              Decision(row.outcome, money(row.approved_qty), money(row.requested_qty), row.limit_hit,
                                       ""), row.profile_version) for row in conn.execute(q)]

    # --- a reservation's life ------------------------------------------------------------------------------------------

    def _active(self, conn: Connection, reservation_id: int, for_update: bool = False):
        r = gate_reservations_t
        q = select(r).where(r.c.decision_id == reservation_id, r.c.released_at.is_(None))
        return conn.execute(q.with_for_update() if for_update else q).first()

    def attach_order(self, reservation_id: int, order_id: str) -> None:
        with self._tx() as conn:
            if self._active(conn, reservation_id) is not None:
                conn.execute(update(gate_decisions_t).where(gate_decisions_t.c.id == reservation_id)
                             .values(order_id=order_id))

    def cancel_sent(self, reservation_id: int, at: datetime) -> None:
        r = gate_reservations_t
        with self._tx() as conn:
            conn.execute(update(r).where(r.c.decision_id == reservation_id, r.c.released_at.is_(None))
                         .values(cancel_sent_at=at))

    def reduce(self, reservation_id: int, filled_qty: Decimal) -> None:
        """A partial fill: what is left reserves its share of the notional, margin and risk. Filled to nothing, it is
        released as 'fill'. filled_qty is this fill's quantity, not the order's cumulative fill. Call it in the fill's
        transaction (using()). The row is locked for the read (DA F219-1): a second fill or the sweep's release waits
        for this one, so no fill is lost and a released row is never written to."""
        r = gate_reservations_t
        filled = money(filled_qty, "filled_qty")
        with self._tx() as conn:
            row = self._active(conn, reservation_id, for_update=True)
            if row is None:
                return
            qty = money(row.remaining_qty)
            left = qty - abs(filled)
            if left <= 0:
                self._release(conn, reservation_id, "fill")
                return
            share = left / qty
            conn.execute(update(r).where(r.c.decision_id == reservation_id, r.c.released_at.is_(None)).values(
                remaining_qty=stored(left), notional=stored(money(row.notional) * share),
                margin=stored(money(row.margin) * share), open_risk=stored(money(row.open_risk) * share)))

    def release(self, reservation_id: int, reason: str) -> None:
        if reason not in RELEASE_REASONS:
            raise ValueError(f"a release reason is one of {RELEASE_REASONS}, not {reason!r}")
        with self._tx() as conn:
            self._release(conn, reservation_id, reason)

    def _release(self, conn: Connection, reservation_id: int, reason: str) -> None:
        r = gate_reservations_t
        conn.execute(update(r).where(r.c.decision_id == reservation_id, r.c.released_at.is_(None))
                     .values(released_at=utcnow(), release_reason=reason))

    # --- the portfolio's state -----------------------------------------------------------------------------------------

    def state(self) -> PortfolioState:
        with self._tx() as conn:
            row = conn.execute(select(portfolio_state_t).where(portfolio_state_t.c.id == 1)).first()
        if row is None:
            return PortfolioState()
        hwm, ref = _num(row.hwm), _num(row.reference_equity)
        return PortfolioState(
            equity=_num(row.book_equity), mark_ts=_aware(row.marked_at), hwm=hwm,
            halt_reference=None if ref == hwm else ref,  # the row keeps reference(); equal to the HWM is no re-base
            day=row.day_start, day_start_equity=_num(row.day_start_equity),
            halted=row.halt_reason if row.status == "halted" else None, paused_until=_aware(row.paused_until),
            paused=row.pause_reason, stale_told=_aware(row.stale_told_at))

    def set_state(self, state: PortfolioState) -> None:
        if state.halted:
            status = "halted"
        elif state.paused_until is not None and (state.mark_ts is None or state.mark_ts < state.paused_until):
            status = "paused"
        else:
            status = "ok"
        row = {"status": status, "paused_until": state.paused_until, "halt_reason": state.halted,
               "pause_reason": state.paused, "reference_equity": _dec(state.reference()), "hwm": _dec(state.hwm),
               "day_start_equity": _dec(state.day_start_equity), "day_start": state.day,
               "book_equity": _dec(state.equity), "marked_at": state.mark_ts,
               # the stale alert's marker belongs to the spell of this mark; the first fresh mark after it clears it
               "stale_told_at": state.stale_told if state.stale_told == (state.mark_ts or NEVER) else None,
               "updated_at": utcnow()}
        t = portfolio_state_t
        with self._tx() as conn:
            row["profile_version"] = conn.execute(select(func.max(portfolio_profile_t.c.version))).scalar()
            if conn.execute(update(t).where(t.c.id == 1).values(**row)).rowcount == 0:
                conn.execute(insert(t), [{"id": 1, **row}])

    def ensure_state(self) -> None:
        """Write the state row (never marked: entries blocked as stale) if there is none. The portfolio gate is in
        force from then on (runtime.portfolio_block)."""
        t = portfolio_state_t
        with self.lock():
            if self._local.conn.execute(select(t.c.id).where(t.c.id == 1)).first() is None:
                self.set_state(PortfolioState())

    def last_book_mark(self) -> datetime | None:
        """When the newest book_marks row was written (the supervisor writes one a minute, read back across restarts)."""
        with self._tx() as conn:
            return _aware(conn.execute(select(func.max(book_marks_t.c.ts))).scalar())

    def order_status(self, order_id: str) -> str | None:
        """An order row's status, in the lock's or fill's transaction when inside one."""
        with self._tx() as conn:
            return conn.execute(select(orders_t.c.status).where(orders_t.c.order_id == order_id)).scalar()

    def alert(self, kind: str, message: str, at: datetime) -> None:
        with self._tx(audit=True) as conn:
            conn.execute(insert(events_t).values(sleeve=None, ts=at, level="error", kind=kind, message=message))

    def write_book_mark(self, ts: datetime, book: Book, state: PortfolioState, profile_version: int) -> None:
        """The supervisor's minute (and status-change) mark into book_marks, from the book it marked: a cache, rebuilt
        from fills. Kept out of set_state so a mark never waits on reading the positions."""
        nets: dict[str, Decimal] = {}
        for h in book.holdings:
            nets[h.underlying] = nets.get(h.underlying, Decimal(0)) + h.notional
        top = max(nets, key=lambda u: abs(nets[u]), default=None)
        status = "halted" if state.halted else "paused" if state.paused_until and ts < state.paused_until else "ok"
        with self._tx() as conn:
            conn.execute(insert(book_marks_t), [{
                "ts": ts, "equity": stored(book.equity), "gross": stored(book.gross()),
                "net_max": stored(abs(nets[top]) if top else Decimal(0)), "net_underlying": top,
                "margin_used": stored(book.margin()), "open_risk": stored(book.open_risk()),
                "hwm": stored(state.hwm), "reference_equity": stored(state.reference()),
                "day_start_equity": stored(state.day_start_equity), "status": status,
                "profile_version": profile_version}])


def _num(x) -> Decimal | None:
    return None if x is None else money(x)
