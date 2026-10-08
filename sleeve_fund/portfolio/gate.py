"""The portfolio gate around decide() and book_breach() (v2 P2-2), as the paper processes use it, behind one small
interface (Ledger) so the Data Architect's tables can replace the in-memory one without touching the rules here.

- check_order: one order that raises a position, under the ledger's lock: the portfolio's blocks first (a trim is
  never a way round a halt, a pause or a stale book), then read the book with every live reservation counted as
  filled, decide, reserve what was approved and record the decision with a sequence number (Advisor 1: first come,
  first served, replayable by P2-7). Any failure fails closed: no entry, and the reason says so.
- A reservation lives as long as its order (Advisor MUST FIX, 17:20 UK): attach_order ties it to the order, a partial
  fill reduces it, and the order's fill, reject or cancel releases it. sweep() only clears orphans: past the TTL, an
  order still live is cancelled (its own cancel then releases it), and only a reservation with no live order is
  released, with an alert. Nothing re-reserves each bar.
- entry_block: the portfolio's reasons no strategy may open anything now: halted, paused for the day, or the book's
  mark too old (PM yes, 6 Oct, board 23:03). Exits, stops, reduce-only orders, close-now and liquidations never ask.
- mark_book: the supervisor's mark of the whole fund, every few seconds: rolls the day, keeps the high-water mark and
  the halt's reference, and sets the halt or the pause (Advisor 9).
- resume_after_halt: the PM's Resume re-bases the halt's reference to the book then; the true HWM is kept.

Money is Decimal throughout (sleeve_fund.money); a float for money fails closed here as a rejected entry."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Protocol

from sleeve_fund.money import money
from sleeve_fund.portfolio.limits import Book, Decision, Holding, Intent, book_breach, decide, rejected
from sleeve_fund.risk import PortfolioProfile, trading_day

MARK_STALE = timedelta(seconds=60)  # a book mark older than this blocks new entries (PM yes, 6 Oct)
RESERVATION_TTL = timedelta(minutes=10)  # past this, a reservation is checked for an orphan (sweep)
CANCEL_CONFIRM = timedelta(seconds=60)  # a sweep's cancel unconfirmed this long is sent again, with an alert
NEVER = datetime(1970, 1, 1, tzinfo=timezone.utc)  # the "mark" of a book never marked, for the stale alert
RELEASE_REASONS = ("fill", "reject", "cancel", "ttl")  # the DA's release_reason values


def next_utc_midnight(now: datetime) -> datetime:
    """When the day `now` belongs to ends (risk.trading_day: a mark at exactly 00:00 is the day before's last)."""
    return datetime.combine(trading_day(now) + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)


@dataclass(frozen=True)
class PortfolioState:
    """What the supervisor last marked (QA's names). hwm: the true high-water mark since the last book reset, kept
    in the record. halt_reference: what the 15% halt measures from, the HWM until a PM Resume re-bases it to the book
    then (None: the HWM). day_start_equity: the book at the start of the UTC day (its last mark before 00:00)."""

    equity: Decimal | None = None
    mark_ts: datetime | None = None
    hwm: Decimal | None = None
    halt_reference: Decimal | None = None
    day: object = None  # the trading_day the day_start_equity belongs to
    day_start_equity: Decimal | None = None
    halted: str | None = None  # why, until the PM resumes
    paused_until: datetime | None = None
    paused: str | None = None
    stale_told: datetime | None = None  # the mark a stale-book alert was raised for, so it is raised once per spell

    def reference(self) -> Decimal | None:
        return self.halt_reference if self.halt_reference is not None else self.hwm


@dataclass(frozen=True)
class Reservation:
    """The room an approved order holds in the book until its fill, reject or cancel. qty: what is still unfilled,
    when the ledger was told it (reduce() needs it); holding: that much, counted as if filled."""

    id: int
    strategy: str
    holding: Holding
    at: datetime
    qty: Decimal | None = None
    order_id: str | None = None
    cancel_sent: datetime | None = None  # when the sweep last asked for its order's cancel


@dataclass(frozen=True)
class Record:
    seq: int
    strategy: str
    at: datetime
    intent: Intent
    decision: Decision
    profile_version: int


class Ledger(Protocol):
    """Where the portfolio's state, reservations and decisions live: the Data Architect's tables in paper, memory in
    tests and backtests. lock() serialises check_order across strategy processes (a Postgres advisory lock)."""

    def lock(self) -> AbstractContextManager[None]: ...
    def positions(self) -> tuple[Decimal, tuple[Holding, ...]]: ...  # (fund equity, every strategy's positions)
    def reservations(self) -> list[Reservation]: ...
    def reserve(self, strategy: str, holding: Holding, at: datetime, qty: Decimal | None = None) -> int: ...
    def attach_order(self, reservation_id: int, order_id: str) -> None: ...
    def cancel_sent(self, reservation_id: int, at: datetime) -> None: ...
    def reduce(self, reservation_id: int, filled_qty: Decimal) -> None: ...
    def release(self, reservation_id: int, reason: str) -> None: ...
    def record(self, strategy: str, at: datetime, intent: Intent, decision: Decision, version: int) -> int: ...
    def state(self) -> PortfolioState: ...
    def set_state(self, state: PortfolioState) -> None: ...
    def alert(self, kind: str, message: str, at: datetime) -> None: ...


def _scaled(h: Holding, share: Decimal) -> Holding:
    return Holding(h.underlying, h.notional * share, h.margin * share, h.risk * share)


@dataclass
class MemoryLedger:
    """The Ledger in memory, for tests, backtests and P2-7's portfolio run (the one in-memory implementation; Advisor
    condition 2). equity and held are what positions() reports."""

    equity: Decimal = Decimal(0)
    held: list[Holding] = field(default_factory=list)
    reserved: dict[int, Reservation] = field(default_factory=dict)
    released: list[tuple[int, str]] = field(default_factory=list)  # (reservation id, reason), in order
    records: list[Record] = field(default_factory=list)
    alerts: list[tuple[str, str, datetime]] = field(default_factory=list)
    current: PortfolioState = field(default_factory=PortfolioState)
    _ids: Iterator[int] = field(default_factory=lambda: itertools.count(1))
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @contextmanager
    def lock(self):
        with self._lock:
            yield

    def positions(self):
        return money(self.equity, "equity"), tuple(self.held)

    def reservations(self):
        return list(self.reserved.values())

    def reserve(self, strategy, holding, at, qty=None):
        rid = next(self._ids)
        self.reserved[rid] = Reservation(rid, strategy, holding, at, None if qty is None else money(qty, "qty"))
        return rid

    def attach_order(self, reservation_id, order_id):
        r = self.reserved.get(reservation_id)
        if r is not None:
            self.reserved[reservation_id] = replace(r, order_id=order_id)

    def cancel_sent(self, reservation_id, at):
        r = self.reserved.get(reservation_id)
        if r is not None:
            self.reserved[reservation_id] = replace(r, cancel_sent=at)

    def reduce(self, reservation_id, filled_qty):
        r = self.reserved.get(reservation_id)
        if r is None:
            return
        if r.qty is None:
            raise ValueError(f"reservation {reservation_id} was made without its quantity, so it can't be reduced")
        left = r.qty - money(filled_qty, "filled_qty")
        if left <= 0:
            self.release(reservation_id, "fill")
        else:
            self.reserved[reservation_id] = replace(r, qty=left, holding=_scaled(r.holding, left / r.qty))

    def release(self, reservation_id, reason):
        if reason not in RELEASE_REASONS:
            raise ValueError(f"a release reason is one of {RELEASE_REASONS}, not {reason!r}")
        if self.reserved.pop(reservation_id, None) is not None:
            self.released.append((reservation_id, reason))

    def record(self, strategy, at, intent, decision, version):
        seq = len(self.records) + 1
        self.records.append(Record(seq, strategy, at, intent, decision, version))
        return seq

    def state(self):
        return self.current

    def set_state(self, state):
        self.current = state

    def alert(self, kind, message, at):
        self.alerts.append((kind, message, at))


@dataclass(frozen=True)
class Checked:
    decision: Decision
    seq: int | None  # the decision's place in the first-come order; None when nothing could be recorded
    reservation: int | None  # attach it to the order; the order's fill, reject or cancel releases it


def _block(ledger: Ledger, now: datetime) -> tuple[str, str] | None:
    """entry_block's answer; the caller holds the ledger's lock."""
    st = ledger.state()
    if st.halted:
        return "halt", f"Portfolio halted: {st.halted}; no strategy opens anything, and only the PM clears that"
    if st.paused_until is not None and now < st.paused_until:
        return "pause", (f"Portfolio paused for the day: {st.paused}; no new entries until "
                                   f"{st.paused_until:%H:%M} UTC, when the day's start resets")
    if st.mark_ts is None or now - st.mark_ts > MARK_STALE:
        age = "never" if st.mark_ts is None else f"{(now - st.mark_ts).total_seconds():.0f} s ago"
        why = (f"No new entries: the fund's book was last marked {age}, over {MARK_STALE.seconds} s, so the "
               "portfolio limits can't be checked")
        spell = st.mark_ts or NEVER
        if st.stale_told != spell:
            ledger.set_state(replace(st, stale_told=spell))
            ledger.alert("portfolio_state_stale", why, now)
        return "portfolio_state_stale", why
    return None


def entry_block(ledger: Ledger, now: datetime) -> tuple[str, str] | None:
    """The portfolio's reason no strategy may open or add now, as (kind, why), or None; kind is the decision's
    limit_hit ("halt", "pause" or "portfolio_state_stale", gate_decisions' CHECK names). Halted until the PM resumes;
    paused until the next 00:00 UTC; or the book's mark older than MARK_STALE (alerted once per stale spell, when it
    starts). Never asked for exits, stops, reduce-only orders, close-now or liquidations."""
    with ledger.lock():
        return _block(ledger, now)


def check_order(ledger: Ledger, strategy: str, intent: Intent, profile: PortfolioProfile, now: datetime) -> Checked:
    """One order that raises `strategy`'s position. While the portfolio blocks entries it is rejected outright, never
    trimmed. Otherwise it is decided against the whole book with every unreleased reservation counted as filled, and
    the approved quantity is reserved before the lock is let go, so the next strategy sees it. Every check is
    recorded. Any error, the lock's included, fails closed: no entry, with the reason, an alert, and an 'error' row
    in the journal (limit 'lock_error' when the lock itself failed), written on its own after the check rolled back
    (p2-2-tables.md, Concurrency). The returned Checked has no seq. The error row takes one, outside the lock, so
    it holds no place in the first-come order: a replay in seq order skips outcome 'error' (CR F219-2)."""
    lock_failed = False
    try:
        with ExitStack() as held:
            try:
                held.enter_context(ledger.lock())
            except Exception:
                lock_failed = True
                raise
            block = _block(ledger, now)
            if block is not None:
                decision = rejected(intent.qty, block[0], f"Entry rejected: {block[1]}")
            else:
                equity, held_now = ledger.positions()
                book = Book(equity, (*held_now, *(r.holding for r in ledger.reservations())))
                decision = decide(book, intent, profile)
            rid = None
            if decision.approved_qty > 0:
                q = decision.approved_qty
                rid = ledger.reserve(strategy, Holding(intent.underlying, intent.side * q * intent.price,
                                                       q * intent.margin_per_unit, q * intent.risk_per_unit), now, q)
            seq = ledger.record(strategy, now, intent, decision, profile.version)
            return Checked(decision, seq, rid)
    except Exception as e:  # noqa: BLE001 - any failure here means no entry
        why = f"Entry rejected: the portfolio check couldn't run ({type(e).__name__}: {e}), so nothing is sent"
        try:
            ledger.alert("portfolio_check_failed", f"{strategy}: {why}", now)
        except Exception:  # noqa: BLE001 - the refusal stands whether or not the alert lands
            pass
        qty = getattr(intent, "qty", Decimal(0))
        qty = qty if isinstance(qty, Decimal) else Decimal(0)
        try:
            ledger.record(strategy, now, intent, Decision("error", Decimal(0), qty,
                                                          "lock_error" if lock_failed else None, why),
                          profile.version)
        except Exception:  # noqa: BLE001 - as for the alert
            pass
        return Checked(rejected(qty, None, why), None, None)


def mark_book(ledger: Ledger, equity, now: datetime, profile: PortfolioProfile) -> str | None:
    """The supervisor's mark of the whole fund. Returns "halt" when it has just halted the portfolio (the caller
    flattens every position through the exit path and halts every strategy), "pause" when it has just paused it,
    else None. The day's start is the last mark before 00:00 UTC (risk.trading_day); the HWM and the halt's reference
    only rise, until a book reset (or, for the reference, a PM Resume) re-bases them."""
    equity = money(equity, "equity")
    with ledger.lock():
        st = ledger.state()
        day = trading_day(now)
        if st.day != day:
            st = replace(st, day=day, day_start_equity=st.equity if st.equity else equity)
        st = replace(st, equity=equity, mark_ts=now, hwm=max(st.hwm or equity, equity),
                     halt_reference=None if st.halt_reference is None else max(st.halt_reference, equity))
        breach = book_breach(profile, equity, st.reference(), st.day_start_equity)
        acted = None
        if breach is not None and breach.action == "halt" and not st.halted:
            st, acted = replace(st, halted=breach.reason), "halt"
            ledger.alert("portfolio_halt", f"Portfolio halted: {breach.reason}", now)
        elif breach is not None and breach.action == "pause" and not (st.paused_until and now < st.paused_until):
            st, acted = replace(st, paused_until=next_utc_midnight(now), paused=breach.reason), "pause"
            ledger.alert("portfolio_pause", f"Portfolio paused for the day: {breach.reason}", now)
        ledger.set_state(st)
        return acted


def resume_after_halt(ledger: Ledger) -> PortfolioState:
    """The PM's Resume after a portfolio halt: clears it and re-bases the halt's reference to the book now, so it
    doesn't fire again at once; the next halt is the drawdown limit below this. A pause for the day stays until
    00:00 UTC. The true high-water mark is kept in the record."""
    with ledger.lock():
        st = ledger.state()
        st = replace(st, halted=None, halt_reference=st.equity)
        ledger.set_state(st)
        return st


def apply_flow(ledger: Ledger, amount) -> None:
    """Capital added (> 0) or withdrawn (< 0): the book, the HWM, the halt's reference and the day's start move by
    it, so a deposit isn't a gain and a withdrawal isn't a loss (Advisor 3)."""
    amount = money(amount, "amount")
    with ledger.lock():
        st = ledger.state()

        def moved(x):
            return None if x is None else x + amount

        ledger.set_state(replace(st, equity=moved(st.equity), hwm=moved(st.hwm),
                                 halt_reference=moved(st.halt_reference), day_start_equity=moved(st.day_start_equity)))


def sweep(ledger: Ledger, now: datetime, is_live: Callable[[str], bool], cancel: Callable[[str], None],
          ttl: timedelta = RESERVATION_TTL) -> list[Reservation]:
    """Orphans only (Advisor MUST FIX, 17:20 UK; refined 20:25 UK): a reservation lives as long as its order. Past
    `ttl`, one whose order is still live has that order cancelled and keeps its reservation until the cancel is
    confirmed: the order's own cancel releases it, or, if the order filled first, its fill converts the reservation
    into the position once (reduce, in the same transaction as the fill). An unconfirmed cancel is never released
    blind: every CANCEL_CONFIRM it is sent again with an alert, the reservation kept. A reservation with no order, or
    whose order is gone, is released as "ttl" with an alert: its process died before releasing it. Returns those."""
    gone = []
    with ledger.lock():
        for r in ledger.reservations():
            if now - r.at <= ttl:
                continue
            if r.order_id is not None and is_live(r.order_id):
                if r.cancel_sent is not None and now - r.cancel_sent <= CANCEL_CONFIRM:
                    continue  # asked; waiting for the venue's answer
                kind = "reservation_order_cancelled" if r.cancel_sent is None else "reservation_cancel_unconfirmed"
                ledger.cancel_sent(r.id, now)
                try:
                    cancel(r.order_id)
                    what = ("rested past its reservation's TTL; cancel sent" if r.cancel_sent is None else
                            f"cancel not confirmed after {CANCEL_CONFIRM.seconds} s; sent again")
                    ledger.alert(kind, f"{r.strategy}: order {r.order_id} {what}, and the reservation holds until "
                                       "the cancel or a fill lands", now)
                except Exception as e:  # noqa: BLE001 - the reservation stays while its order may still fill
                    ledger.alert("reservation_cancel_failed", f"{r.strategy}: couldn't cancel order {r.order_id} "
                                                              f"({type(e).__name__}: {e}); its reservation stays", now)
                continue
            ledger.release(r.id, "ttl")
            gone.append(r)
            ledger.alert("reservation_expired", f"{r.strategy}: a portfolio reservation of "
                                                f"{abs(r.holding.notional):,.2f} in {r.holding.underlying} had no "
                                                f"live order after {ttl}; released", now)
    return gone
