"""The portfolio gate around decide() and book_breach() (v2 P2-2), as the paper processes use it, behind one small
interface (Ledger) so the Data Architect's tables can replace the in-memory one without touching the rules here.

- check_order: one order that raises a position, under the ledger's lock: read the book (resting entries reserved
  in it), decide, reserve what was approved, record the decision with a sequence number (Advisor 1: first come,
  first served, replayable by P2-7). Any failure fails closed: no entry, and the reason says so.
- entry_block: the portfolio's reasons no strategy may open anything now: halted, paused for the day, or the book's
  mark too old (PM yes, 6 Oct, board 23:03). Exits, stops, reduce-only orders, close-now and liquidations never ask.
- mark_book: the supervisor's mark of the whole fund, every few seconds: rolls the day, keeps the reference, and
  sets the halt or the pause (Advisor 9).
- resume_after_halt: the PM's Resume re-bases the halt's reference to the book then (Advisor 9)."""

from __future__ import annotations

import itertools
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Iterator, Protocol

from sleeve_fund.portfolio.limits import Book, Decision, Holding, Intent, book_breach, decide
from sleeve_fund.risk import PortfolioProfile, trading_day

MARK_STALE = timedelta(seconds=60)  # a book mark older than this blocks new entries (PM yes, 6 Oct)
RESERVATION_TTL = timedelta(minutes=10)
NEVER = datetime(1970, 1, 1, tzinfo=timezone.utc)  # the "mark" of a book never marked, for the stale alert  # a reservation its process never released is swept, with an alert


@dataclass(frozen=True)
class PortfolioState:
    """What the supervisor last marked. reference: the high-water mark since the last book reset, re-based at a PM
    Resume after a portfolio halt; day_start: the book at the start of the UTC day (its last mark before 00:00)."""

    equity: float | None = None
    marked_at: datetime | None = None
    reference: float | None = None
    day: object = None  # the trading_day the day_start belongs to
    day_start: float | None = None
    halted: str | None = None  # why, until the PM resumes
    paused_until: datetime | None = None
    paused: str | None = None
    stale_told: datetime | None = None  # the mark a stale-book alert was raised for, so it is raised once per spell


@dataclass(frozen=True)
class Reservation:
    id: int
    strategy: str
    holding: Holding
    at: datetime


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

    def lock(self) -> Iterator[None]: ...
    def positions(self) -> tuple[float, tuple[Holding, ...]]: ...  # (fund equity, every strategy's positions)
    def reservations(self) -> list[Reservation]: ...
    def reserve(self, strategy: str, holding: Holding, at: datetime) -> int: ...
    def release(self, reservation_id: int) -> None: ...
    def record(self, strategy: str, at: datetime, intent: Intent, decision: Decision, version: int) -> int: ...
    def state(self) -> PortfolioState: ...
    def set_state(self, state: PortfolioState) -> None: ...
    def alert(self, kind: str, message: str, at: datetime) -> None: ...


@dataclass
class MemoryLedger:
    """The Ledger in memory, for tests and single-process runs. equity and held are what positions() reports."""

    equity: float = 0.0
    held: list[Holding] = field(default_factory=list)
    reserved: dict[int, Reservation] = field(default_factory=dict)
    records: list[Record] = field(default_factory=list)
    alerts: list[tuple[str, str, datetime]] = field(default_factory=list)
    current: PortfolioState = field(default_factory=PortfolioState)
    _ids: Iterator[int] = field(default_factory=lambda: itertools.count(1))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @contextmanager
    def lock(self):
        with self._lock:
            yield

    def positions(self):
        return self.equity, tuple(self.held)

    def reservations(self):
        return list(self.reserved.values())

    def reserve(self, strategy, holding, at):
        rid = next(self._ids)
        self.reserved[rid] = Reservation(rid, strategy, holding, at)
        return rid

    def release(self, reservation_id):
        self.reserved.pop(reservation_id, None)

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
    reservation: int | None  # release it on fill, reject or cancel


def check_order(ledger: Ledger, strategy: str, intent: Intent, profile: PortfolioProfile, now: datetime) -> Checked:
    """One order that raises `strategy`'s position, decided against the whole book with every unreleased reservation
    counted as filled. The approved quantity is reserved before the lock is let go, so the next strategy sees it.
    Any error, the lock's included, fails closed: no entry, with the reason."""
    try:
        with ledger.lock():
            equity, held = ledger.positions()
            book = Book(equity, (*held, *(r.holding for r in ledger.reservations())))
            decision = decide(book, intent, profile)
            rid = None
            if decision.qty > 0:
                rid = ledger.reserve(strategy, Holding(
                    intent.underlying, intent.side * decision.qty * intent.price,
                    decision.qty * intent.margin_per_unit, decision.qty * intent.risk_per_unit), now)
            seq = ledger.record(strategy, now, intent, decision, profile.version)
            return Checked(decision, seq, rid)
    except Exception as e:  # noqa: BLE001 - any failure here means no entry
        why = f"Entry refused: the portfolio check couldn't run ({type(e).__name__}: {e}), so nothing is sent"
        return Checked(Decision("refused", 0.0, intent.qty, None, why), None, None)


def entry_block(ledger: Ledger, now: datetime) -> tuple[str, str] | None:
    """The portfolio's reason no strategy may open or add now, as (kind, why), or None. Halted until the PM resumes;
    paused until the next 00:00 UTC; or the book's mark older than MARK_STALE (alerted once per stale spell, when it
    starts). Never asked for exits, stops, reduce-only orders, close-now or liquidations."""
    st = ledger.state()
    if st.halted:
        return "portfolio_halt", f"Portfolio halted: {st.halted}; no strategy opens anything, and only the PM clears that"
    if st.paused_until is not None and now < st.paused_until:
        return "portfolio_pause", (f"Portfolio paused for the day: {st.paused}; no new entries until "
                                   f"{st.paused_until:%H:%M} UTC, when the day's start resets")
    if st.marked_at is None or now - st.marked_at > MARK_STALE:
        age = "never" if st.marked_at is None else f"{(now - st.marked_at).total_seconds():.0f} s ago"
        why = (f"No new entries: the fund's book was last marked {age}, over {MARK_STALE.seconds} s, so the "
               "portfolio limits can't be checked")
        spell = st.marked_at or NEVER
        if st.stale_told != spell:
            with ledger.lock():
                st = ledger.state()
                if st.stale_told != spell:
                    ledger.set_state(replace(st, stale_told=spell))
                    ledger.alert("book_mark_stale", why, now)
        return "book_mark_stale", why
    return None


def mark_book(ledger: Ledger, equity: float, now: datetime, profile: PortfolioProfile) -> str | None:
    """The supervisor's mark of the whole fund. Returns "halt" when it has just halted the portfolio (the caller
    flattens every position through the exit path and halts every strategy), "pause" when it has just paused it,
    else None. The day's start is the last mark before 00:00 UTC (risk.trading_day); the reference only rises, until
    a book reset or a PM Resume re-bases it. Stale reservations are swept here, with an alert."""
    st = ledger.state()
    day = trading_day(now)
    if st.day != day:
        st = replace(st, day=day, day_start=st.equity if st.equity else equity)
    st = replace(st, equity=equity, marked_at=now, reference=max(st.reference or equity, equity))
    sweep(ledger, now)
    breach = book_breach(profile, equity, st.reference, st.day_start)
    acted = None
    if breach is not None and breach.action == "halt" and not st.halted:
        st, acted = replace(st, halted=breach.reason), "halt"
        ledger.alert("portfolio_halt", f"Portfolio halted: {breach.reason}", now)
    elif breach is not None and breach.action == "pause" and not (st.paused_until and now < st.paused_until):
        midnight = datetime.combine(trading_day(now) + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        st, acted = replace(st, paused_until=midnight, paused=breach.reason), "pause"
        ledger.alert("portfolio_pause", f"Portfolio paused for the day: {breach.reason}", now)
    ledger.set_state(st)
    return acted


def resume_after_halt(ledger: Ledger) -> PortfolioState:
    """The PM's Resume after a portfolio halt: clears it and re-bases the halt's reference to the book now, so it
    doesn't fire again at once; the next halt is drawdown_halt below this. A pause for the day stays until 00:00 UTC.
    The track record keeps the true high-water mark: this is the halt's reference only."""
    st = ledger.state()
    st = replace(st, halted=None, reference=st.equity)
    ledger.set_state(st)
    return st


def apply_flow(ledger: Ledger, amount: float) -> None:
    """Capital added (> 0) or withdrawn (< 0): the reference and the day's start move by it, so a deposit isn't a
    gain and a withdrawal isn't a loss (Advisor 3)."""
    st = ledger.state()
    ledger.set_state(replace(st, reference=(st.reference or 0.0) + amount if st.reference is not None else None,
                             day_start=st.day_start + amount if st.day_start is not None else None,
                             equity=st.equity + amount if st.equity is not None else None))


def sweep(ledger: Ledger, now: datetime, ttl: timedelta = RESERVATION_TTL) -> list[Reservation]:
    """Release reservations older than ttl (their process died before releasing them), with an alert each, so a
    crashed process can't hold back every later entry (Advisor 10)."""
    gone = [r for r in ledger.reservations() if now - r.at > ttl]
    for r in gone:
        ledger.release(r.id)
        ledger.alert("reservation_swept", f"{r.strategy}: a portfolio reservation of {abs(r.holding.notional):,.2f} "
                                          f"in {r.holding.underlying} was never released; swept after {ttl}", now)
    return gone
