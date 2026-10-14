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
- apply_flow: capital into or out of the book moves the book by it, and scales the HWM, the reference and the day's
  start with it, so the drawdown and the day's change keep their percentages.

Money is Decimal throughout (sleeve_fund.money); a float for money fails closed here as a rejected entry."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Protocol

from sleeve_fund.money import PRECISION, QUANTUM, money
from sleeve_fund.portfolio.limits import Book, Decision, Holding, Intent, book_breach, decide, decide_pair, rejected
from sleeve_fund.risk import PortfolioProfile, trading_day

EMPTY = Decimal("0.01")  # a book worth under a cent holds nothing: empty (F282-3, F-C3B-A)
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
    # MUST-F286-H4 (Advisor 14 Oct 13:35 UK): a withdrawal that empties the book carries its ratios, taken from the book
    # before it, to the next capital's first mark (mark_book), which restores them: set only while unmarked.
    carried_hwm_drawdown: Decimal | None = None  # 1 - book / HWM
    carried_reference_drawdown: Decimal | None = None  # 1 - book / the halt's reference; None: the reference is the HWM
    carried_day_return: Decimal | None = None  # book / the day's start - 1, for carried_day only
    carried_day: object = None  # the UTC day carried_day_return belongs to

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
    def record(self, strategy: str, at: datetime, intent: Intent, decision: Decision, version: int | None,
               compliance_version: int | None = None) -> int: ...
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

    def record(self, strategy, at, intent, decision, version, compliance_version=None):
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


def block_reason(st: PortfolioState, now: datetime) -> tuple[str, str] | None:
    """entry_block's answer from one read of the state, without telling the stale spell: CHOKE's lock-free read
    (runtime.portfolio_block, CR F229-2), which takes the lock through entry_block only when a spell is untold."""
    if st.halted:
        return "halt", f"Portfolio halted: {st.halted}; no strategy opens anything, and only the PM clears that"
    if st.paused_until is not None and now < st.paused_until:
        return "pause", (f"Portfolio paused for the day: {st.paused}; no new entries until "
                                   f"{st.paused_until:%H:%M} UTC, when the day's start resets")
    if st.mark_ts is None or now - st.mark_ts > MARK_STALE:
        age = "never" if st.mark_ts is None else f"{(now - st.mark_ts).total_seconds():.0f} s ago"
        return "portfolio_state_stale", (f"No new entries: the fund's book was last marked {age}, over "
                                         f"{MARK_STALE.seconds} s, so the portfolio limits can't be checked")
    return None


def stale_untold(st: PortfolioState) -> bool:
    """Whether the stale spell of this state's mark hasn't been alerted yet."""
    return st.stale_told != (st.mark_ts or NEVER)


IDENTITY_HOLD = "identity_hold"  # the decision's reason kind while the fund's book check holds entries (RECON-B)


def _block(ledger: Ledger, now: datetime) -> tuple[str, str] | None:
    """entry_block's answer; the caller holds the ledger's lock. A ledger on the journal also holds every entry while
    the fund's book check has failed (identity_hold: latched until the PM's Resume, Advisor RECON-B)."""
    held = getattr(ledger, "identity_hold", None)
    why = held() if callable(held) else None
    if why:
        return IDENTITY_HOLD, (f"No new entries: the fund's book check failed ({why}); exits and stops still go out, "
                               "and only the PM's Resume lifts it")
    st = ledger.state()
    hit = block_reason(st, now)
    if hit is not None and hit[0] == "portfolio_state_stale" and stale_untold(st):
        ledger.set_state(replace(st, stale_told=st.mark_ts or NEVER))
        ledger.alert("portfolio_state_stale", hit[1], now)
    return hit


def entry_block(ledger: Ledger, now: datetime) -> tuple[str, str] | None:
    """The portfolio's reason no strategy may open or add now, as (kind, why), or None; kind is the decision's
    limit_hit ("halt", "pause" or "portfolio_state_stale", gate_decisions' CHECK names). Halted until the PM resumes;
    paused until the next 00:00 UTC; or the book's mark older than MARK_STALE (alerted once per stale spell, when it
    starts). Never asked for exits, stops, reduce-only orders, close-now or liquidations."""
    with ledger.lock():
        return _block(ledger, now)


def _tests(ledger: Ledger, profile: PortfolioProfile | None) -> tuple[PortfolioProfile, int | None]:
    """The tests an idea is checked against, and the compliance version to record: `profile` as given (no version),
    or with None the PM's compliance tests in force, read through the ledger inside its lock at evaluation, so an edit
    between the idea and the gate applies (Bridge Drop A; BRIDGE-3-QA). A read that fails raises: no entry."""
    if profile is not None:
        return profile, None
    tests = ledger.compliance()
    return tests, tests.version


def _record(ledger: Ledger, strategy: str, now: datetime, intent: Intent, decision: Decision, version: int | None,
            cv: int | None) -> int:
    """ledger.record, naming the compliance version only when there is one (a Ledger from before Drop A takes five)."""
    if cv is None:
        return ledger.record(strategy, now, intent, decision, version)
    return ledger.record(strategy, now, intent, decision, None, cv)


def check_order(ledger: Ledger, strategy: str, intent: Intent, profile: PortfolioProfile | None,
                now: datetime) -> Checked:
    """One order that raises `strategy`'s position. While the portfolio blocks entries it is rejected outright, never
    trimmed. Otherwise it is decided against the whole book with every unreleased reservation counted as filled, and
    the approved quantity is reserved before the lock is let go, so the next strategy sees it. Every check is
    recorded. Any error, the lock's included, fails closed: no entry, with the reason, an alert, and an 'error' row
    in the journal (limit 'lock_error' when the lock itself failed), written on its own after the check rolled back
    (p2-2-tables.md, Concurrency). The returned Checked has no seq. The error row takes one, outside the lock, so
    it holds no place in the first-come order: a replay in seq order skips outcome 'error' (CR F219-2). profile None:
    the compliance tests in force (_tests)."""
    lock_failed, version, cv = False, getattr(profile, "version", None), None
    try:
        with ExitStack() as held:
            try:
                held.enter_context(ledger.lock())
            except Exception:
                lock_failed = True
                raise
            tests, cv = _tests(ledger, profile)
            block = _block(ledger, now)
            if block is not None:
                decision = rejected(intent.qty, block[0], f"Entry rejected: {block[1]}")
            else:
                equity, held_now = ledger.positions()
                book = Book(equity, (*held_now, *(r.holding for r in ledger.reservations())))
                decision = decide(book, intent, tests)
            rid = None
            if decision.approved_qty > 0:
                q = decision.approved_qty
                rid = ledger.reserve(strategy, Holding(intent.underlying, intent.side * q * intent.price,
                                                       q * intent.margin_per_unit, q * intent.risk_per_unit), now, q)
            seq = _record(ledger, strategy, now, intent, decision, tests.version, cv)
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
            _record(ledger, strategy, now, intent, Decision("error", Decimal(0), qty,
                                                            "lock_error" if lock_failed else None, why), version, cv)
        except Exception:  # noqa: BLE001 - as for the alert
            pass
        return Checked(rejected(qty, None, why), None, None)


def check_pair(ledger: Ledger, strategy: str | tuple[str, str], first: Intent, second: Intent,
               profile: PortfolioProfile | None, now: datetime) -> tuple[Checked, Checked]:
    """A two-leg entry (P2-8) as one position: both legs decided together under the one lock (limits.decide_pair), the
    same share of each or neither, and both reservations written before the lock is let go, so no other strategy's
    check lands between them (Advisor M1). Each leg's decision is recorded, under its own row's name when `strategy`
    names the two legs. Any failure fails closed for both legs, as check_order does for one."""
    names = (strategy, strategy) if isinstance(strategy, str) else tuple(strategy)
    strategy = names[0]
    lock_failed, version, cv = False, getattr(profile, "version", None), None
    try:
        with ExitStack() as held:
            try:
                held.enter_context(ledger.lock())
            except Exception:
                lock_failed = True
                raise
            tests, cv = _tests(ledger, profile)
            block = _block(ledger, now)
            if block is not None:
                why = f"Entry rejected: {block[1]}"
                decisions = (rejected(first.qty, block[0], why), rejected(second.qty, block[0], why))
            else:
                equity, held_now = ledger.positions()
                book = Book(equity, (*held_now, *(r.holding for r in ledger.reservations())))
                decisions = decide_pair(book, first, second, tests)
            out = []
            for name, intent, decision in zip(names, (first, second), decisions):
                rid = None
                if decision.approved_qty > 0:
                    q = decision.approved_qty
                    rid = ledger.reserve(name, Holding(intent.underlying, intent.side * q * intent.price,
                                                       q * intent.margin_per_unit, q * intent.risk_per_unit), now, q)
                out.append(Checked(decision, _record(ledger, name, now, intent, decision, tests.version, cv), rid))
            return out[0], out[1]
    except Exception as e:  # noqa: BLE001 - any failure here means neither leg is sent
        why = f"Entry rejected: the portfolio check couldn't run ({type(e).__name__}: {e}), so neither leg is sent"
        try:
            ledger.alert("portfolio_check_failed", f"{strategy}: {why}", now)
        except Exception:  # noqa: BLE001 - the refusal stands whether or not the alert lands
            pass
        out = []
        for name, intent in zip(names, (first, second)):
            qty = getattr(intent, "qty", Decimal(0))
            qty = qty if isinstance(qty, Decimal) else Decimal(0)
            try:
                _record(ledger, name, now, intent, Decision("error", Decimal(0), qty,
                                                            "lock_error" if lock_failed else None, why), version, cv)
            except Exception:  # noqa: BLE001 - as for the alert
                pass
            out.append(Checked(rejected(qty, None, why), None, None))
        return out[0], out[1]


def mark_book(ledger: Ledger, equity, now: datetime, profile: PortfolioProfile) -> str | None:
    """The supervisor's mark of the whole fund. Returns "halt" when it has just halted the portfolio (the caller
    flattens every position through the exit path and halts every strategy), "pause" when it has just paused it,
    else None. The day's start is the last mark before 00:00 UTC (risk.trading_day); the HWM and the halt's reference
    only rise, until a book reset (or, for the reference, a PM Resume) re-bases them. At 00:00 UTC the order is the old
    day's final mark, then the new day's start, then decisions (Advisor, 8 Oct): a mark at exactly 00:00 never pauses
    (its pause would end as it starts, with a second alert), and a decision from 00:00 on never sees the old day's
    pause, which does not carry over (F220-3)."""
    equity = money(equity, "equity")
    with ledger.lock():
        st = ledger.state()
        day = trading_day(now)
        if st.equity is None and st.carried_hwm_drawdown is not None:
            st = _restored(st, equity, day)
        if st.day != day:
            st = replace(st, day=day, day_start_equity=st.equity if st.equity else equity)
        st = replace(st, equity=equity, mark_ts=now, hwm=max(st.hwm or equity, equity),
                     halt_reference=None if st.halt_reference is None else max(st.halt_reference, equity))
        breach = book_breach(profile, equity, st.reference(), st.day_start_equity)
        acted = None
        if breach is not None and breach.action == "halt" and not st.halted:
            st, acted = replace(st, halted=breach.reason), "halt"
            ledger.alert("portfolio_halt", f"Portfolio halted: {breach.reason}", now)
        elif (breach is not None and breach.action == "pause" and not (st.paused_until and now < st.paused_until)
              and next_utc_midnight(now) > now):  # the mark at 00:00 is its day's last: a pause there blocks nothing
            st, acted = replace(st, paused_until=next_utc_midnight(now), paused=breach.reason), "pause"
            ledger.alert("portfolio_pause", f"Portfolio paused for the day: {breach.reason}", now)
        ledger.set_state(st)
        return acted


def _restored(st: PortfolioState, equity: Decimal, day) -> PortfolioState:
    """MUST-F286-H4 (Advisor 14 Oct 13:35 UK): the first mark of new capital in a book a withdrawal emptied keeps the
    drawdown and the day's change it had: the HWM at equity / (1 - d), the halt's reference likewise, and the day's
    start at equity / (1 + r) on the same UTC day, else the equity (a new day starts at it). Exact, one rounding each,
    as apply_flow; the carried figures are cleared in the same write (mark_book's one set_state)."""
    if equity < EMPTY:
        raise ValueError("an empty book's mark restores nothing: it is left unmarked (Supervisor.mark_portfolio)")
    if (bad := unrestorable(st)) is not None:
        raise ValueError(bad)
    with localcontext() as ctx:
        ctx.prec = 2 * PRECISION

        def up(d):
            return None if d is None else (equity / (1 - d)).quantize(QUANTUM, ROUND_HALF_EVEN)

        same_day = st.carried_day == day and st.carried_day_return is not None
        start = (equity / (1 + st.carried_day_return)).quantize(QUANTUM, ROUND_HALF_EVEN) if same_day else equity
        return replace(st, hwm=up(st.carried_hwm_drawdown), halt_reference=up(st.carried_reference_drawdown), day=day,
                       day_start_equity=start, carried_hwm_drawdown=None, carried_reference_drawdown=None,
                       carried_day_return=None, carried_day=None)


def unrestorable(st: PortfolioState) -> str | None:
    """Why the carried ratios can't be restored, or None (Advisor MUST, 14 Oct 14:45 UK): restored only while 1 - d > 0
    and 1 + r > 0, so never a division by 0. An emptying withdrawal always starts from a book above 0, so this never
    fires; if it does, the book stays unmarked, entries stay blocked as stale, and the supervisor alerts once."""
    if st.equity is not None or st.carried_hwm_drawdown is None:
        return None
    d, q, r = st.carried_hwm_drawdown, st.carried_reference_drawdown, st.carried_day_return
    if 1 - d > 0 and (q is None or 1 - q > 0) and (r is None or 1 + r > 0):
        return None
    def pct(x):
        return "none" if x is None else f"{x:.2%}"

    return (f"the drawdown kept from when the fund's book was emptied can't be restored (from the high-water mark "
            f"{pct(d)}, from the halt's reference {pct(q)}, the day's change {pct(r)}): the book stays unmarked and "
            "entries stay blocked until it is fixed")


def carried(st: PortfolioState) -> dict:
    """The ratios an emptying withdrawal carries (MUST-F286-H4), from the book before it, never the ~0 after (DA)."""
    if st.equity is None or st.equity <= 0 or not st.hwm:
        return {}
    with localcontext() as ctx:
        ctx.prec = 2 * PRECISION

        def down(ref):  # never below 0: a flow's rounding can leave the book a quantum over its HWM (hwm_range)
            return None if not ref else max(Decimal(0), (1 - st.equity / ref).quantize(QUANTUM, ROUND_HALF_EVEN))

        r = (None if not st.day_start_equity or st.day is None
             else (st.equity / st.day_start_equity - 1).quantize(QUANTUM, ROUND_HALF_EVEN))
        return {"carried_hwm_drawdown": down(st.hwm), "carried_reference_drawdown": down(st.halt_reference),
                "carried_day_return": r, "carried_day": st.day if r is not None else None}


def resume_after_halt(ledger: Ledger) -> PortfolioState:
    """The PM's Resume after a portfolio halt: clears it and re-bases the halt's reference to the book now, so it
    doesn't fire again at once; the next halt is the drawdown limit below this. A pause for the day stays until
    00:00 UTC. The true high-water mark is kept in the record."""
    with ledger.lock():
        st = ledger.state()
        st = replace(st, halted=None, halt_reference=st.equity)
        ledger.set_state(st)
        return st


def apply_flow(ledger: Ledger, amount, idle: Callable[[], str | None] | None = None,
               now: datetime | None = None, carry: bool = True) -> PortfolioState | None:
    """Capital added (> 0) or withdrawn (< 0), so a deposit isn't a gain and a withdrawal isn't a loss (Advisor 3).
    The book moves by it, and the HWM, the halt's reference and the day's start each scale by (book + amount) / book,
    so the drawdown and the day's change keep their percentages (Advisor MUST, 14 Oct 09:45 UK): a shift by the
    amount would dilute a drawdown on a deposit (90 against 100 is 10% down; 190 against 200, 5%) and deepen it on a
    withdrawal. Exact, to the column's 18 places. Refused (ValueError) on a book of 0 or less before, or one never
    marked: no ratio then, so the move fails closed; except a deposit into a book at 0 that never had a high-water
    mark (an empty fund, as a fresh stack marks it), which starts the book at the deposit.

    A withdrawal that empties the book (within EMPTY of 0 after it, either side: one that overshoots by under a cent
    empties it too, rather than counting as below nothing) leaves it unmarked, so the next mark starts it
    afresh, as a first mark does; only when idle() says nothing is open (no position, no Strategy that could trade:
    it returns why not, or None), no reservation is held and no day's pause is in force at `now` (a pause needs a
    mark, so the withdrawal waits for 00:00 UTC; one that has ended is cleared with the marks); otherwise, and below
    0, refused. A halt in force stays until the PM resumes
    it. Returns the state it emptied, for the caller's journal, else
    None (Advisor 14 Oct 10:00 UK). Emptied, it carries the drawdown and the day's change it had (carried) to the next
    capital's first mark, which restores them (MUST-F286-H4, Advisor 14 Oct 13:35 UK); carry False (a clean slate's
    withdrawal: a fresh book) carries nothing."""
    amount = money(amount, "amount")
    with ledger.lock():
        st = ledger.state()
        book = st.equity
        if book is not None and book == 0 and not st.hwm and amount > 0:
            # An empty fund's first capital (a fresh stack marks 0 before any Strategy is made): there was never a
            # high-water mark, so no drawdown to keep, and the book starts at the deposit. A book at 0 below a HWM
            # is a real loss and is still refused below.
            ledger.set_state(replace(st, equity=amount, hwm=amount, halt_reference=None, day_start_equity=amount))
            return None
        if book is None or book <= 0:
            raise ValueError(f"a flow of {amount:,.2f} on a book of {'none' if book is None else f'{book:,.2f}'} "
                             "can't keep the book's percentages: refused")
        if abs(book + amount) < EMPTY:
            busy = "no check of what is open was given" if idle is None else idle()
            if busy is None and ledger.reservations():
                busy = "an order still holds room in the book"
            now = now or datetime.now(timezone.utc)
            if busy is None and st.paused_until is not None and now < st.paused_until:
                # a pause follows a mark (portfolio_state_marked), so the day's pause keeps the book marked: emptied
                # from 00:00 UTC, when it ends (Advisor 14 Oct 10:10 UK; nothing can trade an empty book meanwhile).
                # Said to the PM as it is (the dashboard shows the refusal), with when to try again in UK time.
                from zoneinfo import ZoneInfo

                uk = st.paused_until.astimezone(ZoneInfo("Europe/London"))
                raise ValueError(f"Not done: this would leave the fund with no capital while it is paused for the day "
                                 f"({st.paused}), and the pause has to stay in force until it ends. Nothing is lost "
                                 f"by waiting. Please try again after {uk:%H:%M} UK ({st.paused_until:%H:%M} UTC).")
            if busy is not None:  # said to the PM as it is, as the pause's is (F286-1)
                raise ValueError(f"Not done: this would leave the fund with no capital while {busy}. Nothing is lost "
                                 f"by waiting: try again once that has changed.")
            # a pause that has ended goes with the marks: an unmarked row can't say paused (portfolio_state_marked)
            # MUST-F286-H4: its ratios carried to the next capital's first mark, unless a clean slate emptied it (carry)
            ledger.set_state(replace(st, equity=None, mark_ts=None, hwm=None, halt_reference=None, day=None,
                                     day_start_equity=None, paused_until=None, paused=None,
                                     **(carried(st) if carry else {})))
            return st
        if book + amount <= 0:
            raise ValueError(f"a flow of {amount:,.2f} would take the book of {book:,.2f} below nothing: refused")
        with localcontext() as ctx:
            ctx.prec = 2 * PRECISION  # each product is exact before its one rounding to the column's scale

            def scaled(x):
                return None if x is None else (x * (book + amount) / book).quantize(QUANTUM, ROUND_HALF_EVEN)

            ledger.set_state(replace(st, equity=book + amount, hwm=scaled(st.hwm),
                                     halt_reference=scaled(st.halt_reference),
                                     day_start_equity=scaled(st.day_start_equity)))
        return None


def sweep(ledger: Ledger, now: datetime, is_live: Callable[[str], bool], cancel: Callable[[str], None] | None,
          ttl: timedelta = RESERVATION_TTL, is_acked: Callable[[str], bool] | None = None) -> list[Reservation]:
    """Orphans only (Advisor MUST FIX, 17:20 UK; refined 20:25 UK): a reservation lives as long as its order. Past
    `ttl`, one whose order is still live has that order cancelled and keeps its reservation until the cancel is
    confirmed: the order's own cancel releases it, or, if the order filled first, its fill converts the reservation
    into the position once (reduce, in the same transaction as the fill). An unconfirmed cancel is never released
    blind: every CANCEL_CONFIRM it is sent again with an alert, the reservation kept. A reservation with no order, or
    whose order is gone, is released as "ttl" with an alert: its process died before releasing it. Returns those. is_acked (paper): the TTL covers send to venue ack only, so a reservation whose order the venue has
    acked is skipped whatever its age; it lives with its order, released on a confirmed cancel or converted on a fill
    (Advisor 8 Oct 06:10 UK). cancel None (paper until ORPHAN-GUARD og2): the sweep can't cancel, so it sends nothing
    and says so; the order's own process cancels it at its next bar close, and the reservation is kept (CR W2-2)."""
    gone = []
    with ledger.lock():
        for r in ledger.reservations():
            if now - r.at <= ttl or (is_acked is not None and r.order_id is not None and is_acked(r.order_id)):
                continue
            if r.order_id is not None and is_live(r.order_id):
                if cancel is None:  # nothing to send: told once, kept until its process's cancel or a fill lands
                    if r.cancel_sent is None:
                        ledger.cancel_sent(r.id, now)
                        ledger.alert("reservation_unacked", f"{r.strategy}: order {r.order_id} rested past its "
                                     "reservation's TTL unacked. The supervisor can't cancel it yet (ORPHAN-GUARD), so "
                                     "its own process cancels it at its next bar close; nothing was sent, and the "
                                     "reservation holds until that cancel or a fill lands", now)
                    continue
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
