"""The portfolio gate on the journal (sleeve_fund.gate_ledger.DbLedger, v2 P2-2): PE2's gate cells
(tests/test_portfolio_gate.py) run unchanged against it, on SQLite and, with TEST_DATABASE_URL, on Postgres; then
what only the journal has: seq taken under the lock across processes, releases inside the fill's transaction, the
error rows, the tags a row needs, and the state row before the first mark.

CellLedger is the harness, not the product: it gives each cell its own database (a schema on Postgres), the
in-memory ledger's equity and held as the positions, and the views the cells read (reserved, released, records,
alerts) from the tables. It also tags a plain Intent (the cells' own) and writes the order row an attach names, as
paper will (P2-1b)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import fields
from datetime import timedelta
from decimal import Decimal as D
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, insert, select, text
from sqlalchemy.exc import IntegrityError

import test_portfolio_gate as cells
from sleeve_fund.gate_ledger import DbLedger, TaggedIntent
from sleeve_fund.money import money
from sleeve_fund.portfolio.gate import (
    MemoryLedger,
    PortfolioState,
    Record,
    apply_flow,
    check_order,
    entry_block,
    mark_book,
    resume_after_halt,
)
from sleeve_fund.portfolio.limits import Decision, Holding, Intent
from sleeve_fund.risk import PORTFOLIO
from sleeve_fund.store import Store, _aware, events_t, gate_decisions_t, gate_reservations_t, portfolio_state_t
from test_portfolio_gate import *  # noqa: F401,F403 - the cells, collected here against the DB ledger
from test_portfolio_gate import PX, T0, _buy

PG = os.environ.get("TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[1]
_SCHEMAS: list[str] = []
_ENGINES: list = []
STRATEGIES = ("a", "b", "c", "p1", "p2")


def _engine(schema: str | None = None):
    if not PG:
        return None
    args = {"options": f"-csearch_path={schema}"} if schema else {}
    return create_engine(PG, connect_args=args)


def _store() -> Store:
    if not PG:
        return Store.in_memory(portfolio_gate=True)  # a paper journal: the gate is in force
    name = f"gl_{uuid.uuid4().hex[:12]}"
    with _engine().begin() as c:
        c.execute(text(f"CREATE SCHEMA {name}"))
    _SCHEMAS.append(name)
    eng = _engine(name)
    _ENGINES.append(eng)
    return Store(engine=eng, portfolio_gate=True)


@pytest.fixture(scope="module", autouse=True)
def _drop_schemas():
    yield
    for eng in _ENGINES:
        eng.dispose()
    if PG and _SCHEMAS:
        admin = _engine()
        for s in _SCHEMAS:  # one schema per transaction: dropping them all at once runs out of lock slots
            with admin.begin() as c:
                c.execute(text(f"DROP SCHEMA IF EXISTS {s} CASCADE"))
        admin.dispose()


def _strategies(store: Store) -> None:
    for name in STRATEGIES:
        store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                            starting_balance=1_000)


def tagged(intent: Intent, at, **kw) -> TaggedIntent:
    return TaggedIntent(**{f.name: getattr(intent, f.name) for f in fields(Intent)},
                        **{"intent_id": uuid.uuid4().hex, "bar_ts": at, **kw})


class CellLedger(DbLedger):
    """The harness: MemoryLedger's constructor and the views the cells read, over a DbLedger on its own database."""

    def __init__(self, equity=D(0), held=None, store: Store | None = None):
        self.equity, self.held = equity, list(held or [])
        if store is None:
            store = _store()
            _strategies(store)
        super().__init__(store, lambda: (money(self.equity, "equity"), tuple(self.held)))

    def _tags(self, strategy, at, intent):
        return super()._tags(strategy, at, intent if isinstance(intent, TaggedIntent) else tagged(intent, at))

    def attach_order(self, reservation_id, order_id):
        r = next(r for r in self.reservations() if r.id == reservation_id)
        self.store.record_order(r.strategy, order_id=order_id, side="BUY", qty=r.qty, intent="entry",
                                reason="gate cell", ts=r.at)
        super().attach_order(reservation_id, order_id)

    @property
    def reserved(self):
        return {r.id: r for r in self.reservations()}

    @property
    def released(self):
        r = gate_reservations_t
        with self.engine.connect() as c:
            return [(x.decision_id, x.release_reason) for x in c.execute(
                select(r).where(r.c.released_at.is_not(None)).order_by(r.c.released_at, r.c.decision_id))]

    @property
    def records(self):
        d = gate_decisions_t
        with self.engine.connect() as c:
            names = dict(c.execute(text("SELECT id, name FROM sleeves")).all())
            return [Record(x.seq, names[x.sleeve_id], x.decided_at, None,
                           Decision(x.outcome, money(x.approved_qty), money(x.requested_qty), x.limit_hit, ""),
                           x.profile_version) for x in c.execute(select(d).order_by(d.c.seq))]

    @property
    def alerts(self):
        e = events_t
        with self.engine.connect() as c:
            return [(x.kind, x.message, x.ts) for x in c.execute(select(e).where(e.c.sleeve.is_(None))
                                                                   .order_by(e.c.id))]


@pytest.fixture(autouse=True)
def _cells_on_the_journal(monkeypatch):
    monkeypatch.setattr(cells, "MemoryLedger", CellLedger)


# --- what only the journal has ------------------------------------------------------------------------------------


def _marked(equity="20000", at=T0) -> CellLedger:
    ledger = CellLedger(equity=D(equity))
    mark_book(ledger, D(equity), at, PORTFOLIO)
    return ledger


def test_a_decision_row_carries_its_tags_figures_and_reservation():
    ledger = _marked()
    ledger.held = [Holding("BTC", D(3_000), D(600), D(15))]
    it = tagged(_buy("0.1"), T0, kind="add", regime_weight=D("0.75"), regime_state="trend")
    c = check_order(ledger, "a", it, PORTFOLIO, T0)
    assert (c.decision.outcome, c.decision.approved_qty, c.seq) == ("approved", D("0.1"), 1)
    with ledger.engine.connect() as conn:
        row = conn.execute(select(gate_decisions_t)).one()
        res = conn.execute(select(gate_reservations_t)).one()
    assert (row.id, row.intent_id, row.intent, row.stage, row.underlying) == (c.reservation, it.intent_id, "add",
                                                                             "submit", "BTC")
    assert (row.regime_weight, row.regime_state) == (D("0.75"), "trend")
    assert (row.book_equity, row.gross, row.net_underlying, row.margin_used, row.open_risk) == (
        D(20_000), D(3_000), D(3_000), D(600), D(15))  # the book before this entry
    assert (res.decision_id, res.remaining_qty, res.notional, res.margin) == (c.reservation, D("0.1"), D(6_000),
                                                                              D(1_200))
    assert res.expires_at - res.created_at == timedelta(minutes=10)


def test_a_short_reserves_an_unsigned_quantity_and_a_signed_notional():
    ledger = _marked()
    c = check_order(ledger, "a", _buy("0.1", side=-1), PORTFOLIO, T0)
    r = ledger.reserved[c.reservation]
    assert (r.qty, r.holding.notional) == (D("0.1"), D(-6_000))
    ledger.reduce(c.reservation, D("-0.04"))  # a sell's fill, signed or not, reduces what is left
    assert ledger.reserved[c.reservation].qty == D("0.06")


def test_a_plain_intent_is_refused_by_the_journal_so_the_entry_is_too():
    ledger = _marked()
    plain = DbLedger(ledger.store, ledger._positions)  # the product, without the harness's tagging
    c = check_order(plain, "a", _buy("0.1"), PORTFOLIO, T0)
    assert (c.decision.outcome, c.reservation, c.seq) == ("rejected", None, None)
    assert "TaggedIntent" in c.decision.reason and not ledger.reserved and ledger.records == []


def test_a_failed_check_journals_an_error_row_in_its_own_transaction():
    ledger = _marked()

    class Broken(CellLedger):
        def positions(self):
            raise ConnectionError("database gone")

    broken = Broken(D(20_000), store=ledger.store)
    c = check_order(broken, "a", _buy("0.1"), PORTFOLIO, T0)
    assert (c.decision.outcome, c.seq) == ("rejected", None)
    (rec,) = ledger.records
    assert (rec.decision.outcome, rec.decision.limit_hit, rec.decision.approved_qty) == ("error", None, D(0))
    assert not ledger.reserved


def test_a_lock_that_cant_be_taken_journals_lock_error():
    ledger = _marked()

    class NoLock(CellLedger):
        def lock(self):
            raise TimeoutError("lock wait timed out")

    c = check_order(NoLock(D(20_000), store=ledger.store), "a", _buy("0.1"), PORTFOLIO, T0)
    assert c.decision.outcome == "rejected" and "TimeoutError" in c.decision.reason
    assert [(r.decision.outcome, r.decision.limit_hit) for r in ledger.records] == [("error", "lock_error")]


def test_a_failure_inside_the_lock_leaves_no_reservation_behind():
    """The reservation and its decision commit together or not at all: a record that fails after reserve() rolls
    the reservation back with it."""
    ledger = _marked()

    class FailingRecord(CellLedger):
        def record(self, strategy, at, intent, decision, version):
            if decision.outcome != "error":
                super().record(strategy, at, intent, decision, version)
                raise RuntimeError("disk full")
            return super().record(strategy, at, intent, decision, version)

    c = check_order(FailingRecord(D(20_000), store=ledger.store), "a", _buy("0.1"), PORTFOLIO, T0)
    assert c.decision.outcome == "rejected" and not ledger.reserved
    assert [r.decision.outcome for r in ledger.records] == ["error"]


def test_a_release_joins_the_fills_transaction_and_rolls_back_with_it():
    """The fill's own write (an event row stands in for it here) and the release commit together or not at all."""
    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    fill = {"sleeve": "a", "ts": T0, "level": "info", "kind": "fill", "message": "BUY 0.1"}
    with pytest.raises(RuntimeError, match="venue said no"), ledger.engine.begin() as conn, ledger.using(conn):
        conn.execute(insert(events_t), [fill])
        ledger.reduce(a.reservation, D("0.1"))
        raise RuntimeError("venue said no")
    assert a.reservation in ledger.reserved  # the fill rolled back, so did the release
    with ledger.engine.begin() as conn, ledger.using(conn):
        conn.execute(insert(events_t), [fill])
        ledger.reduce(a.reservation, D("0.1"))
    assert not ledger.reserved and ledger.released == [(a.reservation, "fill")]


def _row(ledger, rid):
    with ledger.engine.connect() as c:
        return c.execute(select(gate_reservations_t).where(gate_reservations_t.c.decision_id == rid)).one()


def test_da_f219_1_a_fill_after_the_release_leaves_the_row_released_and_unchanged():
    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    ledger.release(a.reservation, "cancel")
    before = _row(ledger, a.reservation)
    ledger.reduce(a.reservation, D("0.04"))
    assert _row(ledger, a.reservation) == before and ledger.released == [(a.reservation, "cancel")]


def test_da_f219_1_two_fills_at_once_lose_neither_and_a_fill_waits_for_the_release():
    """Postgres: the second fill (or a fill behind the sweep's release) waits on the first one's row lock, then
    re-reads the row: both fills count, and a released row is never written to."""
    if not PG:
        pytest.skip("row locks are Postgres's; SQLite's writes are serialised whole")
    import threading

    def behind(first, then):
        """Run first() in an open transaction, start then() in a thread, check it waits, commit, let it finish."""
        done = threading.Event()
        with ledger.engine.begin() as conn, ledger.using(conn):
            first()
            t = threading.Thread(target=lambda: (then(), done.set()))
            t.start()
            assert not done.wait(0.5)  # blocked on the row
        t.join(5)
        assert done.is_set()

    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    behind(lambda: ledger.reduce(a.reservation, D("0.03")), lambda: ledger.reduce(a.reservation, D("0.03")))
    assert money(_row(ledger, a.reservation).remaining_qty) == D("0.04")
    behind(lambda: ledger.release(a.reservation, "cancel"), lambda: ledger.reduce(a.reservation, D("0.01")))
    row = _row(ledger, a.reservation)
    assert (money(row.remaining_qty), row.release_reason) == (D("0.04"), "cancel")


def test_hoqa_f219_q1_a_stalled_lock_holder_fails_the_check_closed_within_a_bounded_wait():
    """Postgres (HoQA G1): another process holds the gate's lock, stalled, not dead. The entry check fails closed as
    lock_error within the lock timeout, not after an unbounded wait, and leaves no reservation."""
    if not PG:
        pytest.skip("the advisory lock is Postgres's")
    import threading

    ledger = _marked()
    out = {}
    with ledger.engine.connect() as holder, holder.begin():
        holder.execute(text("SELECT pg_advisory_xact_lock(hashtext('portfolio_gate'))"))
        t = threading.Thread(target=lambda: out.setdefault("c", check_order(ledger, "a", _buy("0.1"), PORTFOLIO,
                                                                              T0)), daemon=True)
        started = time.monotonic()
        t.start()
        t.join(10)
        waited, blocked = time.monotonic() - started, t.is_alive()
    t.join(10)
    assert not blocked and waited < 5, f"check_order waited {waited:.1f}s on the gate lock"
    assert out["c"].decision.outcome == "rejected" and not ledger.reserved
    assert [(r.decision.outcome, r.decision.limit_hit) for r in ledger.records] == [("error", "lock_error")]


def test_cr_f219_4_an_entry_check_inside_a_fills_transaction_fails_closed_as_lock_error():
    ledger = _marked()
    with ledger.engine.begin() as conn, ledger.using(conn):
        c = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    assert c.decision.outcome == "rejected" and not ledger.reserved
    assert [(r.decision.outcome, r.decision.limit_hit) for r in ledger.records] == [("error", "lock_error")]


def test_cr_f219_5_a_failed_checks_error_row_and_alert_outlive_the_fill_that_rolls_back():
    """Postgres: an entry check inside a fill's transaction fails closed (F219-4); its error row and alert are audit,
    written in their own transaction, so they stay when the fill rolls back. SQLite shares one connection: they join."""
    if not PG:
        pytest.skip("SQLite's store shares one connection, so the audit row joins the fill's transaction there")
    ledger = _marked()
    fill = {"sleeve": "a", "ts": T0, "level": "info", "kind": "fill", "message": "BUY 0.1"}
    with pytest.raises(RuntimeError, match="venue said no"), ledger.engine.begin() as conn, ledger.using(conn):
        conn.execute(insert(events_t), [fill])
        c = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
        raise RuntimeError("venue said no")
    assert c.decision.outcome == "rejected" and not ledger.reserved
    assert [(r.decision.outcome, r.decision.limit_hit) for r in ledger.records] == [("error", "lock_error")]
    assert [k for k, *_ in ledger.alerts] == ["portfolio_check_failed"]
    with ledger.engine.connect() as conn:  # the fill itself rolled back
        assert conn.execute(select(func.count()).select_from(events_t).where(events_t.c.kind == "fill")).scalar() == 0


def test_cr_f219_3_a_halt_before_the_first_mark_is_written_and_blocks_entries():
    """DA 0012: a halt, the PM's included, is never refused for coming before the supervisor's first mark (refusing it
    would fail open). A daily pause always follows a mark, so one without a mark is still refused."""
    ledger = CellLedger(D(20_000))
    ledger.set_state(PortfolioState(halted="the PM halted the portfolio"))
    with ledger.engine.connect() as c:
        row = c.execute(select(portfolio_state_t)).one()
    assert (row.status, row.marked_at, row.book_equity, row.halt_reason) == ("halted", None, None,
                                                                             "the PM halted the portfolio")
    assert ledger.state().halted == "the PM halted the portfolio" and entry_block(ledger, T0)[0] == "halt"
    c = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)
    assert (c.decision.outcome, c.decision.limit_hit) == ("rejected", "halt") and not ledger.reserved
    with pytest.raises(IntegrityError):
        CellLedger(D(20_000)).set_state(PortfolioState(paused_until=T0 + timedelta(hours=1), paused="3% down"))


def test_cr_f219_2_a_replay_follows_seq_and_skips_error_rows():
    """An 'error' row takes its seq outside the lock and decided nothing: the replay reader leaves it out and keeps
    the first-come order of the rest."""
    ledger = _marked()
    a = check_order(ledger, "a", _buy("0.1"), PORTFOLIO, T0)

    class NoLock(CellLedger):
        def lock(self):
            raise TimeoutError("lock wait timed out")

    check_order(NoLock(D(20_000), store=ledger.store), "c", _buy("0.1"), PORTFOLIO, T0)
    b = check_order(ledger, "b", _buy("0.1"), PORTFOLIO, T0)
    assert [r.decision.outcome for r in ledger.records] == ["approved", "error", "trimmed"]
    rows = ledger.replayable()
    assert [(r.seq, r.strategy, r.decision.outcome, r.decision.approved_qty, r.decision.limit_hit) for r in rows] == [
        (a.seq, "a", "approved", D("0.1"), None), (b.seq, "b", "trimmed", D("0.066"), "net_instrument")]
    assert a.seq < b.seq and all(r.underlying == "BTC" and r.price == PX and r.stage == "submit" for r in rows)


def test_the_state_row_round_trips_and_exists_before_the_first_mark():
    ledger = CellLedger(D(20_000))
    entry_block(ledger, T0)  # never marked: one alert, remembered in the row
    with ledger.engine.connect() as c:
        row = c.execute(select(portfolio_state_t)).one()
    assert (row.status, row.marked_at, row.book_equity) == ("ok", None, None) and row.stale_told_at is not None
    mark_book(ledger, D(20_000), T0, PORTFOLIO)
    mark_book(ledger, D(19_400), T0 + timedelta(seconds=5), PORTFOLIO)  # 3% down: paused
    st = ledger.state()
    assert (st.equity, st.hwm, st.halt_reference, st.day_start_equity) == (D(19_400), D(20_000), None, D(20_000))
    assert st.paused and st.paused_until is not None and st.halted is None
    with ledger.engine.connect() as c:
        assert c.execute(select(portfolio_state_t.c.status)).scalar() == "paused"
    apply_flow(ledger, D(-400))
    assert (ledger.state().equity, ledger.state().hwm) == (D(19_000), D(19_600))


def test_two_processes_take_seq_first_come_and_never_double_book_headroom(tmp_path):
    """Done-when (2): two processes check orders on one book at once. Every decision has its own seq, 1..N with no
    gaps, and replaying them in seq order through the in-memory ledger gives the same decisions: each saw every
    earlier reservation, so the net room (10,000 of a 20,000 book) was never handed out twice."""
    if not PG:
        pytest.skip("cross-process locking is Postgres's (the advisory lock); SQLite's lock is per process")
    ledger = _marked()
    ledger.engine.dispose()
    schema = _SCHEMAS[-1]
    go = tmp_path / "go"
    script = textwrap.dedent(f"""
        import os, sys, time, uuid
        from dataclasses import fields
        from datetime import datetime, timezone
        from decimal import Decimal as D
        from sqlalchemy import create_engine
        from sleeve_fund.gate_ledger import DbLedger, TaggedIntent
        from sleeve_fund.portfolio.gate import check_order
        from sleeve_fund.portfolio.limits import Intent
        from sleeve_fund.risk import PORTFOLIO
        from sleeve_fund.store import Store
        name = sys.argv[1]
        eng = create_engine({PG!r}, connect_args={{"options": "-csearch_path={schema}"}})

        def slow_book():  # a slow read inside the lock: without the lock the two checks would overlap here
            time.sleep(0.05)
            return D(20000), ()

        led = DbLedger(Store(engine=eng), slow_book)
        T0 = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
        open(os.path.join({str(tmp_path)!r}, "ready-" + name), "w").close()
        while not os.path.exists({str(go)!r}):
            time.sleep(0.01)
        for i in range(12):
            px = D("60000")
            it = TaggedIntent("BTC", 1, D("0.01"), px, px / 5, px * D("0.005"), D("0.001"), D("0.001"),
                              intent_id=f"{{name}}-{{i}}", bar_ts=T0)
            c = check_order(led, name, it, PORTFOLIO, T0)
            assert c.seq is not None, c
    """)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    procs = [subprocess.Popen([sys.executable, "-c", script, n], env=env, cwd=ROOT) for n in ("p1", "p2")]
    deadline = time.monotonic() + 60
    while not all((tmp_path / f"ready-{n}").exists() for n in ("p1", "p2")):  # both loaded, then both go at once
        assert time.monotonic() < deadline and all(p.poll() is None for p in procs), "a process didn't start"
        time.sleep(0.01)
    go.touch()
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    rows = ledger.records
    assert [r.seq for r in rows] == list(range(1, 25)) and {r.strategy for r in rows} == {"p1", "p2"}
    order = [r.strategy for r in rows]
    assert order != sorted(order), "the two processes never interleaved; the pin proves nothing"
    replay = MemoryLedger(equity=D(20_000))
    mark_book(replay, D(20_000), T0, PORTFOLIO)
    for r in rows:
        want = check_order(replay, r.strategy, _buy("0.01"), PORTFOLIO, T0).decision
        assert (r.decision.outcome, r.decision.approved_qty) == (want.outcome, want.approved_qty), r.seq
    with ledger.engine.connect() as c:
        held = c.execute(select(func.sum(gate_reservations_t.c.notional))).scalar()
        seen = c.execute(select(gate_decisions_t.c.seq, gate_decisions_t.c.gross, gate_decisions_t.c.approved_qty)
                         .order_by(gate_decisions_t.c.seq)).all()
    before = D(0)
    for seq, gross, qty in seen:  # each decision saw exactly the reservations of every earlier seq
        assert gross == before, (seq, gross, before)
        before += qty * PX
    assert held <= D(10_000) and sum(r.decision.approved_qty for r in rows) * PX == held



def test_the_stale_marker_is_cleared_by_the_first_fresh_mark_and_an_unmarked_book_is_stale():
    """DA 0011: stale_told_at belongs to its spell, and a book with no mark is stale (fail closed)."""
    ledger = CellLedger(D(20_000))
    assert entry_block(ledger, T0)[0] == "portfolio_state_stale"  # marked_at NULL: blocked, alerted once

    def told():
        with ledger.engine.connect() as c:
            return c.execute(select(portfolio_state_t.c.stale_told_at)).scalar()

    assert told() is not None
    mark_book(ledger, D(20_000), T0, PORTFOLIO)
    assert told() is None  # the fresh mark ended the spell
    late = T0 + timedelta(seconds=61)
    assert entry_block(ledger, late)[0] == "portfolio_state_stale" and _aware(told()) == T0
    mark_book(ledger, D(20_000), late, PORTFOLIO)
    assert told() is None and [k for k, *_ in ledger.alerts] == ["portfolio_state_stale"] * 2


def test_a_resume_rebases_below_the_hwm_and_round_trips_and_a_flow_moves_both():
    """DA 0011 (c): halt_reference lives in reference_equity; after a PM Resume it is a value of its own."""
    ledger = _marked()
    mark_book(ledger, D(16_000), T0 + timedelta(seconds=5), PORTFOLIO)  # 20% down: halted
    resume_after_halt(ledger)
    st = ledger.state()
    assert (st.halted, st.halt_reference, st.hwm) == (None, D(16_000), D(20_000))
    apply_flow(ledger, D(1_000))
    st = ledger.state()
    assert (st.halt_reference, st.hwm, st.equity) == (D(17_000), D(21_000), D(17_000))
