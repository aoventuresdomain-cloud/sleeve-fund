"""T-FIX5 (#286, HoQA's c3c engine probe on T-s3b): a flat Strategy the supervisor's poll withdraws leaves the fund's
book in ONE transaction on every store: the archive row's `withdrawn`, the state's flow and the portfolio_flow event
persist together or not at all. On a store sharing one connection (Store.in_memory, a StaticPool), a read made inside
the gate's lock after its writes reset that connection and rolled them back, while the event written after survived."""

import os
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select, text

from sleeve_fund import supervisor
from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.store import Store, events_t, qty_words

T0 = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
PX = 60_000.0
PG = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(params=["memory", "file", "postgres"])
def new_store(request, tmp_path):
    if request.param == "memory":
        yield Store.in_memory(portfolio_gate=True)
    elif request.param == "file":
        yield Store(f"sqlite:///{tmp_path}/t.db", portfolio_gate=True)
    else:
        if not PG or not PG.startswith("postgresql"):
            pytest.skip("Postgres only with TEST_DATABASE_URL")
        schema = f"tfix5_{uuid.uuid4().hex[:10]}"
        admin = create_engine(PG)
        with admin.begin() as c:
            c.execute(text(f"CREATE SCHEMA {schema}"))
        eng = create_engine(PG, connect_args={"options": f"-csearch_path={schema}"})
        yield Store(engine=eng, portfolio_gate=True)
        eng.dispose()
        with admin.begin() as c:
            c.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        admin.dispose()


def _mark(store, name, at):
    book = store.journal_book(name, store.sleeve(name).starting_balance)
    qty, cash = float(book["qty"]), float(book["cash"])
    store.record_equity(name, equity=cash + qty * PX, cash=cash, qty=qty, price=PX, benchmark=10_000.0, ts=at)


def _strategy(store, name, inst, qty=0.0, archived=False):
    store.create_sleeve(name=name, strategy="ping_pong", instrument=inst, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE", params={"market": "perp"})
    if qty:
        store.record_order(name, order_id=f"e-{name}", side="BUY", qty=qty, intent="entry", reason="t", ts=T0)
        store.book_fill(name, side="BUY", qty=qty, price=PX, fee=0.0, order_id=f"e-{name}", trade_id=f"t-{name}",
                        ts=T0)
    _mark(store, name, T0 + timedelta(minutes=1))
    if archived:
        store.set_desired_state(name, "stopped")
        store.set_status(name, "stopped", "t: stopped")
        store.archive(name)


def _close(store, name, at=T0 + timedelta(minutes=2)):
    store.record_order(name, order_id=f"x-{name}", side="SELL", qty=0.1, intent="exit", reason="t", ts=at)
    store.book_fill(name, side="SELL", qty=0.1, price=PX, fee=0.0, order_id=f"x-{name}", trade_id=f"x-{name}", ts=at)
    _mark(store, name, at + timedelta(seconds=30))


def _c3c(store):
    """HoQA's probe: 'old' flat and put away by the clean slate; 'held' archived while holding, kept by it, then
    traded flat; the poll after that withdraws it."""
    _strategy(store, "fund", "BTC/USDT", qty=0.1)
    _strategy(store, "entry", "SOL/USDT")
    _strategy(store, "old", "ETH/USDT", qty=0.1)
    _close(store, "old")
    _strategy(store, "held", "XRP/USDT", qty=0.1, archived=True)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write('[[clear]]\nid = "t5"\nreason = "T-FIX5 clean slate"\nkeep = ["entry", "fund", "held"]\n')
    try:
        supervisor.clear(store, fh.name)
    finally:
        os.unlink(fh.name)
    t1, t2 = T0 + timedelta(minutes=1, seconds=30), T0 + timedelta(minutes=3)
    _mark(store, "fund", t1)
    _mark(store, "held", t1)
    sup = supervisor.Supervisor(store)
    sup.mark_portfolio(t1)
    _close(store, "held")
    _mark(store, "fund", t2)
    return sup, t2


def _flows(store, sleeve):
    with store.engine.connect() as c:
        return [r.message for r in c.execute(select(events_t).where(events_t.c.kind == "portfolio_flow"))
                if r.message.startswith(f"{sleeve} put away")]


def test_the_poll_withdraws_a_flat_holder_the_clean_slate_kept_on_every_store(new_store):
    store = new_store
    sup, t2 = _c3c(store)
    sup.mark_portfolio(t2)
    st = DbLedger(store, lambda: None).state()
    assert set(store.withdrawn()) == {"old", "held"} and store.withdrawal_pending() == []
    assert supervisor.fund_equity(store) == Decimal(20_000)
    assert (st.equity, st.hwm, st.reference(), st.day_start_equity) == (Decimal(20_000),) * 4
    assert len(_flows(store, "held")) == 1


def test_a_failure_midway_leaves_none_of_the_three_writes(new_store, monkeypatch):
    """Killed after the archive row is written and before the flow lands: no withdrawn row, no state flow, no event."""
    from sleeve_fund.portfolio import gate

    store = new_store
    _, t2 = _c3c(store)
    before = DbLedger(store, lambda: None).state()

    def boom(*a, **k):
        raise RuntimeError("killed midway")

    monkeypatch.setattr(gate, "apply_flow", boom)
    with pytest.raises(RuntimeError, match="killed midway"), store._put_away("held", t2, "flat after it was put "
                                                                                 "away", polled=True):
        pass
    st = DbLedger(store, lambda: None).state()
    assert "held" not in store.withdrawn() and store.withdrawal_pending() == ["held"]
    assert (st.equity, st.hwm, st.reference()) == (before.equity, before.hwm, before.reference())
    assert _flows(store, "held") == []


def test_the_flow_event_names_the_dust_that_leaves_with_a_withdrawn_holder(new_store):
    """F-T-3: T-FIX5 reads the dust before the move writes; its words carry it (and none when there is none)."""
    store = new_store
    sup, t2 = _c3c(store)
    store.record_order("held", order_id="d-held", side="BUY", qty=1e-9, intent="entry", reason="t", ts=t2)
    store.book_fill("held", side="BUY", qty=1e-9, price=PX, fee=0.0, order_id="d-held", trade_id="d-held", ts=t2)
    _mark(store, "held", t2 + timedelta(seconds=10))
    sup.mark_portfolio(t2 + timedelta(seconds=20))
    (flow,) = _flows(store, "held")
    assert ", with dust of 0.000000001 no order can close, which leaves with it" in flow, flow
    assert "held" in store.withdrawn()


@pytest.mark.parametrize("qty, words", [(Decimal("1.000000000E-9"), "0.000000001"), (Decimal("1E-9"), "0.000000001"),
                                        (Decimal(100), "100"), (Decimal("100.000000000000000000"), "100"),
                                        (Decimal("0.01500002"), "0.01500002")])
def test_a_quantity_reads_in_plain_digits_whatever_scale_the_engine_hands_back(qty, words):
    """TEC-DUST-FMT: Postgres returns a NUMERIC at its column's scale, SQLite without; never an exponent (1E+2)."""
    assert qty_words(qty) == words
