"""F-RRP-5 (CR; HoE 15 Oct 02:27 UK; DA): the supervisor's poll for put-away Strategies with orders left open
(Supervisor.close_dead_orders) reads the book once, not once per archived run. Every reset puts a run away, so the
archive only grows; each poll looked each one up (account, lifecycle) though almost none has anything open. Now one
query names the runs with an open order or active room, and only those are looked at. Both engines."""

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, insert, select, text
from test_refresh_res import _reserve

from sleeve_fund.store import Store, make_engine, metadata, orders_t, sleeve_archive_t
from sleeve_fund.supervisor import Proc, Supervisor

T = datetime(2026, 10, 15, 2, 30, tzinfo=UTC)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/scan.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    yield Store(engine=eng)
    metadata.drop_all(eng)


def _put_away(store, name, withdrawn=None):
    """A run put away as a reset leaves it (archived, its equity still in the book) or one withdrawn."""
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, desired_state="stopped")
    with store.engine.begin() as c:
        c.execute(insert(sleeve_archive_t).values(sleeve=name, archived_at=T, withdrawn=withdrawn))


def _statements(store, sup) -> int:
    seen = []
    listen = lambda *a, **k: seen.append(1)
    event.listen(store.engine, "before_cursor_execute", listen)
    try:
        sup.close_dead_orders()
    finally:
        event.remove(store.engine, "before_cursor_execute", listen)
    return len(seen)


def test_a_poll_with_nothing_left_open_costs_the_same_however_many_runs_are_put_away(store):
    sup = Supervisor(store, python="true")
    for i in range(3):
        _put_away(store, f"r{i}")
    few = _statements(store, sup)
    for i in range(3, 40):
        _put_away(store, f"r{i}")
    assert _statements(store, sup) == few <= 2


def test_among_many_put_away_runs_only_one_left_open_is_closed_and_a_withdrawn_one_is_skipped(store):
    for i in range(20):
        _put_away(store, f"r{i}")
    _reserve(store, "r7", 1, order_id="o7")  # an entry left 'accepted' when its process stopped
    _reserve(store, "r12", 2)  # room held for an order never sent
    store.record_order("r3", order_id="o3", side="SELL", qty=1.0, intent="exit", reason="r", order_type="LIMIT")
    store.update_order("o3", status="accepted")  # an order open that holds no room
    _put_away(store, "gone", withdrawn=0)
    _reserve(store, "gone", 3, order_id="og")
    sup = Supervisor(store, python="true")
    sup.procs["r7"] = Proc()
    sup.close_dead_orders()
    with store.engine.connect() as c:
        status = dict(c.execute(select(orders_t.c.order_id, orders_t.c.status)).all())
    assert status == {"o3": "canceled", "o7": "canceled", "og": "accepted"}
    assert sorted(r.strategy for r in store._gate_ledger().reservations()) == ["gone"]
    assert _statements(store, sup) <= 2  # what is left is the withdrawn one's, never looked at
