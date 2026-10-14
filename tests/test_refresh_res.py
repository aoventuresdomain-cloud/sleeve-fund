"""F-REFRESH-RES (HoE, Drop A SHOULD; DA with PE1): once a paper Strategy's process has stopped, an order the journal
still shows open can't fill (the simulated venue went with the process), so Store.close_dead_orders closes it and
releases the room it held in the book, and a reservation holding no order too. Nothing is left to keep the
Strategy from settling flat (withdrawal_blocker) or the book from being empty, so the PM's refresh reaches its deposit
instead of waiting at exit 2. A live account's orders outlive the process and are never touched. Both engines."""

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, select, text

from sleeve_fund.store import Store, gate_decisions_t, gate_reservations_t, make_engine, metadata, orders_t, sleeves_t

T = datetime(2026, 10, 15, 1, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/res.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    yield Store(engine=eng)
    metadata.drop_all(eng)


def _reserve(store, sleeve, seq, order_id=None):
    with store.engine.begin() as c:
        sid = c.execute(select(sleeves_t.c.id).where(sleeves_t.c.name == sleeve)).scalar()
        did = c.execute(insert(gate_decisions_t).values(
            seq=seq, sleeve_id=sid, bar_ts=T, intent_id=f"i{seq}", intent="entry", underlying="BTC", profile_version=1,
            compliance_version=1, outcome="approved", requested_qty=1, approved_qty=1, price=100, stage="submit",
            decided_at=T, order_id=None)).inserted_primary_key[0]
        c.execute(insert(gate_reservations_t).values(
            decision_id=did, sleeve_id=sid, underlying="BTC", remaining_qty=1, notional=100, margin=100, open_risk=5,
            created_at=T, expires_at=T + timedelta(minutes=10)))
    if order_id is not None:
        store.record_order(sleeve, order_id=order_id, side="BUY", qty=1.0, intent="entry", reason="r",
                           order_type="LIMIT", reservation=did)
        store.update_order(order_id, status="accepted")
    return did


def _make(store, name="s"):
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0)
    store.set_desired_state(name, "stopped")


def test_a_stopped_paper_strategys_open_order_and_its_room_are_closed_so_it_settles_flat(store):
    _make(store)
    _reserve(store, "s", 1, order_id="o1")  # an entry left resting ('accepted') when its process stopped
    _reserve(store, "s", 2)  # room held for an order never sent
    led = store._gate_ledger()
    assert len(led.reservations()) == 2
    assert store.withdrawal_blocker("s") == "an order of it is still open"
    assert store.close_dead_orders("s", "PM refresh 2026-10-15-reset") == ["o1"]
    with store.engine.connect() as c:
        row = c.execute(select(orders_t.c.status, orders_t.c.message).where(orders_t.c.order_id == "o1")).one()
        released = c.execute(select(gate_reservations_t.c.release_reason)).scalars().all()
    assert row.status == "canceled" and row.message.startswith("closed: PM refresh")
    assert led.reservations() == [] and released == ["cancel", "cancel"]
    assert store.withdrawal_blocker("s") is None
    assert [e["kind"] for e in store.events("s")][0] == "orders_closed_stopped"
    assert store.close_dead_orders("s", "again") == []  # nothing left: no second event
    assert [e["kind"] for e in store.events("s")].count("orders_closed_stopped") == 1


def test_another_strategys_orders_and_room_are_untouched(store):
    """CR F-RR-1: scoped to the Strategy, for its orders and for room held with no order."""
    _make(store, "s")
    _make(store, "t")
    _reserve(store, "t", 1, order_id="t1")
    _reserve(store, "t", 2)
    _reserve(store, "s", 3)
    store.close_dead_orders("s", "refresh")
    with store.engine.connect() as c:
        assert c.execute(select(orders_t.c.status).where(orders_t.c.order_id == "t1")).scalar() == "accepted"
    assert sorted(r.strategy for r in store._gate_ledger().reservations()) == ["t", "t"]


def test_room_an_order_left_behind_is_released_too(store):
    """CR F-RR-2: a reservation still active though its order has finished (left behind) holds room no order can use:
    released as well, never left counting against the tests."""
    _make(store)
    did = _reserve(store, "s", 1, order_id="o1")
    with store.engine.begin() as c:  # the order finished without settling its reservation (the case to clean up)
        c.execute(orders_t.update().where(orders_t.c.order_id == "o1").values(status="filled"))
    assert [r.id for r in store._gate_ledger().reservations()] == [did]
    assert store.close_dead_orders("s", "refresh") == []
    assert store._gate_ledger().reservations() == []


def test_a_live_accounts_orders_are_never_closed(store):
    _make(store)
    store.create_account("live-a", "live", venue="binance")
    store.assign_account("s", "live-a")
    with pytest.raises(ValueError, match="live account"):
        store.close_dead_orders("s", "refresh")
