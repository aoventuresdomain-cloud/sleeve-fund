"""F-WED1-3 (Advisor 14 Oct 03:35 UK; DA rule A): an open position whose orders went to another venue than the pin
(a pre-0020 order backfilled to the default venue, or the default venue changed under an unpinned Strategy) keeps
its own venue. The journal writes its exits there, refuses an opening order, and is unchanged otherwise."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from sleeve_fund.store import Store, make_engine, metadata

MISMATCH = "on another venue than its pin"


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/j.db")
    _clean(eng)
    yield Store(engine=eng)
    _clean(eng)


def _held_elsewhere(store, name="s"):
    """Pinned to BINANCE, holding 1 from an entry whose row says KRAKEN (as a backfilled pre-0020 order would)."""
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE")
    store.record_order(name, order_id=f"{name}-e", side="BUY", qty=1.0, intent="entry", reason="r")
    store.record_fill(name, side="BUY", qty=1.0, price=100.0, fee=0.0, order_id=f"{name}-e", trade_id="t1")
    with store.engine.begin() as c:
        c.execute(text("UPDATE orders SET venue = 'KRAKEN' WHERE order_id = :o"), {"o": f"{name}-e"})


def _venue(store, order_id):
    with store.engine.connect() as c:
        return c.execute(text("SELECT venue FROM orders WHERE order_id = :o"), {"o": order_id}).scalar()


def _count(store, table, order_id):
    with store.engine.connect() as c:
        return c.execute(text(f"SELECT count(*) FROM {table} WHERE order_id = :o"), {"o": order_id}).scalar()


@pytest.mark.parametrize("intent", ["stop_loss", "exit", "take_profit", "risk_halt", "pm_flatten"])
def test_a_holding_mismatch_journals_its_exits_on_the_positions_venue(store, intent):
    _held_elsewhere(store)
    assert store.position_venue("s") == "KRAKEN"
    store.record_order("s", order_id="s-x", side="SELL", qty=1.0, intent=intent, reason="r")
    assert _venue(store, "s-x") == "KRAKEN"


@pytest.mark.parametrize("intent", ["entry", "rebalance"])
def test_a_holding_mismatch_refuses_an_opening_order_and_writes_nothing(store, intent):
    _held_elsewhere(store)
    with pytest.raises(ValueError, match=MISMATCH):
        store.record_order("s", order_id="s-n", side="BUY", qty=1.0, intent=intent, reason="r",
                           timing={"decided": 1_760_000_000_000_000_000})
    assert _count(store, "orders", "s-n") == 0 and _count(store, "order_timings", "s-n") == 0


def test_a_flat_mismatch_journals_on_the_pin(store):
    _held_elsewhere(store)
    store.record_order("s", order_id="s-x", side="SELL", qty=1.0, intent="exit", reason="r")
    store.record_fill("s", side="SELL", qty=1.0, price=101.0, fee=0.0, order_id="s-x", trade_id="t2")
    assert store.position_venue("s") is None
    store.record_order("s", order_id="s-n", side="BUY", qty=1.0, intent="entry", reason="r")
    assert _venue(store, "s-n") == "BINANCE"


def test_holding_on_the_pin_is_unchanged(store):
    store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE")
    assert store.position_venue("s") is None
    store.record_order("s", order_id="s-e", side="BUY", qty=1.0, intent="entry", reason="r")
    store.record_fill("s", side="BUY", qty=1.0, price=100.0, fee=0.0, order_id="s-e", trade_id="t1")
    assert store.position_venue("s") == "BINANCE"
    store.record_order("s", order_id="s-a", side="BUY", qty=1.0, intent="entry", reason="r")
    store.record_order("s", order_id="s-x", side="SELL", qty=2.0, intent="stop_loss", reason="r")
    assert _venue(store, "s-a") == _venue(store, "s-x") == "BINANCE"
