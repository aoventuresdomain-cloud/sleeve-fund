"""F-RX-1: every closing order the venue or the risk engine refused is in the journal, in order, with why and when, so
the "N closes refused in a row" alert (F-RX-2, ops_health) can count them from the tail. A close that filled or is
working breaks the streak; entries and rebalances are not closes. An order the order-state check could not find, and
which it confirmed never filled, is "unfilled_unknown". A refused resize of a resting stop is journaled as an event and
never moves its order's status. CI also runs these against Postgres."""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.store import (CLOSING_INTENTS, FINISHED_ORDER_STATUSES, INTENTS, OPEN_ORDER_STATUSES, ORDER_STATUSES,
                               Store, make_engine, metadata)
from sleeve_fund.strategies.base import LongFlatStrategy

T = datetime(2026, 10, 15, 8, tzinfo=timezone.utc)


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/t.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    yield eng
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def store(engine):
    s = Store(engine=engine)
    for name in ("s", "other"):
        s.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT-PERP",
                        bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000)
    return s


def _order(store, oid, intent, minute, sleeve="s"):
    store.record_order(sleeve, order_id=oid, side="SELL", qty=0.1, intent=intent, reason="test",
                       ts=T + timedelta(minutes=minute))


def test_closing_intents_are_every_close_and_nothing_else():
    assert set(CLOSING_INTENTS) == {"exit", "stop_loss", "take_profit", "risk_halt", "risk_pause", "pm_flatten",
                                    "liquidation_cut", "wind_down", "wind_down_stop", "leg_unwind"}
    assert set(CLOSING_INTENTS) <= set(INTENTS)
    # the venue's own liquidation is not an order the Strategy sent; entries and rebalances open
    assert not {"entry", "rebalance", "liquidation"} & set(CLOSING_INTENTS)


def test_every_refused_close_is_listed_in_order_with_why_and_when(store):
    _order(store, "e1", "entry", 0)
    _order(store, "x1", "stop_loss", 1)
    store.update_order("x1", status="rejected", message="ReduceOnly Order is rejected")
    _order(store, "x2", "exit", 2)
    store.update_order("x2", status="denied", message="exceeds max notional")
    _order(store, "r1", "rebalance", 3)
    store.update_order("r1", status="rejected", message="margin")
    _order(store, "o1", "exit", 4, sleeve="other")
    store.update_order("o1", status="rejected", message="other Strategy")

    rows = store.closing_orders("s")
    assert [r["order_id"] for r in rows] == ["x1", "x2"]  # oldest first; no entry, rebalance or other Strategy
    assert [(r["intent"], r["status"], r["message"]) for r in rows] == [
        ("stop_loss", "rejected", "ReduceOnly Order is rejected"), ("exit", "denied", "exceeds max notional")]
    for r in rows:
        assert set(r) >= {"order_id", "intent", "status", "message", "ts", "updated_at", "instrument"}
        assert r["instrument"] == "BTC/USDT-PERP"
        assert r["updated_at"] is not None and r["ts"] is not None


def test_a_close_that_filled_or_is_working_is_listed_so_it_breaks_the_streak(store):
    _order(store, "x1", "stop_loss", 1)
    store.update_order("x1", status="rejected", message="no")
    _order(store, "x2", "stop_loss", 2)
    store.update_order("x2", fill_qty=0.1, fill_px=60_000.0)
    _order(store, "x3", "take_profit", 3)
    store.update_order("x3", status="accepted")
    assert [(r["order_id"], r["status"]) for r in store.closing_orders("s")] == [
        ("x1", "rejected"), ("x2", "filled"), ("x3", "accepted")]


def test_since_drops_older_closes(store):
    _order(store, "x1", "exit", 1)
    _order(store, "x2", "exit", 10)
    assert [r["order_id"] for r in store.closing_orders("s", since=T + timedelta(minutes=5))] == ["x2"]


def test_unfilled_unknown_is_a_status_that_fits_and_is_neither_open_nor_finished(store):
    assert "unfilled_unknown" in ORDER_STATUSES and len("unfilled_unknown") <= 16
    # not open: nothing is working at the venue as far as the journal knows; not finished: the venue's own answer
    # (rejected, cancelled, filled) still lands on it when it comes
    assert "unfilled_unknown" not in OPEN_ORDER_STATUSES and "unfilled_unknown" not in FINISHED_ORDER_STATUSES
    _order(store, "x1", "stop_loss", 1)
    store.update_order("x1", status="unfilled_unknown", message="not found at the venue; nothing filled")
    assert [(r["status"], r["message"]) for r in store.closing_orders("s")] == [
        ("unfilled_unknown", "not found at the venue; nothing filled")]
    store.update_order("x1", status="rejected", message="late reject")
    assert store.closing_orders("s")[0]["status"] == "rejected"


def test_a_refused_close_writes_its_warning_event(store):
    rt = SleeveRuntime(store, "s", now=lambda: T)
    _order(store, "x1", "stop_loss", 1)
    rt.on_order_status("x1", "rejected", "ReduceOnly Order is rejected")
    ev = [e for e in store.events("s", min_level="warning") if e["kind"] == "order_rejected"]
    assert ev and "x1" in ev[0]["message"] and "ReduceOnly" in ev[0]["message"]


def test_a_refused_resize_is_journaled_and_leaves_the_order_as_it_was(store):
    rt = SleeveRuntime(store, "s", now=lambda: T)
    _order(store, "x1", "stop_loss", 1)
    store.update_order("x1", status="accepted")
    strat = SimpleNamespace(runtime=rt, _resizing={"x1": "resize"}, _resize_if_due=lambda: None)
    LongFlatStrategy.on_order_modify_rejected(strat, SimpleNamespace(client_order_id="x1",
                                                                     reason="Order would immediately trigger"))
    assert strat._resizing == {}
    ev = [e for e in store.events("s", min_level="warning") if e["kind"] == "order_modify_rejected"]
    assert ev and "x1" in ev[0]["message"] and "immediately trigger" in ev[0]["message"]
    row = store.orders("s")[0]
    assert row["status"] == "accepted"  # still working at the venue at its old size: not a refused close
    assert [r["status"] for r in store.closing_orders("s")] == ["accepted"]

