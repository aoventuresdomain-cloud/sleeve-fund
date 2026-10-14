"""F-WED1-3 PRE (Advisor 14 Oct 03:35 UK): a Strategy's venue pin (sleeve_venues) can't be changed or replaced while
it holds a position or has a working order, on either engine and whoever writes it: the database refuses, so the
mismatched-pin start stays unreachable from the journal. The journal's own writers (create_sleeve, split_run) still
work."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import router
from sleeve_fund.store import Store, backtests_t, make_engine, metadata

PINNED = "venue is pinned"  # the guard's message on both engines (DA, 0020)


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


def _sleeve(store, name, venue=None, instrument="BTC/USD"):
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument=instrument, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue=venue)


def _holding(store, name):
    """A filled entry: the Strategy holds a position."""
    store.record_order(name, order_id=f"{name}-o", side="BUY", qty=1.0, intent="entry", reason="r")
    store.record_fill(name, side="BUY", qty=1.0, price=100.0, fee=0.0, order_id=f"{name}-o", trade_id="t1")


def _working(store, name):
    """A submitted order the venue hasn't filled: a working order."""
    store.record_order(name, order_id=f"{name}-w", side="BUY", qty=1.0, intent="entry", reason="r")


STATES = {"holding": _holding, "working": _working}


def _pin(store, name):
    with store.engine.connect() as c:
        return c.execute(text("SELECT venue FROM sleeve_venues WHERE sleeve = :s"), {"s": name}).scalar()


@pytest.mark.parametrize("state", STATES)
def test_a_pinned_strategys_pin_cannot_be_updated_while_it_holds_or_has_a_working_order(store, state):
    _sleeve(store, "s", "BINANCE", "BTC/USDT")
    STATES[state](store, "s")
    with pytest.raises(IntegrityError, match=PINNED), store.engine.begin() as c:
        c.execute(text("UPDATE sleeve_venues SET venue = 'KRAKEN' WHERE sleeve = 's'"))
    assert _pin(store, "s") == "BINANCE"


@pytest.mark.parametrize("state", STATES)
def test_a_pinned_strategys_pin_cannot_be_replaced_by_a_delete_and_an_insert(store, state):
    _sleeve(store, "s", "BINANCE", "BTC/USDT")
    STATES[state](store, "s")
    with pytest.raises(IntegrityError, match=PINNED), store.engine.begin() as c:  # refused at the DELETE
        c.execute(text("DELETE FROM sleeve_venues WHERE sleeve = 's'"))
        c.execute(text("INSERT INTO sleeve_venues (sleeve, venue) VALUES ('s', 'KRAKEN')"))
    assert _pin(store, "s") == "BINANCE"
    with pytest.raises(IntegrityError, match=PINNED), store.engine.begin() as c:  # nor the delete alone
        c.execute(text("DELETE FROM sleeve_venues WHERE sleeve = 's'"))
    assert _pin(store, "s") == "BINANCE"


@pytest.mark.parametrize("state", STATES)
def test_an_unpinned_strategy_cannot_be_given_a_pin_once_it_holds_or_has_a_working_order(store, state):
    # A pin added late moves it off the default venue its orders went to: a replacement too.
    _sleeve(store, "s")
    STATES[state](store, "s")
    with pytest.raises(IntegrityError, match=PINNED), store.engine.begin() as c:
        c.execute(text("INSERT INTO sleeve_venues (sleeve, venue) VALUES ('s', 'KRAKEN')"))
    assert _pin(store, "s") is None


def test_the_journals_own_writers_still_pin_a_new_strategy_and_carry_the_pin_to_an_archived_run(store):
    _sleeve(store, "s", "BINANCE", "BTC/USDT")
    assert _pin(store, "s") == "BINANCE"
    _holding(store, "s")
    store.record_order("s", order_id="s-x", side="SELL", qty=1.0, intent="exit", reason="r")
    store.record_fill("s", side="SELL", qty=1.0, price=101.0, fee=0.0, order_id="s-x", trade_id="t2")  # flat: a reset
    store.request_reset("s", "Settings changed")
    run = store.split_run(store.pending_resets()[0])  # copies the pin to the run, then moves the orders to it
    assert _pin(store, run) == "BINANCE" and _pin(store, "s") == "BINANCE"
    assert router.venue_for(store.sleeve(run).venue) == "BINANCE"


def test_the_prune_still_removes_a_saved_backtests_pin_with_its_orders(store):
    import pandas as pd

    _sleeve(store, "bt:b1", "BINANCE", "BTC/USDT")
    _holding(store, "bt:b1")
    with store.engine.begin() as c:
        c.execute(backtests_t.insert().values(id="b1", sleeve="bt:b1", key="k", title="t", query="q",
                                              created_at=pd.Timestamp("2026-10-01", tz="UTC"), result="{}"))
    assert store.prune_backtests(keep=0) == 1
    assert _pin(store, "bt:b1") is None
