"""DA-2 (phase 2 DB hardening): a fill is booked once per (strategy, order, trade). A replayed fill changes nothing:
not the fills, not the order's filled quantity, not cash. A different fill under a booked key is kept out and raised
as an error. CI also runs the Store cases against Postgres via TEST_DATABASE_URL."""

import os
import threading

import pytest
from sqlalchemy.exc import IntegrityError

from sleeve_fund.paper.journal import MemoryJournal
from sleeve_fund.paper.queued import QueuedStore
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.store import Store, fills_t, insert


@pytest.fixture
def store(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


def _setup(j, name="s1"):
    j.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                    starting_balance=10_000)
    j.record_order(name, order_id="O-1", side="BUY", qty=0.1, intent="entry", reason="test")


FILL = dict(side="BUY", qty=0.05, price=60_000.0, fee=1.5, order_id="O-1", trade_id="T-1")


@pytest.fixture(params=["store", "memory"])
def journal(request, store):
    return store if request.param == "store" else MemoryJournal()


def test_a_replayed_fill_is_booked_once_and_moves_its_order_once(journal):
    _setup(journal)
    assert journal.book_fill("s1", **FILL) == "new"
    book = journal.journal_book("s1", 10_000)
    assert journal.book_fill("s1", **FILL) == "same"
    assert len(journal.fills("s1")) == 1
    assert journal.journal_book("s1", 10_000) == book
    order = next(o for o in journal.orders("s1") if o["order_id"] == "O-1")
    assert float(order["filled_qty"]) == pytest.approx(0.05) and order["status"] == "partially_filled"
    assert sum(e["kind"] == "fill" for e in journal.events("s1")) == 1


def test_a_different_fill_under_a_booked_key_is_kept_out_and_raised(journal):
    _setup(journal)
    journal.book_fill("s1", **FILL)
    assert journal.book_fill("s1", **{**FILL, "qty": 0.07}) == "differs"
    (f,) = journal.fills("s1")
    assert float(f["qty"]) == pytest.approx(0.05)
    order = next(o for o in journal.orders("s1") if o["order_id"] == "O-1")
    assert float(order["filled_qty"]) == pytest.approx(0.05)
    (e,) = [e for e in journal.events("s1", min_level="error") if e["kind"] == "fill_conflict"]
    assert "O-1" in e["message"] and "T-1" in e["message"]


def test_the_same_trade_id_on_a_new_order_is_a_new_fill(journal):
    """The paper venue's trade ids repeat after a restart; the order id doesn't, so the key includes it."""
    _setup(journal)
    journal.record_order("s1", order_id="O-2", side="BUY", qty=0.05, intent="entry", reason="test")
    assert journal.book_fill("s1", **FILL) == "new"
    assert journal.book_fill("s1", **{**FILL, "order_id": "O-2"}) == "new"
    assert len(journal.fills("s1")) == 2


def test_the_database_refuses_a_second_row_with_the_same_key(store):
    _setup(store)
    store.record_fill("s1", **FILL)
    with pytest.raises(IntegrityError):
        with store.engine.begin() as c:
            c.execute(insert(fills_t).values(sleeve="s1", ts=store.fills("s1")[0]["ts"], **FILL))


def test_two_writers_of_the_same_fill_book_it_once(store):
    _setup(store)
    out, go = [], threading.Barrier(4)

    def write():
        go.wait()
        out.append(store.book_fill("s1", **FILL))

    threads = [threading.Thread(target=write) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(out) == ["new", "same", "same", "same"]
    assert len(store.fills("s1")) == 1
    assert float(store.orders("s1")[0]["filled_qty"]) == pytest.approx(0.05)


def test_paper_runtime_through_the_queued_journal_books_a_replay_once(store):
    _setup(store)
    rt = SleeveRuntime(QueuedStore(store), "s1", tick_seconds=60)
    for _ in range(2):
        rt.on_fill(**FILL)
    rt.store.flush()
    assert len(store.fills("s1")) == 1 and float(store.orders("s1")[0]["filled_qty"]) == pytest.approx(0.05)


def test_the_migration_stops_on_fills_already_booked_twice_and_changes_nothing(tmp_path):
    from alembic import command
    from sqlalchemy import text

    from sleeve_fund import schema
    from sleeve_fund.store import make_engine

    eng = make_engine(f"sqlite:///{tmp_path}/m.db")
    with eng.begin() as conn:
        command.upgrade(schema._config(conn), "0007")
        conn.execute(text("PRAGMA foreign_keys=OFF"))  # the fills alone; no strategy row needed for the check
        for _ in range(2):
            conn.execute(text("INSERT INTO fills (sleeve, ts, side, qty, price, fee, order_id, trade_id) VALUES "
                              "('s1', CURRENT_TIMESTAMP, 'BUY', 1, 1, 0, 'O-1', 'T-1')"))
    with pytest.raises(RuntimeError, match="s1 O-1/T-1 x2"):
        with eng.begin() as conn:
            command.upgrade(schema._config(conn), "head")
    with eng.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM fills")).scalar() == 2
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0007"
