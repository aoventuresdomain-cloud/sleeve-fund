"""DA-5 and DA-6 (DA charter; Drop B rider): a reset never leaves a table that names a Strategy unplanned, and the
journal's links hold on SQLite too.

DA-6: every table with a Strategy column (sleeve, sleeve_id, parent, strategy_id) or a foreign key to sleeves is in
exactly one of the reset's lists (moves with the run, copied to it, cleared, or stays, with its reason), and a reset
does what its list says. A new table in none of them fails here first.
DA-5: SQLite enforces foreign keys only when a connection asks; with them on, a journal written and reset through the
store's own paths has no broken link, and a row pointing at no Strategy is refused, as Postgres refuses it."""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import store as store_mod
from sleeve_fund.store import (RESET_CLEARS, RESET_COPIES, RESET_MOVES, RESET_STAYS, STRATEGY_COLUMNS, Store,
                               equity_t, feed_seen_t, make_engine, metadata, sleeves_t)

T = datetime(2026, 10, 15, 5, 0, tzinfo=timezone.utc)


def _names_a_strategy(t) -> bool:
    return bool(STRATEGY_COLUMNS & set(t.c.keys())) or any(fk.column.table is sleeves_t for fk in t.foreign_keys)


def test_every_table_naming_a_strategy_is_in_exactly_one_reset_list():
    lists = {"moves": RESET_MOVES, "copies": RESET_COPIES, "clears": RESET_CLEARS, "stays": tuple(RESET_STAYS)}
    seen = {}
    for kind, tables in lists.items():
        for t in tables:
            assert t.name not in seen, f"{t.name} is in both {seen[t.name]} and {kind}"
            seen[t.name] = kind
    naming = {t.name for t in metadata.sorted_tables if _names_a_strategy(t)}
    missing = sorted(naming - set(seen))
    assert not missing, (f"tables naming a Strategy that a reset doesn't plan for: {missing}. Add each to RESET_MOVES, "
                         "RESET_COPIES, RESET_CLEARS or RESET_STAYS (with why) in store.py, and ask the Data Architect")
    assert set(seen) <= naming, f"listed but naming no Strategy: {sorted(set(seen) - naming)}"
    assert all(why.strip() for why in RESET_STAYS.values())


def test_the_lists_cover_a_table_added_later():
    """The guard itself: a table with a sleeve column, or only a foreign key to sleeves under another name, is caught."""
    from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table

    m = MetaData()
    Table("sleeves", m, Column("id", Integer, primary_key=True), Column("name", String(64)))
    by_column = Table("new_journal", m, Column("id", Integer, primary_key=True), Column("sleeve", String(64)))
    assert _names_a_strategy(by_column)
    by_fk = Table("new_marks", metadata, Column("id", Integer, primary_key=True),
                  Column("owner", String(64), ForeignKey("sleeves.name")))
    try:
        assert _names_a_strategy(by_fk)
    finally:
        metadata.remove(by_fk)


@pytest.fixture
def store(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/reset.db")
    if eng.dialect.name == "sqlite":  # DA-5: links enforced, as Postgres enforces them
        event.listen(eng, "connect", lambda dbapi, _: dbapi.execute("PRAGMA foreign_keys=ON"))
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    yield Store(engine=eng)
    metadata.drop_all(eng)


def _count(c, t, name):
    return c.execute(select(func.count()).select_from(t).where(t.c.sleeve == name)).scalar()


def test_a_reset_moves_copies_clears_and_keeps_exactly_what_its_lists_say(store):
    store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE")
    for side, n in (("BUY", 1), ("SELL", 2)):  # in and out again: flat, so it can be reset
        store.record_order("s", order_id=f"o{n}", side=side, qty=1.0, intent="entry" if n == 1 else "exit",
                           reason="r")
        store.record_fill("s", side=side, qty=1.0, price=100.0 + n, fee=0.1, order_id=f"o{n}", trade_id=f"t{n}")
    with store.engine.begin() as c:
        c.execute(feed_seen_t.insert().values(sleeve="s", seen_at=T))
    store.request_reset("s", "Settings changed")
    with store.engine.connect() as c:
        before = {t.name: _count(c, t, "s") for t in RESET_MOVES + RESET_COPIES}
    assert before["fills"] == 2 and before["orders"] == 2 and before["sleeve_venues"] == 1 and before["decisions"]
    run = store.split_run(store.pending_resets()[0])
    with store.engine.connect() as c:
        for t in RESET_MOVES:
            assert (_count(c, t, "s"), _count(c, t, run)) == (0, before[t.name]), t.name
        for t in RESET_COPIES:
            assert _count(c, t, "s") == _count(c, t, run) == before[t.name], t.name
        for t in RESET_CLEARS:
            assert _count(c, t, "s") == 0, t.name
        if c.dialect.name == "sqlite":
            assert c.execute(text("PRAGMA foreign_key_check")).all() == []
    assert store.journal_book("s", 10_000.0)["fills"] == 0


def test_a_journal_row_naming_no_strategy_is_refused_on_both_engines(store):
    with pytest.raises(IntegrityError), store.engine.begin() as c:
        c.execute(equity_t.insert().values(sleeve="nobody", ts=T, equity=1, cash=1, qty=0, price=1, benchmark=1))


def test_the_reset_lists_are_the_ones_split_run_uses():
    """No second copy of the lists inside split_run: the code names the module's lists."""
    import inspect

    src = inspect.getsource(store_mod.Store.split_run)
    assert all(n in src for n in ("RESET_MOVES", "RESET_COPIES", "RESET_CLEARS"))
    assert "fills_t" not in src and "signal_state_t" not in src
