"""F217-3: a liquidation_reset names the command it carries out (events.command_id, migration 0014), and the process's
retry after a crash reads whether its own reset was journaled from that key, never from the message's wording.
One reset per command is enforced by the database (events_command_once). CI also runs these against Postgres via
TEST_DATABASE_URL."""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
from sqlalchemy import inspect, text, update
from sqlalchemy.exc import IntegrityError

from sleeve_fund import schema
from sleeve_fund.store import LIQUIDATION_RESET, Store, events_t, make_engine, metadata
from tests.test_ral_xfails import (  # noqa: F401  (store is a fixture)
    NAME,
    NEXT_DAY,
    _liquidate,
    _noted,
    _ral,
    _ral_events,
    _runtime,
    store,
)


def _ral_command(store):
    return next(c for c in store.pending_commands(NAME) if c["incident"] is not None)


def test_a_second_reset_for_the_same_command_is_refused_by_the_database(store, tmp_path):
    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    cid = _ral_command(store)["id"]
    store.event(NAME, "info", LIQUIDATION_RESET, "QA: the reset", command_id=cid)
    with pytest.raises(IntegrityError):
        store.event(NAME, "info", LIQUIDATION_RESET, "QA: the same reset again", command_id=cid)
    assert store.reset_for_command(cid)["message"] == "QA: the reset"


def test_only_a_liquidation_reset_names_a_command(store, tmp_path):
    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    with pytest.raises(ValueError):
        store.event(NAME, "info", "pm_pause", "QA", command_id=_ral_command(store)["id"])


def test_rows_without_a_command_never_collide(store):
    for _ in range(3):
        store.event(NAME, "info", "restart", "QA")
        store.event(NAME, "info", LIQUIDATION_RESET, "QA: a book reset's own reset, no command")
    with store.engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM events WHERE command_id IS NULL")).scalar() >= 6


def test_the_retry_reads_its_reset_from_the_key_whatever_the_message_says(store, tmp_path, monkeypatch):
    """The process journals the reset and dies before marking the command applied; the message is then reworded (an
    old wording, an edit). The restart still finds its own reset by command_id and finishes without a second one."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    cid = _ral_command(store)["id"]
    real = SleeveRuntime._set
    monkeypatch.setattr(SleeveRuntime, "_set", lambda self, status, *a, **kw: (_ for _ in ()).throw(
        RuntimeError("QA: dies")) if status == "running" else real(self, status, *a, **kw))
    with pytest.raises(RuntimeError):
        _runtime(store, NEXT_DAY, f.rem)
    monkeypatch.setattr(SleeveRuntime, "_set", real)
    assert store.reset_for_command(cid) is not None
    with store.engine.begin() as c:
        c.execute(update(events_t).where(events_t.c.command_id == cid).values(message="reworded: no incident here"))
    rt = _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    assert rt.status == "running" and not store.pending_commands(NAME) and len(_ral_events(store)) == 1
    assert not [e for e in store.events(NAME, limit=50) if e["kind"] == "ral_ignored"]


def test_a_reset_applied_for_a_stopped_strategy_names_its_command_too(store, tmp_path):
    """Store._apply_ral (a stopped, flat strategy no process runs) writes the same key in the command's own
    transaction, so the reset and its command are applied together."""
    f = _liquidate(tmp_path, store)
    store.set_desired_state(NAME, "stopped")
    _ral(store, incident=_noted(store, f.liq))
    assert not store.pending_commands(NAME) and len(_ral_events(store)) == 1
    with store.engine.connect() as c:
        cid = c.execute(text("SELECT id FROM commands WHERE incident IS NOT NULL")).scalar()
    assert store.reset_for_command(cid) is not None


# --- migration 0014 -------------------------------------------------------------------------------------------------

def _clean(eng):
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield eng
    _clean(eng)


def _events_shape(eng):
    i = inspect(eng)
    return (sorted((c["name"], c["nullable"]) for c in i.get_columns("events")),
            sorted((x["name"], tuple(x["column_names"]), bool(x["unique"])) for x in i.get_indexes("events")),
            sorted((tuple(f["constrained_columns"]), f["referred_table"], (f.get("options") or {}).get("ondelete"))
                   for f in i.get_foreign_keys("events")))


def test_0014_migrated_events_table_equals_create_alls(engine, tmp_path):
    schema.migrate(engine)
    migrated = _events_shape(engine)
    fresh = make_engine(f"sqlite:///{tmp_path}/fresh.db") if engine.dialect.name == "sqlite" else engine
    if fresh is engine:
        _clean(engine)
    Store(engine=fresh)
    built = _events_shape(fresh)
    assert migrated == built
    assert ("command_id", True) in built[0] and ("events_command_once", ("command_id",), True) in built[1]
    assert (("command_id",), "commands", "RESTRICT") in built[2]


def test_0014_on_a_database_create_all_already_built_adds_nothing_twice(engine):
    """A database a newer Store built (create_all: the column and the index exist) whose history says 0013: 0014 skips
    both steps, and the result matches the code."""
    from alembic import command

    Store(engine=engine)
    with engine.begin() as conn:
        command.stamp(schema._config(conn), "0013")
    assert schema.migrate(engine, log=lambda _: None) == f"at migration {schema.head()}"
    assert _events_shape(engine)[1].count(("events_command_once", ("command_id",), True)) == 1
    assert schema.report(engine)[1] == []


def test_0014_on_a_database_at_0013_adds_the_column_and_index(engine):
    from alembic import command

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0013")
    assert "command_id" not in {c["name"] for c in inspect(engine).get_columns("events")}
    assert schema.migrate(engine, log=lambda _: None) == f"at migration {schema.head()}"
    assert ("events_command_once", ("command_id",), True) in _events_shape(engine)[1]
    assert schema.report(engine)[1] == []


def test_drop_all_clears_a_database_built_before_0014(engine):
    """The harnesses' drop_all on the shared Postgres meets databases another test left at an older revision: events
    without events_command_id_fkey. Dropping that FK by name must be skipped there, not fail every later test."""
    from alembic import command

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0013")
    metadata.drop_all(engine)
    assert not inspect(engine).has_table("events")


def test_the_fk_drop_runs_only_where_the_fk_exists(engine):
    from sqlalchemy import Column, ForeignKey, Integer, MetaData, Table
    from sqlalchemy.schema import AddConstraint, DropConstraint

    from sleeve_fund.store import _unless_dropping_an_absent_fk

    # A copy of the FK on its own MetaData: building DropConstraint on the shared one would stop create_all rendering it.
    m = MetaData()
    Table("commands", m, Column("id", Integer, primary_key=True))
    events = Table("events", m, Column("id", Integer, primary_key=True),
                   Column("command_id", Integer, ForeignKey("commands.id", use_alter=True, name="events_command_id_fkey")))
    fk = next(iter(events.foreign_key_constraints))
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE events (id INTEGER PRIMARY KEY, command_id INTEGER)"))
    with engine.connect() as conn:
        assert not _unless_dropping_an_absent_fk(DropConstraint(fk), fk, conn)
        assert _unless_dropping_an_absent_fk(AddConstraint(fk), fk, conn)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE events"))
    metadata.create_all(engine)
    with engine.connect() as conn:
        assert _unless_dropping_an_absent_fk(DropConstraint(fk), fk, conn)


def _migration_0014():
    import importlib.util
    from pathlib import Path

    import sleeve_fund

    path = Path(sleeve_fund.__file__).parent / "migrations" / "versions" / "0014_events_command_id.py"
    spec = importlib.util.spec_from_file_location("migration_0014", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_reset_journaled_before_the_migration_is_keyed_by_its_backfill_and_finishes_once(store, tmp_path,
                                                                                          monkeypatch):
    """DA on F217-3: a RAL still pending across the deploy that brings 0014, its reset already journaled under the old
    wording with no key. 0014's one-off backfill matches it as the process used to ("answering incident #N (") and
    keys it to its command, so the restart finishes it without a second reset."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    cid = _ral_command(store)["id"]
    store.event(NAME, "info", "restart", "QA: an unrelated row stays unkeyed")
    real = SleeveRuntime._set
    monkeypatch.setattr(SleeveRuntime, "_set", lambda self, status, *a, **kw: (_ for _ in ()).throw(
        RuntimeError("QA: dies")) if status == "running" else real(self, status, *a, **kw))
    with pytest.raises(RuntimeError):
        _runtime(store, NEXT_DAY, f.rem)
    monkeypatch.setattr(SleeveRuntime, "_set", real)
    with store.engine.begin() as c:  # as the row stood before 0014: no key
        c.execute(update(events_t).where(events_t.c.command_id == cid).values(command_id=None))
    assert store.reset_for_command(cid) is None
    with store.engine.begin() as c:
        _migration_0014()._backfill(c)
        _migration_0014()._backfill(c)  # idempotent
    assert store.reset_for_command(cid) is not None
    with store.engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM events WHERE command_id IS NOT NULL")).scalar() == 1
    rt = _runtime(store, NEXT_DAY + timedelta(minutes=1), f.rem)
    assert rt.status == "running" and not store.pending_commands(NAME) and len(_ral_events(store)) == 1
