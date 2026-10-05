"""Schema migrations (sleeve_fund.schema.migrate, sleeve_fund/migrations): the newest migration builds exactly
store.py's schema, an existing database without history is stamped and never rebuilt, and a drifted one stops
the deploy with nothing changed. CI also runs these against Postgres via TEST_DATABASE_URL."""

import os

import pytest
from sqlalchemy import event, inspect, text

from sleeve_fund import schema
from sleeve_fund.store import Store, make_engine, metadata


def _clean(eng):
    with eng.begin() as conn:
        for t in ("alembic_version", "stray"):
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield eng
    _clean(eng)


def _statements(eng):
    seen = []
    event.listen(eng, "before_cursor_execute", lambda *a: seen.append(" ".join(a[2].split()[:3]).upper()))
    return seen


def _version(eng):
    with eng.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def test_an_empty_database_is_built_by_the_migrations_and_matches_the_code(engine):
    # This is the "every schema change has a migration" check: a table or column added to store.py
    # without one makes the migrated database differ from the code here.
    done = schema.migrate(engine, log=lambda _: None)
    assert done == f"at migration {schema.head()}" and _version(engine) == schema.head()
    _, diffs = schema.report(engine)
    assert diffs == []


def test_an_existing_database_without_history_is_stamped_not_rebuilt(engine):
    Store(engine=engine)  # what production is today: tables made by create_all, no migration history
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO accounts (name, kind, venue, note, created_at) "
                          "VALUES ('paper', 'paper', 'X', '', CURRENT_TIMESTAMP)"))
    seen = _statements(engine)
    done = schema.migrate(engine, log=lambda _: None)
    assert done.startswith("stamped existing schema")
    assert _version(engine) == schema.head()
    touched = [s for s in seen if s.split()[0] in ("CREATE", "ALTER", "DROP", "DELETE", "UPDATE", "TRUNCATE")]
    assert all("ALEMBIC_VERSION" in s for s in touched), touched  # only alembic's own bookkeeping table
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM accounts")).scalar() == 1  # the data is untouched


def test_a_drifted_database_without_history_stops_the_migration_and_nothing_changes(engine):
    Store(engine=engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE fills ADD COLUMN extra INTEGER"))
    with pytest.raises(schema.Drift, match="fills.extra"):
        schema.migrate(engine, log=lambda _: None)
    assert "alembic_version" not in inspect(engine).get_table_names()


def test_migrating_again_changes_nothing(engine):
    schema.migrate(engine, log=lambda _: None)
    seen = _statements(engine)
    schema.migrate(engine, log=lambda _: None)
    assert not [s for s in seen if s.split()[0] in ("CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE")]


def test_cli_holds_migrations_on_first_stamp_drift_and_raises_an_alert(engine, monkeypatch, capfd):
    # Before the first stamp the app runs as it always has, so the deploy goes on; the drift is an error
    # event in the alerts inbox, and nothing is stamped or migrated until it is fixed.
    store = Store(engine=engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE stray (x INTEGER)"))
    monkeypatch.setattr(schema, "make_engine", lambda: engine)
    assert schema.main(["migrate"]) == 0
    assert "migrations held, nothing changed" in capfd.readouterr().err
    (ev,) = store.events_of(("schema_drift",))
    assert ev["level"] == "error" and "table stray" in ev["message"]
    assert "alembic_version" not in inspect(engine).get_table_names()


def test_cli_migrate_fails_the_deploy_when_a_migration_fails(engine, monkeypatch):
    schema.migrate(engine, log=lambda _: None)
    monkeypatch.setattr(schema.command, "upgrade", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(schema, "make_engine", lambda: engine)
    with pytest.raises(RuntimeError):
        schema.main(["migrate"])  # uncaught: the process exits non-zero and the services don't start
