"""The live-schema drift check (sleeve_fund.schema): silent when the database matches store.py, loud when it
doesn't, and it never writes. CI also runs it against Postgres via TEST_DATABASE_URL."""

import os

import pytest
from sqlalchemy import event, text

from sleeve_fund import schema
from sleeve_fund.store import Store, make_engine, metadata


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/t.db")
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS stray"))
    metadata.drop_all(eng)
    Store(engine=eng)
    yield eng
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS stray"))
    metadata.drop_all(eng)


def test_a_database_built_from_the_code_matches(engine):
    facts, diffs = schema.report(engine)
    assert diffs == []
    assert f"tables: {len(metadata.tables)} in the database, {len(metadata.tables)} in the code" in facts


def test_drift_is_listed_table_by_table_and_column_by_column(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE stray (x INTEGER)"))
        conn.execute(text("ALTER TABLE fills ADD COLUMN extra INTEGER"))
        conn.execute(text("DROP INDEX fills_sleeve_ts"))
    _, diffs = schema.report(engine)
    assert "table stray: in the database but not in the code" in diffs
    assert "column fills.extra: in the database but not in the code" in diffs
    assert "index fills_sleeve_ts on fills: missing from the database" in diffs


def test_the_check_only_reads(engine):
    statements = []
    event.listen(engine, "before_cursor_execute", lambda *a: statements.append(a[2].lstrip().split()[0].upper()))
    schema.report(engine)
    writes = {"INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT"}
    assert statements and not writes & set(statements)


def test_cli_exits_non_zero_on_drift(engine, monkeypatch, capfd):
    monkeypatch.setattr(schema, "make_engine", lambda: engine)
    assert schema.main(["check"]) == 0 and "schema: matches the code" in capfd.readouterr().out
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE stray (x INTEGER)"))
    assert schema.main(["check"]) == 1 and "SCHEMA DRIFT: 1 difference" in capfd.readouterr().out
    assert schema.main([]) == 2
