"""TRIALS-WORM (QD #238, HoE 13 Oct; migration 0018): the trials register is write-once in the database itself. A
row is never deleted; an UPDATE may change only status and error. A bare UPDATE of created_at, definition_hash,
settings or the data range, which code review can't see, is refused. CI also runs these against Postgres."""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import schema
from sleeve_fund.store import TRIALS_FROZEN, Store, make_engine, metadata

T = datetime(2026, 10, 13, 9, tzinfo=timezone.utc)
ROW = dict(id="t1", definition_hash="d", idea_hash="i", code_version="c", definition_name="n", family="f",
           settings="{}", dataset="ds", stage="in_sample", source="study", created_at=T, data_start=T, data_end=T)


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield eng
    _clean(eng)


def _edits():
    later = "'2026-10-14 09:00:00+00:00'"
    return {"created_at": f"created_at = {later}", "definition_hash": "definition_hash = 'other'",
            "settings": "settings = '{\"fast\": 9}'", "data_start": f"data_start = {later}",
            "data_end": "data_end = NULL", "sharpe": "sharpe = 3.0", "kind": "kind = 'ablation'",
            "idea_hash": "idea_hash = 'other'"}


def _refuses_edits_and_deletes(eng, row_id):
    for column, change in _edits().items():
        with pytest.raises(IntegrityError, match="write-once"), eng.begin() as c:
            c.execute(text(f"UPDATE trials SET {change} WHERE id = :i"), {"i": row_id})
    with pytest.raises(IntegrityError, match="write-once"), eng.begin() as c:
        c.execute(text("DELETE FROM trials WHERE id = :i"), {"i": row_id})
    with eng.begin() as c:  # a failed count re-counted: status and error only
        c.execute(text("UPDATE trials SET status = 'failed', error = 'x' WHERE id = :i"), {"i": row_id})
        c.execute(text("UPDATE trials SET status = 'ok', error = NULL WHERE id = :i"), {"i": row_id})
        c.execute(text("UPDATE trials SET definition_hash = definition_hash WHERE id = :i"), {"i": row_id})  # no change
        assert c.execute(text("SELECT definition_hash, settings, status FROM trials WHERE id = :i"),
                         {"i": row_id}).one() == ("d", "{}", "ok")


def test_a_store_built_by_create_all_refuses_every_edit_but_status_and_error_and_every_delete(engine):
    store = Store(engine=engine)
    assert store.add_trials([ROW]) == 1
    _refuses_edits_and_deletes(engine, "t1")


def test_0018_makes_rows_written_before_it_write_once_too_and_changes_none(engine):
    from alembic import command

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0014")
        conn.execute(text("INSERT INTO trials (id, definition_hash, idea_hash, code_version, definition_name, family, "
                          "settings, dataset, stage, source, created_at, data_start, data_end) VALUES ('old', 'd', "
                          "'i', 'c', 'n', 'f', '{}', 'ds', 'in_sample', 'study', :t, :t, :t)"), {"t": T})
    schema.migrate(engine, log=lambda _: None)
    schema.migrate(engine, log=lambda _: None)  # again: nothing to do, nothing doubled
    _refuses_edits_and_deletes(engine, "old")


def test_every_column_but_status_and_error_is_frozen():
    assert set(TRIALS_FROZEN) == {c.name for c in metadata.tables["trials"].columns} - {"status", "error"}
