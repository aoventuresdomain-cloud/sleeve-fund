"""B-16 archive storage (DA, migration 0031; PM 15 Oct 17:22 UK, mock-up v5 approved 17:39 UK). Archiving a research
Strategy only hides it from the Strategies list: one row in definition_archive and one decision log entry, in one
transaction. The rule file, its trials rows and its tries count never change, and nothing is deleted. Restore sets the
row's restored_* once. The database itself refuses a delete or any other edit, on SQLite and on Postgres
(TEST_DATABASE_URL), as the trials register does."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete, select, text, update
from sqlalchemy.exc import DatabaseError, IntegrityError

from sleeve_fund.research.trials import model_run_row
from sleeve_fund.store import Store

PG = os.environ.get("TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[1]
FILE = "qr01_trend_ema_long"
_SCHEMAS: list[str] = []
_ENGINES: list = []


def _engine(schema: str | None = None):
    args = {"options": f"-csearch_path={schema}"} if schema else {}
    return create_engine(PG, connect_args=args)


@pytest.fixture(scope="module", autouse=True)
def _drop_schemas():
    yield
    for eng in _ENGINES:
        eng.dispose()
    if PG and _SCHEMAS:
        admin = _engine()
        for s in _SCHEMAS:
            with admin.begin() as c:
                c.execute(text(f"DROP SCHEMA IF EXISTS {s} CASCADE"))
        admin.dispose()


def _stores():
    out = [Store.in_memory()]
    if PG:
        name = f"da_{uuid.uuid4().hex[:12]}"
        with _engine().begin() as c:
            c.execute(text(f"CREATE SCHEMA {name}"))
        _SCHEMAS.append(name)
        eng = _engine(name)
        _ENGINES.append(eng)
        out.append(Store(engine=eng))
    return out


def _files() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "configs" / "definitions").glob("*.toml"))}


def _with_trials(store):
    row = model_run_row(strategy="rules", params={"definition": FILE}, dataset="d", source="backtest",
                        setup={"risk_profile": None}, sharpe=0.8)
    store.add_trials([row])
    return row["idea_hash"]


@pytest.fixture(params=["sqlite", "postgres"])
def store(request):
    if request.param == "postgres" and not PG:
        pytest.skip("Postgres cells run where TEST_DATABASE_URL is set (CI)")
    return _stores()[-1 if request.param == "postgres" else 0]


def test_archive_then_restore_changes_no_file_trial_or_tries_and_logs_both(store):
    idea = _with_trials(store)
    files, trials = _files(), store.trials(idea)
    row = store.archive_definition(FILE, definition_hash="h1", status="tested", reason="no_edge", note="flat after fees",
                                   actor="pm")
    assert (row["file_id"], row["reason"], row["note"], row["status_at_archive"], row["restored_at"]) == (
        FILE, "no_edge", "flat after fees", "tested", None)
    assert [r["file_id"] for r in store.archived_definitions()] == [FILE]
    logged = store.decisions(action="research_archive")
    assert len(logged) == 1 and logged[0]["sleeve"] is None and logged[0]["actor"] == "pm"
    assert logged[0]["reason"] == f"{FILE}: No edge after testing: flat after fees"

    back = store.restore_definition(FILE, reason="Re-test at new fees", actor="pm")
    assert back["restored_at"] is not None and back["restored_by"] == "pm"
    assert back["restore_reason"] == "Re-test at new fees"
    assert store.archived_definitions() == []
    logged = store.decisions(action="research_restore")
    assert len(logged) == 1 and logged[0]["sleeve"] is None
    assert logged[0]["reason"] == f"{FILE}: Re-test at new fees"
    assert _files() == files and store.trials(idea) == trials  # nothing about the idea or its tries moved
    assert [r["restored_at"] is not None for r in store.definition_archive_history(FILE)] == [True]


def test_a_restored_strategy_can_be_archived_again_as_a_new_row(store):
    store.archive_definition(FILE, definition_hash="h1", status="tested", reason="duplicate", note=None, actor="pm")
    store.restore_definition(FILE, reason="Wanted back", actor="pm")
    store.archive_definition(FILE, definition_hash="h1", status="tested", reason="superseded", note=None, actor="pm")
    hist = store.definition_archive_history(FILE)
    assert [h["reason"] for h in hist] == ["superseded", "duplicate"]  # newest first
    assert [r["reason"] for r in store.archived_definitions()] == ["superseded"]
    assert store.decisions(action="research_archive")[0]["reason"] == f"{FILE}: Superseded by another Strategy"


def test_refusals_write_nothing(store):
    with pytest.raises(ValueError):
        store.archive_definition(FILE, definition_hash=None, status="tested", reason="bored", note=None, actor="pm")
    with pytest.raises(ValueError):
        store.restore_definition(FILE, reason="x", actor="pm")
    with pytest.raises(ValueError):
        store.archive_definition(FILE, definition_hash=None, status="tested", reason="not_wanted", note=None, actor="")
    store.archive_definition(FILE, definition_hash=None, status="tested", reason="not_wanted", note=None, actor="pm")
    with pytest.raises(ValueError):
        store.archive_definition(FILE, definition_hash=None, status="tested", reason="not_wanted", note=None,
                                 actor="pm")
    with pytest.raises(ValueError):
        store.restore_definition(FILE, reason="  ", actor="pm")  # a restore needs a reason
    assert len(store.definition_archive_history(FILE)) == 1
    assert len(store.decisions(action="research_archive")) == 1 and store.decisions(action="research_restore") == []


def test_two_open_rows_for_one_file_are_refused_by_the_database(store):
    from sleeve_fund.store import definition_archive_t, utcnow

    row = dict(file_id=FILE, definition_hash=None, status_at_archive="tested", reason="duplicate", note=None,
               actor="pm", archived_at=utcnow())
    with store.engine.begin() as c:
        c.execute(definition_archive_t.insert().values(**row))
    with pytest.raises(IntegrityError), store.engine.begin() as c:
        c.execute(definition_archive_t.insert().values(**row))


def test_the_database_refuses_a_delete_or_an_edit_and_sets_restored_once(store):
    from sleeve_fund.store import definition_archive_t as t

    store.archive_definition(FILE, definition_hash="h1", status="tested", reason="duplicate", note=None, actor="pm")
    for stmt in (delete(t), update(t).values(reason="no_edge"), update(t).values(note="edited"),
                 update(t).values(restored_by="pm")):  # restored_* go in together, never one alone
        with pytest.raises(DatabaseError), store.engine.begin() as c:
            c.execute(stmt)
    store.restore_definition(FILE, reason="Back", actor="pm")
    with pytest.raises(DatabaseError), store.engine.begin() as c:
        c.execute(update(t).values(restore_reason="changed"))  # set once: never edited after
    with store.engine.connect() as c:
        assert c.execute(select(t.c.reason, t.c.restore_reason)).one() == ("duplicate", "Back")
