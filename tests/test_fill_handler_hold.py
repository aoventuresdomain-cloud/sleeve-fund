"""T-FIX8 (the Advisor's loud failure, HoQA on T): a fill handler that raises is logged and journaled as
handler_failed, and the Strategy opens nothing more until the PM resumes or restarts it; its exits still go."""

import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from sleeve_fund.research.replay import replay
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import FILL_HANDLER_PAUSE, LongFlatStrategy
from tests.test_replay import _synthetic

PG = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def journal(request):
    if request.param == "sqlite":
        yield Store.in_memory()
        return
    if not PG or not PG.startswith("postgresql"):
        pytest.skip("Postgres only with TEST_DATABASE_URL")
    schema = f"tfix8_{uuid.uuid4().hex[:10]}"
    admin = create_engine(PG)
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {schema}"))
    eng = create_engine(PG, connect_args={"options": f"-csearch_path={schema}"})
    yield Store(engine=eng)
    eng.dispose()
    with admin.begin() as c:
        c.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    admin.dispose()


@pytest.mark.strategy_errors
def test_a_fill_handler_that_raises_holds_new_entries_and_the_exit_still_goes(tmp_path, monkeypatch, journal):
    real, calls = LongFlatStrategy.on_order_filled, []

    def breaks_once(self, event):
        real(self, event)  # the fill is booked; the handler fails after it, so the journal and the engine agree
        calls.append(event)
        if len(calls) == 1:
            raise RuntimeError("fill handler broke")

    monkeypatch.setattr(LongFlatStrategy, "on_order_filled", breaks_once)
    orders = replay(_synthetic(tmp_path / "s.jsonl.gz"), store=journal)
    intents = [o["intent"] for o in orders]
    assert intents[0] == "entry" and intents.count("entry") == 1, intents  # nothing opens after the failed fill
    assert "exit" in intents[1:], intents  # its exit still goes
    s = journal.sleeve("replay-test")
    assert s.status == "paused" and s.paused_until is None and s.status_reason == FILL_HANDLER_PAUSE
    kinds = [e["kind"] for e in journal.events("replay-test", limit=1000)]
    assert "handler_failed" in kinds and "reconcile_mismatch" not in kinds
