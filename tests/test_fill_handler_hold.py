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


def _with_stop(path, stop=0.02):
    """The synthetic session, its Model given a 2% stop-loss (the header's sleeve params), for the F-T-2 cells."""
    import gzip
    import json

    with gzip.open(path, "rt") as f:
        lines = f.read().splitlines()
    head = json.loads(lines[0])
    head["sleeve"]["params"]["stop_loss"] = stop
    with gzip.open(path, "wt") as f:
        f.write("\n".join([json.dumps(head), *lines[1:]]) + "\n")
    return path

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


@pytest.mark.strategy_errors
def test_a_fill_handler_failing_at_the_stop_step_closes_the_position_and_holds(tmp_path, monkeypatch, journal):
    """F-T-2 cell B (Advisor ~21:40 UK): the handler fails after booking the fill but before its stop is set. No
    position is left without a working stop: it is closed at market, and the hold is set."""
    real, calls = LongFlatStrategy.on_order_filled, []

    def breaks_at_the_stop(self, event):
        real(self, event)
        calls.append(event)
        if len(calls) == 1:
            self._stop_frac = None  # the stop step never ran
            raise RuntimeError("stop step broke")

    monkeypatch.setattr(LongFlatStrategy, "on_order_filled", breaks_at_the_stop)
    orders = replay(_with_stop(_synthetic(tmp_path / "s.jsonl.gz")), store=journal)
    assert [o["intent"] for o in orders][:2] == ["entry", "exit"], orders
    assert orders[1]["reason"].startswith("Fill-handler hold: the position has no working stop")
    assert [o["intent"] for o in orders].count("entry") == 1
    s = journal.sleeve("replay-test")
    assert s.status == "paused" and s.paused_until is None and s.status_reason == FILL_HANDLER_PAUSE
    kinds = [e["kind"] for e in journal.events("replay-test", limit=1000)]
    assert "fill_hold_flatten" in kinds and "reconcile_mismatch" not in kinds
    assert journal.journal_book("replay-test", s.starting_balance)["qty"] == pytest.approx(0, abs=1e-9)


@pytest.mark.strategy_errors
def test_a_fill_handler_failing_after_its_stop_keeps_the_position_under_that_stop(tmp_path, monkeypatch, journal):
    """F-T-2 (Advisor ~21:40 UK): when the stop was set from the fill before the handler failed, the position stays
    under it (the hold only blocks entries) and the journal shows the watched stop."""
    real, calls = LongFlatStrategy.on_order_filled, []

    def breaks_once(self, event):
        real(self, event)
        calls.append(event)
        if len(calls) == 1:
            raise RuntimeError("fill handler broke")

    monkeypatch.setattr(LongFlatStrategy, "on_order_filled", breaks_once)
    replay(_with_stop(_synthetic(tmp_path / "s.jsonl.gz")), store=journal)
    kinds = [e["kind"] for e in journal.events("replay-test", limit=1000)]
    assert "fill_hold_flatten" not in kinds
    assert any(o["intent"] == "stop_loss" and (o.get("signal") or {}).get("watched")
               for o in journal.orders("replay-test", limit=1000))
