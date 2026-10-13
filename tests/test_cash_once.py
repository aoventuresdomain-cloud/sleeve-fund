"""CASH-1: every cash movement is booked once. A funding settlement charged, reversed or trued up a second time, or a
second insurance credit for one closing fill, is refused by the database ("same", or "differs" with an error event),
and the engine moves cash only for a movement the journal took. CI also runs these against Postgres."""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import schema
from sleeve_fund.paper.journal import MemoryJournal
from sleeve_fund.store import Store, funding_t, insurance_t, make_engine, metadata
from sleeve_fund.strategies.base import LongFlatStrategy

T = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/t.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    yield eng
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def store(engine):
    s = Store(engine=engine)
    s.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT-PERP", bar_spec="1-HOUR-LAST-INTERNAL",
                    starting_balance=1_000)
    return s


def _fund(j, amount=-0.6, kind="settled", ts=T):
    return j.record_funding("s", qty=0.1, price=60_000.0, rate=0.0001, amount=amount, ts=ts, kind=kind)


@pytest.mark.parametrize("journal", ["store", "memory"])
def test_a_settlement_is_charged_once_and_a_second_charge_moves_nothing(store, journal):
    j = store if journal == "store" else MemoryJournal()
    assert _fund(j) == "new"
    assert _fund(j) == "same"  # a second process, or a restart that lost _funding_since
    assert _fund(j, kind="baseline") == "differs"  # the other charge kind for the same settlement
    assert _fund(j, amount=-0.7) == "differs"
    assert _fund(j, ts=T + timedelta(hours=8)) == "new"
    assert len(j.funding("s")) == 2 and j.funding_total("s") == pytest.approx(-1.2)


@pytest.mark.parametrize("journal", ["store", "memory"])
def test_a_settlement_is_reversed_and_trued_up_at_most_once(store, journal):
    j = store if journal == "store" else MemoryJournal()
    assert _fund(j, kind="baseline") == "new"
    assert _fund(j, amount=0.6, kind="reversal") == "new"
    assert _fund(j, amount=0.6, kind="reversal") == "same"
    assert _fund(j, amount=0.1, kind="true_up") == "new"
    assert _fund(j, amount=0.2, kind="true_up") == "differs"
    assert j.funding_total("s") == pytest.approx(0.1)


def test_a_conflicting_second_charge_raises_an_error_event_and_keeps_the_first(store):
    _fund(store)
    _fund(store, amount=-0.7)
    assert [e["kind"] for e in store.events("s") if e["level"] == "error"] == ["funding_conflict"]
    assert float(store.funding("s")[0]["amount"]) == -0.6


@pytest.mark.parametrize("journal", ["store", "memory"])
def test_one_insurance_credit_per_closing_fill(store, journal):
    j = store if journal == "store" else MemoryJournal()
    credit = dict(price=50_000.0, amount=12.5, order_id="O-liq", trade_id="T-1", ts=T)
    assert j.record_insurance("s", **credit) == "new"
    assert j.record_insurance("s", **credit) == "same"
    assert j.record_insurance("s", **(credit | {"amount": 13.0})) == "differs"
    assert j.record_insurance("s", **(credit | {"trade_id": "T-2"})) == "new"
    assert j.insurance_total("s") == pytest.approx(25.0)


def test_the_database_itself_refuses_a_second_charge_or_credit(store):
    """The keys hold whoever writes: a writer that skips the check is refused."""
    row = dict(sleeve="s", ts=T, qty=0.1, price=60_000, rate=0.0001, amount=-0.6, kind="settled")
    with store.engine.begin() as c:
        c.execute(insert(funding_t).values(**row))
    for kind in ("settled", "baseline"):
        with pytest.raises(IntegrityError), store.engine.begin() as c:
            c.execute(insert(funding_t).values(**(row | {"kind": kind})))
    cr = dict(sleeve="s", ts=T, price=50_000, amount=1, order_id="O", trade_id="T")
    with store.engine.begin() as c:
        c.execute(insert(insurance_t).values(**cr))
        c.execute(insert(insurance_t).values(**(cr | {"order_id": None, "trade_id": None})))  # unkeyed, as before 0013
        c.execute(insert(insurance_t).values(**(cr | {"order_id": None, "trade_id": None})))
    with pytest.raises(IntegrityError), store.engine.begin() as c:
        c.execute(insert(insurance_t).values(**cr))


def _engine_like(store):
    return SimpleNamespace(_cash_adj=0.0, funding_log=[], funding_notes={}, _backtest=False, _funding_paid={},
                           insurance_log=[], runtime=SimpleNamespace(store=store, name="s", now=lambda: T),
                           clock=SimpleNamespace(utc_now=lambda: T), _mark=lambda: (-2.004, 0.0))


def test_the_engine_moves_cash_only_for_a_movement_the_journal_took(store):
    eng = _engine_like(store)
    for _ in range(2):  # the second: a restart that forgot the settlement, or a second process
        LongFlatStrategy._book_funding(eng, T, 0.1, 60_000.0, 0.0001, -0.6, kind="baseline")
    assert eng._cash_adj == -0.6 and len(eng.funding_log) == 1 and len(store.funding("s")) == 1
    for _ in range(2):
        LongFlatStrategy._cover_shortfall(eng, 50_000.0, ("O-liq", "T-1"))
    assert eng._cash_adj == pytest.approx(-0.6 + 2.01) and len(eng.insurance_log) == 1
    assert store.insurance("s")[0]["order_id"] == "O-liq"


def test_0013_stops_on_a_settlement_already_charged_twice_and_changes_nothing(engine):
    from alembic import command

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0012")
        conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, "
                          "risk_profile, warmup_bars, desired_state, status, status_reason, created_at, updated_at) "
                          "SELECT 's', 'x', 'BTC/USDT-PERP', 'b', '{}', 1000, 'r', 0, 'stopped', 'stopped', '', "
                          "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP"))
        for kind in ("settled", "baseline"):
            conn.execute(text("INSERT INTO funding (sleeve, ts, qty, price, rate, amount, kind) VALUES "
                              f"('s', '2026-10-08 08:00:00', 0.1, 60000, 0.0001, -0.6, '{kind}')"))
    with pytest.raises(RuntimeError, match="funding charges"):
        schema.migrate(engine, log=lambda _: None)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM funding")).scalar() == 2
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0012"


@pytest.mark.parametrize("journal", ["store", "memory"])
def test_a_check_reads_every_settlement_however_many(store, journal):
    """The restart and reversal checks read every row, not the newest 1,000 (CASH-1)."""
    j = store if journal == "store" else MemoryJournal()
    for h in range(1_100):
        _fund(j, ts=T + timedelta(hours=h))
    assert len(j.funding("s")) == 1_000  # the default read is unchanged
    every = j.funding("s", limit=None)
    assert len(every) == 1_100 and every[-1]["ts"] == T
