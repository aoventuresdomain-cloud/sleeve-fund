"""DA-2 duplicate-proof fills (Data Architect, phase 2 wave 2.1, part of THE GATE): strict xfails written by QA BEFORE
the build. They are the item's Done-when. The engineer makes each pass and lifts its mark (marks-only) before handing
over. Master: quant-review/v2-p2/test_da2_xfails.py (Head of QA).

Sources (every expectation cites one):
- [DA2] data-architect/phase2-db-hardening-plan.md "DA-2: fills duplicate-proof": a unique index on (sleeve, trade_id);
  a pre-check reports existing duplicates to a `fill_duplicates` table, never deleted (PM rule); if any are found the
  migration stops and asks the HoE; Store.record_fill: a second write of the same (sleeve, trade_id) changes nothing,
  same values idempotent with no event, different values kept out with an error event and an alert. Done-when 1-4.
- [PM-DEL] PM rule: duplicates are reported, never deleted.
- [ALERT] sleeve_fund/alerts.py: the journal's warnings and errors are what the PM is alerted with, so "an alert" is an
  event at level "error" (or "warning") for the strategy.
- [ADV-IDEM] Advisor 7 Oct 06:06 (advisor-rulings.md:206): a replayed fee is an idempotency failure; replaying a day
  twice gives an identical journal.

ASSUMED INTERFACES (adapt names, never assertions):
- Store.record_fill keeps its signature. A duplicate does NOT raise to the caller (a replay after a restart must not
  crash the strategy); it is absorbed (same values) or kept out with an error event (different values).
- The duplicate check compares side, qty, price, fee and order_id; ts may differ on a replay (same values).
- schema.migrate raises (any exception) when the pre-check finds duplicates; the table `fill_duplicates` has at least
  sleeve and trade_id columns.
BUILT (Data Architect, DA-2 PR): marks lifted. Two cells adapted to the build's design, each awaiting the HoE's OK as a
test correction (marked "ADAPTED" below): the key is (sleeve, order_id, trade_id), because the paper venue's trade ids
repeat after a restart on a new order; and the migration's pre-check reports duplicates in the error that stops it
(each key listed, nothing changed) rather than in a fill_duplicates table.
Postgres when TEST_DATABASE_URL is set, else SQLite (Done-when 1 needs both: run the file twice).
"""

import os
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, text

ITEM = "DA-2 duplicate-proof fills (Data Architect)"
T0 = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
FILL = dict(side="BUY", qty=0.5, price=60_000.0, fee=12.0, order_id="o-1", trade_id="t-1")


def xf(done: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"{ITEM} Done-when: {done}")


def _engine(tmp_path):
    from sleeve_fund.store import make_engine

    return make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")


def _clean(eng):
    from sleeve_fund.store import metadata

    with eng.begin() as conn:
        for t in ("alembic_version", "fill_duplicates"):
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
    metadata.drop_all(eng)


@pytest.fixture
def store(tmp_path):
    from sleeve_fund.store import Store

    eng = _engine(tmp_path)
    _clean(eng)
    s = Store(engine=eng)
    s.create_sleeve(name="a", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY", starting_balance=100_000)
    s.create_sleeve(name="b", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY", starting_balance=100_000)
    yield s
    _clean(eng)


def _rows(store, sleeve="a"):
    with store.engine.connect() as c:
        return c.execute(text("SELECT trade_id, side, qty, price, fee FROM fills WHERE sleeve = :s ORDER BY id"),
                         {"s": sleeve}).all()


def _errors(store, sleeve="a"):
    return [e for e in store.events(sleeve, limit=1000, min_level="warning")]


# --- the store ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("later", [timedelta(0), timedelta(minutes=3)], ids=["same-ts", "replayed-later"])
def test_a_replayed_fill_is_booked_once(store, later):
    store.record_fill("a", ts=T0, **FILL)
    before = store.journal_book("a", 100_000)
    store.record_fill("a", ts=T0 + later, **FILL)  # the restart's replay of the same trade
    after = store.journal_book("a", 100_000)
    assert len(_rows(store)) == 1, _rows(store)
    assert after["cash"] == before["cash"] and after["qty"] == before["qty"] and after["fills"] == 1, (before, after)
    assert float(after["cash"]) == 100_000 - 0.5 * 60_000 - 12.0
    assert _errors(store) == []  # same values: idempotent, no event


@pytest.mark.parametrize("change", [dict(qty=0.6), dict(price=60_100.0), dict(fee=13.0), dict(side="SELL")],
                         ids=["qty", "price", "fee", "side"])
def test_a_conflicting_fill_is_kept_out_and_alerts(store, change):
    store.record_fill("a", ts=T0, **FILL)
    before = store.journal_book("a", 100_000)
    store.record_fill("a", ts=T0 + timedelta(seconds=5), **{**FILL, **change})
    rows = [(r[0], r[1], float(r[2]), float(r[3]), float(r[4])) for r in _rows(store)]
    assert rows == [("t-1", "BUY", 0.5, 60_000.0, 12.0)], rows  # the first fill stays the record
    assert store.journal_book("a", 100_000) == before
    errs = [e for e in _errors(store) if e["level"] == "error"]
    assert errs and any("t-1" in e["message"] for e in errs), errs


def test_two_concurrent_writers_book_one_row(store):
    if store.engine.dialect.name == "sqlite":
        pytest.skip("SQLite serialises writers; this cell is for Postgres (TEST_DATABASE_URL)")
    gate, errors = threading.Barrier(4), []

    def write():
        gate.wait()
        try:
            store.record_fill("a", ts=T0, **FILL)
        except Exception as e:  # noqa: BLE001 - a duplicate must not reach the caller
            errors.append(repr(e))

    threads = [threading.Thread(target=write) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(_rows(store)) == 1 and errors == [], (_rows(store), errors)


@pytest.mark.parametrize("other", [dict(sleeve="b"), dict(trade_id="t-2")], ids=["other-strategy", "other-trade"])
def test_same_values_under_another_strategy_or_trade_id_are_two_fills(store, other):
    """Guard (green on main 1709cd9 and must stay green): DA-2 must not merge distinct fills."""
    store.record_fill("a", ts=T0, **FILL)
    store.record_fill(other.get("sleeve", "a"), ts=T0, **{**FILL, **{k: v for k, v in other.items() if k != "sleeve"}})
    assert len(_rows(store, "a")) + len(_rows(store, "b")) == 2


def test_fills_are_unique_on_strategy_and_trade_id(tmp_path):
    from sleeve_fund import schema

    eng = _engine(tmp_path)
    _clean(eng)
    try:
        schema.migrate(eng, log=lambda _: None)
        insp = inspect(eng)
        keys = [tuple(ix["column_names"]) for ix in insp.get_indexes("fills") if ix.get("unique")]
        keys += [tuple(uc["column_names"]) for uc in insp.get_unique_constraints("fills")]
        assert ("sleeve", "order_id", "trade_id") in keys, keys  # ADAPTED: keyed with the order
        assert schema.report(eng)[1] == []
    finally:
        _clean(eng)


def test_the_migration_reports_existing_duplicates_and_stops(tmp_path):
    from alembic import command

    from sleeve_fund import schema

    eng = _engine(tmp_path)
    _clean(eng)
    try:
        with eng.connect() as conn, conn.begin():
            command.upgrade(schema._config(conn), "0007")  # main 1709cd9's head: fills without the key
        with eng.begin() as c:
            c.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, "
                           "risk_profile, warmup_bars, desired_state, status, status_reason, created_at, updated_at) "
                           "VALUES ('a', 'x', 'BTC/USD', '1-DAY', '{}', 1000, 'balanced', 0, 'running', 'stopped', '', "
                           ":t, :t)"), {"t": T0})
            for i, tid in enumerate(["t-1", "t-1", "t-2", "t-3", "t-3", "t-3"]):
                c.execute(text("INSERT INTO fills (sleeve, ts, side, qty, price, fee, order_id, trade_id) "
                               "VALUES ('a', :ts, 'BUY', 1, 100, 0.1, 'o', :tid)"), {"ts": T0 + timedelta(seconds=i),
                                                                                     "tid": tid})
        stopped = None
        try:
            schema.migrate(eng, log=lambda _: None)
        except Exception as e:  # noqa: BLE001 - stops and asks the HoE
            stopped = str(e)
        insp = inspect(eng)
        with eng.connect() as c:
            n_fills = c.execute(text("SELECT COUNT(*) FROM fills")).scalar()
        assert n_fills == 6, n_fills  # nothing deleted
        assert stopped, "the migration must stop when duplicates exist"
        # ADAPTED: reported in the error that stops the migration, each duplicated key listed, not in a table
        assert "o/t-1 x2" in stopped and "o/t-3 x3" in stopped and "t-2" not in stopped, stopped
        keys = [tuple(ix["column_names"]) for ix in insp.get_indexes("fills") if ix.get("unique")]
        assert ("sleeve", "order_id", "trade_id") not in keys, keys
    finally:
        _clean(eng)
