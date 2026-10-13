"""CASH-4 (Advisor 8 Oct 04:10 UK, ruling 2; migration 0016): a settlement charged the baseline is trued up once, when
the venue's rate arrives, by settled - baseline, signed as the cash moves. The row keeps the settlement's ts and moves
cash at its arrival, so an as-of read before it shows the baseline only; the baseline is never edited. Nothing is
trued up that was settled at the venue's rate or reversed. CI also runs these against Postgres. (QA's c6b cells pin the
two signs and a second arrival.)"""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import insert, inspect, select, text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import schema
from sleeve_fund.store import Store, events_t, funding_t, make_engine, metadata

S = datetime(2026, 10, 13, 8, tzinfo=timezone.utc)  # the settlement
ARRIVED = S + timedelta(hours=2)


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


@pytest.fixture
def store(engine):
    s = Store(engine=engine)
    s.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT-PERP", bar_spec="1-HOUR-LAST-INTERNAL",
                    starting_balance=1_000)
    return s


def _charge(store, kind="baseline", amount="-1.00", ts=S):
    return store.record_funding("s", qty=0.1, price=60_000.0, rate=0.0001, amount=D(amount), ts=ts, kind=kind)


def _rows(store, kind=None):
    q = select(funding_t).where(funding_t.c.sleeve == "s").order_by(funding_t.c.id)
    with store.engine.connect() as c:
        return [r for r in c.execute(q) if kind is None or r.kind == kind]


def test_the_true_up_keeps_the_baselines_position_and_price_and_moves_cash_from_its_arrival(store):
    _charge(store)
    assert store.record_true_up("s", S, rate=0.0002, amount=D("-1.20"), arrived_at=ARRIVED) == ("new", D("-0.20"))
    (base,), (up,) = _rows(store, "baseline"), _rows(store, "true_up")
    assert (D(str(up.qty)), D(str(up.price)), up.rate) == (D(str(base.qty)), D(str(base.price)), 0.0002)
    assert D(str(base.amount)) == D("-1.00")  # never edited
    assert store.funding_total("s", before=ARRIVED) == D("-1.00")
    assert store.funding_total("s", before=ARRIVED + timedelta(seconds=1)) == store.funding_total("s") == D("-1.20")
    # windows still tile at every seam, the arrival's included
    seams = [None, S - timedelta(seconds=1), S, ARRIVED - timedelta(seconds=1), ARRIVED, None]
    parts = [store.strategy_cash_pnl("s", a, b) for a, b in zip(seams, seams[1:])]
    assert parts == [D(0), D("-1.00"), D(0), D("-0.20"), D(0)] and sum(parts) == store.strategy_cash_pnl("s")


def test_nothing_is_trued_up_that_was_settled_at_the_venues_rate_or_reversed(store):
    _charge(store, kind="settled")
    assert store.record_true_up("s", S, rate=0.0002, amount=D("-1.20"), arrived_at=ARRIVED) == ("none", None)
    later = S + timedelta(hours=8)
    _charge(store, ts=later)
    _charge(store, kind="reversal", amount="1.00", ts=later)
    assert store.record_true_up("s", later, rate=0.0002, amount=D("-1.20"), arrived_at=ARRIVED) == ("none", None)
    assert store.record_true_up("s", S - timedelta(hours=8), rate=0.0002, amount=D("-1.20"),
                                arrived_at=ARRIVED) == ("none", None)  # nothing charged at all
    assert _rows(store, "true_up") == []


def test_another_figure_for_the_same_settlement_is_refused_with_an_error_and_moves_nothing(store):
    _charge(store)
    store.record_true_up("s", S, rate=0.0002, amount=D("-1.20"), arrived_at=ARRIVED)
    assert store.record_true_up("s", S, rate=0.0002, amount=D("-1.20"),
                                arrived_at=ARRIVED + timedelta(minutes=5)) == ("same", D("-0.20"))
    assert store.record_true_up("s", S, rate=0.0003, amount=D("-1.80"), arrived_at=ARRIVED) == ("differs", None)
    assert len(_rows(store, "true_up")) == 1 and store.funding_total("s") == D("-1.20")
    with store.engine.connect() as c:
        kinds = [r.kind for r in c.execute(select(events_t).where(events_t.c.sleeve == "s"))]
    assert kinds.count("funding_conflict") == 1


def test_only_a_true_up_carries_an_arrival_time_and_the_database_holds_to_it(store):
    with pytest.raises(ValueError, match="only a true-up"):
        store.record_funding("s", qty=0.1, price=60_000.0, rate=0.0001, amount=-1.0, ts=S, kind="baseline",
                             arrived_at=ARRIVED)
    row = dict(sleeve="s", ts=S, qty=D("0.1"), price=D("60000"), rate=0.0001, amount=D("-1"))
    for bad in (dict(kind="true_up", arrived_at=None), dict(kind="settled", arrived_at=ARRIVED)):
        with pytest.raises(IntegrityError), store.engine.begin() as c:
            c.execute(insert(funding_t).values(**row, **bad))
    assert _rows(store) == []


def test_0016_stops_on_a_true_up_it_cannot_give_an_arrival_and_changes_nothing(engine):
    from alembic import command

    with engine.begin() as conn:
        command.upgrade(schema._config(conn), "0013")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, starting_balance, created_at, "
                          "updated_at, desired_state, status, status_reason, risk_profile, params, warmup_bars) "
                          "SELECT 's', 'buy_and_hold', 'BTC/USDT-PERP', 'b', 1000, :t, :t, 'stopped', 'stopped', '', "
                          "'balanced', '{}', 0"), {"t": S})
        conn.execute(text("INSERT INTO funding (sleeve, ts, qty, price, rate, amount, kind) VALUES "
                          "('s', :t, 0.1, 60000, 0.0001, 0.25, 'true_up')"), {"t": S})
    with pytest.raises(Exception, match="no arrival time"):
        schema.migrate(engine, log=lambda _: None)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0013"
        assert conn.execute(text("SELECT COUNT(*) FROM funding")).scalar() == 1
    assert "arrived_at" not in {c["name"] for c in inspect(engine).get_columns("funding")}
    with engine.begin() as conn:  # leave it migrated, so the teardown drops the whole shape
        conn.execute(text("DELETE FROM funding"))
    schema.migrate(engine, log=lambda _: None)


def test_0016_migrated_funding_equals_create_alls(engine, tmp_path):
    def shape(eng):
        i = inspect(eng)
        return (sorted((c["name"], c["nullable"]) for c in i.get_columns("funding")),
                sorted(c["name"] for c in i.get_check_constraints("funding")),
                sorted((x["name"], tuple(x["column_names"]), bool(x["unique"])) for x in i.get_indexes("funding")))

    schema.migrate(engine, log=lambda _: None)
    migrated = shape(engine)
    fresh = make_engine(f"sqlite:///{tmp_path}/fresh.db") if engine.dialect.name == "sqlite" else engine
    if fresh is engine:
        _clean(engine)
    Store(engine=fresh)
    assert migrated == shape(fresh)
    assert ("arrived_at", True) in migrated[0] and "funding_true_up_arrived" in migrated[1]


def test_a_stopped_archived_strategy_still_books_a_true_up_that_arrives_after_it(store):
    """HoE / PE1 (P2-6): retire stops the process but the journal stays writable; nothing in the writer looks at
    the strategy's state."""
    _charge(store)
    store.set_desired_state("s", "stopped")
    store.archive("s")
    assert store.record_true_up("s", S, rate=0.0002, amount=D("-1.20"), arrived_at=ARRIVED) == ("new", D("-0.20"))
    assert store.funding_total("s") == D("-1.20")


def test_open_baselines_are_those_neither_trued_up_nor_reversed(store):
    for h in (0, 8, 16):
        _charge(store, ts=S + timedelta(hours=h))
    _charge(store, kind="settled", ts=S + timedelta(hours=24))
    store.record_true_up("s", S, rate=0.0002, amount=D("-1.20"), arrived_at=ARRIVED)
    _charge(store, kind="reversal", amount="1.00", ts=S + timedelta(hours=8))
    assert [r["ts"].replace(tzinfo=timezone.utc) for r in store.open_baselines("s")] == [S + timedelta(hours=16)]


@pytest.fixture
def stored_rates(monkeypatch):
    """The collector's stored rates for the strategy's instrument, and terms naming a venue that publishes them."""
    import pandas as pd

    from sleeve_fund import funding, markets

    kept = {}
    monkeypatch.setattr(markets, "terms", lambda params, venue: type("T", (), {"funding_venue": "BINANCE"})())
    monkeypatch.setattr(funding, "rates", lambda venue, pair, root=None: pd.Series(
        list(kept.values()), index=pd.DatetimeIndex(list(kept), tz="UTC"), dtype=float))
    return kept


def test_with_no_process_the_supervisor_trues_up_once_the_rate_is_stored(store, stored_rates):
    from sleeve_fund.supervisor import Proc, Supervisor, true_up_stopped

    _charge(store)  # -1.00 at the baseline, on 0.1 at 60,000
    store.set_desired_state("s", "stopped")
    (s,) = store.sleeves()
    assert true_up_stopped(store, s, ARRIVED) == 0  # not published yet: the baseline stays
    stored_rates[S] = 0.0002
    sup = Supervisor(store)
    sup.procs["s"] = type("Alive", (Proc,), {"alive": property(lambda self: True)})()
    sup.true_up_stopped()
    assert _rows(store, "true_up") == []  # a running strategy trues up its own
    sup.procs.clear()
    sup.true_up_stopped()
    (up,) = _rows(store, "true_up")
    assert D(str(up.amount)) == D("-1.20") - D("-1.00") and up.rate == 0.0002
    assert true_up_stopped(store, s, ARRIVED) == 0 and len(_rows(store, "true_up")) == 1  # once
