"""CASH-2: one cash reader. strategy_cash_pnl sums every fill's cash leg less its fee, funding and insurance cover over
(since, at], uncapped and exact, and agrees with the journal's book; the dashboard's fee and fill counts read every
fill; an insurance credit that names no fill is refused. CI also runs these against Postgres."""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import func, select, text

from sleeve_fund.dashboard.metrics import sleeve_summary
from sleeve_fund.paper.journal import MemoryJournal
from sleeve_fund.store import Store, fills_t, insurance_t, make_engine, metadata

T = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)
H = timedelta(hours=1)


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    s = Store(engine=eng)
    s.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT-PERP", bar_spec="1-HOUR-LAST-INTERNAL",
                    starting_balance=1_000)
    yield s
    metadata.drop_all(eng)


def _book(j):
    """A buy, a funding charge, an insurance cover, a sell and a funding receipt, an hour apart."""
    j.record_fill("s", side="BUY", qty=0.1, price=60_000.0, fee=3.0, order_id="O-1", trade_id="T-1", ts=T)
    j.record_funding("s", qty=0.1, price=60_000.0, rate=0.0001, amount=-1.25, ts=T + H)
    j.record_insurance("s", price=50_000.0, amount=2.5, order_id="O-1", trade_id="T-1", ts=T + 2 * H)
    j.record_fill("s", side="SELL", qty=0.1, price=61_000.0, fee=3.05, order_id="O-2", trade_id="T-2", ts=T + 3 * H)
    j.record_funding("s", qty=0.1, price=61_000.0, rate=-0.0001, amount=0.75, ts=T + 4 * H)


@pytest.mark.parametrize("journal", ["store", "memory"])
def test_the_one_reader_counts_what_was_journaled_in_since_exclusive_to_at_inclusive(store, journal):
    j = store if journal == "store" else MemoryJournal()
    _book(j)
    buy, sell = D("-6003.0"), D("6096.95")
    assert j.strategy_cash_pnl("s", None, T) == buy
    assert j.strategy_cash_pnl("s", None, T + 2 * H) == buy - D("1.25") + D("2.5")
    assert j.strategy_cash_pnl("s", T, T + 2 * H) == D("1.25")  # the buy at `since` is out
    assert j.strategy_cash_pnl("s", T + 3 * H) == D("0.75")
    whole = buy - D("1.25") + D("2.5") + sell + D("0.75")
    assert j.strategy_cash_pnl("s") == whole
    for seam in (T, T + H, T + 2 * H, T + 3 * H):  # windows tile, with no row counted twice or dropped at a seam
        assert j.strategy_cash_pnl("s", None, seam) + j.strategy_cash_pnl("s", seam) == whole
    # the same figure the journal's book reaches, which the engine reconciles against
    assert D(str(j.journal_book("s", 1_000)["cash"])) - 1_000 == whole


def test_the_one_reader_and_the_dashboard_read_every_fill(store):
    n = 10_050
    with store.engine.begin() as c:
        c.execute(fills_t.insert(), [dict(sleeve="s", ts=T + timedelta(minutes=i), side="BUY" if i % 2 else "SELL",
                                          qty=D("0.001"), price=D("60000"), fee=D("0.01"), order_id=f"O-{i}",
                                          trade_id=f"T-{i}") for i in range(n)])
    assert store.strategy_cash_pnl("s") == -D("0.01") * n  # buys and sells net; every fee counts
    assert store.fill_totals("s") == (n, D("0.01") * n)
    x = sleeve_summary(store, store.sleeve("s"))
    assert x["fills"] == n and D(str(x["fees"])) == D("0.01") * n


@pytest.mark.parametrize("journal", ["store", "memory"])
@pytest.mark.parametrize("key", [dict(), dict(order_id="O-1"), dict(trade_id="T-1"), dict(order_id="", trade_id="")])
def test_an_insurance_credit_that_names_no_fill_is_refused_and_nothing_is_written(store, journal, key):
    j = store if journal == "store" else MemoryJournal()
    with pytest.raises(ValueError, match="must name the fill"):
        j.record_insurance("s", price=50_000.0, amount=2.5, ts=T, **key)
    if journal == "store":
        with store.engine.connect() as c:
            assert c.execute(select(func.count()).select_from(insurance_t)).scalar() == 0
    else:
        assert j.insurance("s") == []
