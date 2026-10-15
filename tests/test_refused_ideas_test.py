"""F-XA12-TESTNAME (HoE 15 Oct 03:00 UK; DA store side, FE shows it): a refusal or trim on the Strategy page names the
compliance test that made it and the tests' version it was checked under. Store.refused_ideas returns test_hit and
compliance_version with each idea, so a T1 or T2 refusal no longer reads as a bare "Refused". Both engines."""

import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert, select, text

from sleeve_fund.store import Store, gate_decisions_t, make_engine, metadata, sleeves_t

T0 = datetime(2026, 10, 15, 3, 0, tzinfo=UTC)


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
        if eng.dialect.name == "postgresql":
            for f in ("fund_flows_write_once", "fill_instrument", "compliance_write_once", "gate_test_hit_known"):
                c.execute(text(f"DROP FUNCTION IF EXISTS {f}() CASCADE"))
    metadata.drop_all(eng)


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/r.db")
    _clean(eng)
    yield Store(engine=eng)
    _clean(eng)


def _idea(store, seq, outcome, test_hit, limit_hit=None, approved=0):
    with store.engine.begin() as c:
        sid = c.execute(select(sleeves_t.c.id).where(sleeves_t.c.name == "s")).scalar()
        c.execute(insert(gate_decisions_t).values(
            seq=seq, sleeve_id=sid, bar_ts=T0, intent_id=f"i{seq}", intent="entry", underlying="BTC",
            profile_version=1, compliance_version=1 if test_hit else None, test_hit=test_hit, outcome=outcome,
            limit_hit=limit_hit, requested_qty=1, approved_qty=approved, price=60_000, stage="submit",
            decided_at=T0 + timedelta(minutes=seq)))


def test_a_t1_refusal_and_a_t2_trim_name_their_test_and_its_version(store):
    store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0)
    _idea(store, 1, "rejected", "asset_stop_loss")  # T1: loss if every stop is hit, per asset
    _idea(store, 2, "trimmed", "type_stop_loss", approved=0.5)  # T2: per asset group
    _idea(store, 3, "rejected", None, limit_hit="portfolio_state_stale")  # not a test: no name, no version
    got = [(r["outcome"], r["test_hit"], r["compliance_version"], r["limit_hit"]) for r in store.refused_ideas("s")]
    assert got == [("rejected", None, None, "portfolio_state_stale"),
                   ("trimmed", "type_stop_loss", 1, None),
                   ("rejected", "asset_stop_loss", 1, None)]
