"""Drop B, data lane (DA, migration 0028; PE1 multi-asset-spec, Advisor MA-SPEC): a multi-asset Strategy's legs are rows
"<name>~<ASSET>" under one parent, one row per (Strategy, asset); its book per asset sums exactly to the Strategy's;
reconcile names a journal row keyed to the wrong asset; a decision carries the idea (planned stop and target, side,
the bound test's measure); a trimmed or refused idea's outcome is written once. Both engines: CI runs these against
Postgres via TEST_DATABASE_URL."""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from sleeve_fund import schema
from sleeve_fund.store import Store, fills_t, gate_decisions_t, make_engine, metadata, sleeves_t

T0 = datetime(2026, 10, 15, 1, 0, tzinfo=timezone.utc)
REFUSED = (IntegrityError, DBAPIError)
PG_FUNCTIONS = ("fund_flows_write_once", "fill_instrument", "compliance_write_once", "gate_test_hit_known",
                "gate_idea_shape", "shadow_outcome_write_once", "shadow_outcome_kind", "trials_write_once")


def _clean(eng):
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        if eng.dialect.name == "postgresql":
            for f in PG_FUNCTIONS:
                conn.execute(text(f"DROP FUNCTION IF EXISTS {f}() CASCADE"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/assets.db")
    _clean(eng)
    yield eng
    _clean(eng)


@pytest.fixture
def store(engine):
    return Store(engine=engine)


def _row(conn, name, instrument="BTC/USDT", parent=None, balance=1000):
    conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, risk_profile,"
                      " warmup_bars, desired_state, status, status_reason, created_at, updated_at, parent) VALUES (:n, "
                      "'buy_and_hold', :i, '1-MINUTE-LAST-EXTERNAL', '{}', :b, 'balanced', 0, 'stopped', 'stopped', '', "
                      ":t, :t, :p)"), {"n": name, "i": instrument, "b": balance, "t": T0, "p": parent})


def _trend(engine):
    """A three-asset Strategy: the parent on BTC, legs on ETH and SOL (made in the definition's order)."""
    with engine.begin() as c:
        _row(c, "trend", "BTC/USDT", balance=1000)
        _row(c, "trend~ETH", "ETH/USDT", "trend", balance=500)
        _row(c, "trend~SOL", "SOL/USDT", "trend", balance=250)


def _refused(engine, name, instrument, parent):
    with pytest.raises(REFUSED), engine.begin() as c:
        _row(c, name, instrument, parent)


# --- legs ---------------------------------------------------------------------------------------------------------


def test_a_multi_asset_strategy_has_one_leg_row_per_asset_and_a_two_leg_carry_still_fits(store):
    _trend(store.engine)
    with store.engine.begin() as c:
        _row(c, "carry", "BTC/USDT")
        _row(c, "carry~2", "BTC/USDT", "carry")  # spot and perp on one symbol: the "~2" row is outside the asset key
        _row(c, "solo", "BTC/USDT")
        _row(c, "solo~1000PEPE", "1000PEPE/USDT", "solo")  # digits in the suffix
    with store.engine.connect() as c:
        assert c.execute(select(func.count()).select_from(sleeves_t)
                         .where(sleeves_t.c.parent == "trend")).scalar() == 2


@pytest.mark.parametrize("name,instrument,parent,why", [
    ("trend~eth", "ETH/USDT", "trend", "lower case"),
    ("trend~E-TH", "ETH/USDT", "trend", "punctuation"),
    ("trend~", "ETH/USDT", "trend", "no asset"),
    ("trend~ABCDEFGHIJKLMNOPQ", "ETH/USDT", "trend", "17 characters"),
    ("trend~ETH~ADA", "ADA/USDT", "trend~ETH", "a leg under a leg"),
    ("other~ADA", "ADA/USDT", "trend", "named for another Strategy"),
    ("tren~ADA", "ADA/USDT", "trend", "a prefix of its parent"),
    ("top~ADA", "ADA/USDT", None, "a '~' with no parent"),
    ("trend~BTC", "BTC/USDT", "trend", "the parent's asset"),
    ("trend~ETH2", "ETH/USDT", "trend", "another leg's asset"),
])
def test_the_database_refuses_every_other_leg_shape(store, name, instrument, parent, why):
    _trend(store.engine)
    _refused(store.engine, name, instrument, parent)


def test_a_parent_name_holding_like_wildcards_matches_only_itself(store):
    with store.engine.begin() as c:
        _row(c, "a_b", "BTC/USDT")
        _row(c, "axb", "ETH/USDT")
        _row(c, "a_b~ETH", "ETH/USDT", "a_b")
    _refused(store.engine, "axb~SOL", "SOL/USDT", "a_b")  # "_" in the parent is a character, not a wildcard


# --- readers: (Strategy, asset) -----------------------------------------------------------------------------------


def test_strategy_assets_lists_the_parent_then_its_legs_from_any_row(store):
    _trend(store.engine)
    with store.engine.begin() as c:
        _row(c, "single", "ETH/USDT")
    want = [("trend", "BTC/USDT", "BTC", False), ("trend~ETH", "ETH/USDT", "ETH", True),
            ("trend~SOL", "SOL/USDT", "SOL", True)]
    for name in ("trend", "trend~SOL"):
        got = store.strategy_assets(name)
        assert [(a["sleeve"], a["instrument"], a["underlying"], a["leg"]) for a in got] == want
        assert {a["strategy"] for a in got} == {"trend"}
    assert [a["sleeve"] for a in store.strategy_assets("single")] == ["single"]
    with pytest.raises(KeyError):
        store.strategy_assets("nobody")


def _fills(store):
    fills = [("trend", "BUY", "0.013", "61234.57", "0.79"), ("trend", "SELL", "0.013", "61891.03", "0.81"),
             ("trend~ETH", "SELL", "0.37", "2411.19", "0.3567"), ("trend~ETH", "BUY", "0.12", "2400.01", "0.1153"),
             ("trend~SOL", "BUY", "3.3", "141.237", "0.0933")]
    for n, (sleeve, side, qty, price, fee) in enumerate(fills):
        store.record_fill(sleeve, side=side, qty=Decimal(qty), price=Decimal(price), fee=Decimal(fee),
                          order_id=f"o{n}", trade_id=f"t{n}", ts=T0 + timedelta(minutes=n))


def test_the_book_per_asset_sums_exactly_to_the_strategys_and_matches_the_fills_by_asset(store):
    """Done-when 6 (multi-asset-spec): per-asset cash, fees and funding add up to the Strategy's to the last digit, and
    each asset's figures are what the journal holds under (Strategy, instrument)."""
    _trend(store.engine)
    _fills(store)
    book = store.asset_book("trend~ETH")
    assert book["strategy"] == "trend" and [a["instrument"] for a in book["assets"]] == \
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    for k in ("starting_balance", "cash", "fees", "funding", "insurance"):
        assert book["total"][k] == sum((a[k] for a in book["assets"]), Decimal(0))
    assert book["total"]["starting_balance"] == Decimal(1750) and book["total"]["fills"] == 5
    with store.engine.connect() as c:
        by_asset = dict(c.execute(
            select(fills_t.c.instrument, func.count())
            .select_from(fills_t.join(sleeves_t, sleeves_t.c.name == fills_t.c.sleeve))
            .where(func.coalesce(sleeves_t.c.parent, sleeves_t.c.name) == "trend")
            .group_by(fills_t.c.instrument)).all())
    assert by_asset == {a["instrument"]: a["fills"] for a in book["assets"]}
    eth = book["assets"][1]
    # cash from the journal: 1000/2 + 0.37 x 2411.19 - 0.3567 - 0.12 x 2400.01 - 0.1153
    assert eth["cash"] == Decimal("500") + Decimal("0.37") * Decimal("2411.19") - Decimal("0.3567") \
        - Decimal("0.12") * Decimal("2400.01") - Decimal("0.1153")
    assert eth["qty"] == Decimal("-0.25") and eth["fees"] == Decimal("0.472")
    assert book["assets"][0]["qty"] == 0 and book["total"]["fees"] == Decimal("2.1653")


def test_reconcile_names_a_journal_row_keyed_to_the_wrong_asset(store):
    _trend(store.engine)
    _fills(store)
    assert store.asset_mismatches() == []
    with store.engine.begin() as c:
        c.execute(text("UPDATE fills SET instrument = 'SOL/USDT' WHERE order_id = 'o2'"))
    got = store.asset_mismatches()
    assert [(m["table"], m["sleeve"], m["instrument"], m["expected"]) for m in got] == \
        [("fills", "trend~ETH", "SOL/USDT", "ETH/USDT")]


# --- the idea on its decision, and its shadow outcome -------------------------------------------------------------


def _decision(eng, n, outcome="rejected", test_hit="asset_stop_loss", sleeve="trend~ETH", **idea):
    with eng.begin() as c:
        sid = c.execute(select(sleeves_t.c.id).where(sleeves_t.c.name == sleeve)).scalar()
        return c.execute(insert(gate_decisions_t).values(
            seq=n, sleeve_id=sid, bar_ts=T0, intent_id=f"i{n}", intent="entry", underlying="ETH", profile_version=1,
            compliance_version=1, test_hit=test_hit, outcome=outcome, requested_qty=1,
            approved_qty=0 if outcome == "rejected" else Decimal("0.5"), price=2400, stage="submit", decided_at=T0,
            **idea)).inserted_primary_key[0]


def test_a_decision_carries_the_idea_and_the_database_holds_its_shape(store):
    _trend(store.engine)
    _decision(store.engine, 1, planned_stop=Decimal("2350"), planned_target=Decimal("2500"), side=-1,
              would_be=Decimal("0.0412"))
    _decision(store.engine, 2, outcome="approved", test_hit=None)  # rows written before the gate writes them
    for n, idea in enumerate(({"side": 0}, {"side": 2}, {"planned_stop": 0}, {"planned_target": Decimal("-1")}),
                             start=3):
        with pytest.raises(REFUSED):
            _decision(store.engine, n, **idea)
    with pytest.raises(REFUSED):
        _decision(store.engine, 9, outcome="approved", test_hit=None, would_be=Decimal("0.01"))  # a measure, no test
    with pytest.raises(REFUSED), store.engine.begin() as c:
        c.execute(text("UPDATE gate_decisions SET side = 5 WHERE seq = 1"))


def test_a_shadow_outcome_is_written_once_and_matches_its_decision(store):
    _trend(store.engine)
    refused = _decision(store.engine, 1, planned_stop=Decimal("2350"), planned_target=Decimal("2500"), side=1)
    trimmed = _decision(store.engine, 2, outcome="trimmed", test_hit="asset_net")
    approved = _decision(store.engine, 3, outcome="approved", test_hit=None)
    assert store.record_shadow_outcome(refused, kind="refusal", pnl=Decimal("-12.5"), method="replay",
                                       computed_at=T0) == "new"
    assert store.record_shadow_outcome(refused, kind="refusal", pnl=Decimal("-12.50"), method="replay") == "same"
    with pytest.raises(ValueError, match="never rewritten"):
        store.record_shadow_outcome(refused, kind="refusal", pnl=Decimal("3"), method="replay")
    assert store.record_shadow_outcome(trimmed, kind="trim", pnl=Decimal("7.25"), method="trade_scaled") == "new"
    for decision, kind, method in ((approved, "trim", "trade_scaled"), (refused + 100, "refusal", "replay")):
        with pytest.raises(REFUSED):  # an approved idea has no shadow; nor does a decision that doesn't exist
            store.record_shadow_outcome(decision, kind=kind, pnl=1, method=method)
    for seq, (kind, method, hypothetical) in enumerate((("trim", "replay", False), ("refusal", "trade_scaled", True),
                                                         ("refusal", "replay", False)), start=10):
        n = _decision(store.engine, seq)
        with pytest.raises(REFUSED), store.engine.begin() as c:
            c.execute(text("INSERT INTO shadow_outcomes VALUES (:d, :k, 1, :h, :m, :t)"),
                      {"d": n, "k": kind, "h": hypothetical, "m": method, "t": T0})
    for sql in ("UPDATE shadow_outcomes SET pnl = 0", "DELETE FROM shadow_outcomes"):
        with pytest.raises(REFUSED), store.engine.begin() as c:
            c.execute(text(sql))
    got = store.shadow_outcomes("trend")
    by_id = {r["decision_id"]: r for r in got}
    assert by_id[refused]["kind"] == "refusal" and by_id[refused]["pnl"] == Decimal("-12.5")
    assert by_id[refused]["planned_stop"] == Decimal("2350") and by_id[refused]["side"] == 1
    assert by_id[trimmed]["kind"] == "trim" and by_id[trimmed]["hypothetical"] is False
    assert approved not in by_id and {r["strategy"] for r in got} == {"trend"}
    assert store.shadow_outcomes("other") == []


# --- migration 0028 -----------------------------------------------------------------------------------------------


def _at(eng, rev):
    from alembic import command

    with eng.begin() as conn:
        command.upgrade(schema._config(conn), rev)


def _down(eng, rev):
    from alembic import command

    with eng.begin() as conn:
        command.downgrade(schema._config(conn), rev)


def test_0028_keeps_every_row_widens_the_legs_and_matches_the_code(engine):
    _at(engine, "0027")
    with engine.begin() as c:
        _row(c, "carry", "BTC/USDT")
        _row(c, "carry~2", "BTC/USDT", "carry")
        _row(c, "trend", "BTC/USDT")
    with pytest.raises(REFUSED), engine.begin() as c:
        _row(c, "trend~ETH", "ETH/USDT", "trend")  # before 0028: "~2" only
    _at(engine, "0028")
    with engine.connect() as c:
        assert c.execute(text("SELECT name, parent FROM sleeves ORDER BY id")).all() == \
            [("carry", None), ("carry~2", "carry"), ("trend", None)]
    with engine.begin() as c:
        _row(c, "trend~ETH", "ETH/USDT", "trend")
    _refused(engine, "trend~BTC", "BTC/USDT", "trend")
    _refused(engine, "trend~eth", "ETH/USDT", "trend")
    if schema.head() == "0028":
        assert schema.report(engine)[1] == []


def test_0028_on_a_database_a_newer_store_already_opened_changes_nothing(engine):
    _at(engine, "0027")
    Store(engine=engine)
    _at(engine, "0028")
    with engine.begin() as c:
        _row(c, "trend", "BTC/USDT")
        _row(c, "trend~ETH", "ETH/USDT", "trend")
    _refused(engine, "trend~SOL", "ETH/USDT", "trend")


def test_0028_goes_down_only_while_nothing_uses_it(engine):
    _at(engine, "0028")
    with engine.begin() as c:
        _row(c, "carry", "BTC/USDT")
        _row(c, "carry~2", "BTC/USDT", "carry")
    _down(engine, "0027")
    _refused(engine, "carry~ETH", "ETH/USDT", "carry")
    _at(engine, "0028")
    with engine.begin() as c:
        _row(c, "carry~ETH", "ETH/USDT", "carry")
    with pytest.raises(NotImplementedError, match="never dropped"):
        _down(engine, "0027")
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM sleeves")).scalar() == 3
