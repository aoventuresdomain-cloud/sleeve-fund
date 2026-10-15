"""Bridge A, data lane (DA spec bridge-a-data-spec.md 3-4, migration 0027; Advisor BRIDGE-3, PM DEFAULTS-2): the PM's
compliance tests are versioned whole with the asset-group map, written only through the PM's checked write, never
edited, and every gate decision can name the test of the version it recorded. Both engines: CI runs these against
Postgres via TEST_DATABASE_URL."""

import os
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from sleeve_fund import compliance_rules as rules, schema
from sleeve_fund.store import (Store, compliance_asset_types_t, compliance_tests_t, compliance_versions_t, decisions_t,
                               make_engine, metadata)

T0 = datetime(2026, 10, 14, 22, 0, tzinfo=timezone.utc)
REFUSED = (IntegrityError, DBAPIError)
V1 = {"asset_stop_loss": "0.04", "type_stop_loss": "0.06", "fund_stop_loss": "0.10", "fund_gross": "1.5",
      "asset_net": "0.5", "fund_margin": "0.5", "fund_day_loss": "0.03", "fund_drawdown": "0.15"}
ACTIONS = {"fund_day_loss": "pause_day", "fund_drawdown": "halt"}


def _clean(eng):
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        if eng.dialect.name == "postgresql":
            for f in ("fund_flows_write_once", "fill_instrument", "compliance_write_once", "gate_test_hit_known"):
                conn.execute(text(f"DROP FUNCTION IF EXISTS {f}() CASCADE"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield eng
    _clean(eng)


@pytest.fixture
def store(engine):
    return Store(engine=engine)


def _edit(c, **limits):
    tests = [dict(t) for t in c.tests]
    for t in tests:
        if t["test"] in limits:
            v = limits[t["test"]]
            if isinstance(v, bool):
                t["enabled"] = v
            else:
                t["limit"] = Decimal(v)
    return tests


def _count(eng, t):
    with eng.connect() as c:
        return c.execute(select(func.count()).select_from(t)).scalar()


# --- version 1 ----------------------------------------------------------------------------------------------------


def test_version_1_is_the_pms_defaults_every_test_on(store):
    c = store.compliance()
    assert c.version == 1 and c.created_by == "migration 0027"
    assert {t["test"]: (t["limit"], t["enabled"]) for t in c.tests} == {k: (Decimal(v), True) for k, v in V1.items()}
    assert {t["test"]: t["action"] for t in c.tests} == {k: ACTIONS.get(k, "trim") for k in V1}
    assert c.asset_types == {"BTC": "large", "ETH": "large", "SOL": "mid", "XRP": "mid"}
    assert rules.validate(c.tests) == []
    assert [v["version"] for v in store.compliance_versions()] == [1]


# --- the PM's write -----------------------------------------------------------------------------------------------


def test_an_edit_is_a_whole_new_version_journaled_old_to_new_and_v1_stays(store):
    c1 = store.compliance()
    v = store.add_compliance_version(_edit(c1, asset_stop_loss="0.03", asset_net=False),
                                     c1.asset_types | {"DOGE": "small"}, expected_version=1, note="tighter", now=T0)
    assert v == 2 and store.compliance().version == 2 and store.compliance(1) == c1
    t = {x["test"]: x for x in store.compliance().tests}
    assert t["asset_stop_loss"]["limit"] == Decimal("0.03") and t["asset_net"]["enabled"] is False
    assert store.compliance().asset_types["DOGE"] == "small"
    with store.engine.connect() as c:
        d = c.execute(select(decisions_t).where(decisions_t.c.action == "compliance_version")).one()
    assert d.actor == "pm" and "asset_stop_loss 0.0400 -> 0.03" in d.reason and "asset_net off" in d.reason
    assert "DOGE group none -> small" in d.reason
    assert [x["version"] for x in store.compliance_versions(limit=1)] == [2]


def test_a_stale_form_is_refused_and_writes_nothing(store):
    c1 = store.compliance()
    store.add_compliance_version(_edit(c1, asset_stop_loss="0.03"), c1.asset_types, expected_version=1, note="a")
    with pytest.raises(ValueError, match="the compliance tests changed since this page was opened"):
        store.add_compliance_version(_edit(c1, asset_stop_loss="0.02"), c1.asset_types, expected_version=1, note="b")
    assert _count(store.engine, compliance_versions_t) == 2


@pytest.mark.parametrize("edit, words", [
    ({"fund_stop_loss": False}, "can't be switched off"),
    ({"fund_day_loss": False}, "can't be switched off"),
    ({"fund_drawdown": False}, "can't be switched off"),
    ({"fund_stop_loss": "0.11"}, "between 1% and 10%"),
    ({"fund_day_loss": "0.06"}, "between 1% and 5%"),
    ({"fund_drawdown": "0.04"}, "between 5% and 25%"),
    ({"asset_stop_loss": "0.06"}, "between"),
    ({"asset_stop_loss": "0.03125"}, "four decimal places"),
])
def test_an_edit_outside_the_bounds_is_refused_in_the_pms_words(store, edit, words):
    c1 = store.compliance()
    with pytest.raises(ValueError, match=words):
        store.add_compliance_version(_edit(c1, **edit), c1.asset_types, expected_version=1, note="x")
    assert _count(store.engine, compliance_versions_t) == 1 and _count(store.engine, decisions_t) == 0


def test_a_test_cant_be_dropped_reshaped_or_written_by_anyone_but_the_pm(store):
    c1 = store.compliance()
    with pytest.raises(ValueError, match="missing"):
        store.add_compliance_version(c1.tests[1:], c1.asset_types, expected_version=1, note="x")
    tests = _edit(c1)
    tests[0]["action"] = "refuse"
    with pytest.raises(ValueError, match="can't change"):
        store.add_compliance_version(tests, c1.asset_types, expected_version=1, note="x")
    with pytest.raises(ValueError, match="say why"):
        store.add_compliance_version(_edit(c1), c1.asset_types, expected_version=1, note=" ")
    with pytest.raises(ValueError, match="PM only"):
        store.add_compliance_version(_edit(c1), c1.asset_types, expected_version=1, note="x", created_by="qd")
    with pytest.raises(ValueError, match="group name"):
        store.add_compliance_version(_edit(c1), c1.asset_types | {"ADA": ""}, expected_version=1, note="x")
    assert _count(store.engine, compliance_versions_t) == 1


@pytest.mark.parametrize("table", ["compliance_versions", "compliance_tests", "compliance_asset_types"])
@pytest.mark.parametrize("sql", ["DELETE FROM {t}", "UPDATE {t} SET version = version"])
def test_a_version_is_never_edited_or_deleted_at_the_database(store, table, sql):
    with pytest.raises(REFUSED), store.engine.begin() as c:
        c.execute(text(sql.format(t=table)))
    assert store.compliance().version == 1 and len(store.compliance().tests) == 8


def test_the_database_refuses_a_pm_version_without_a_reason_and_rows_outside_the_rules(store):
    for values in ({"version": 2, "created_at": T0, "created_by": "pm", "note": None},
                   {"version": 2, "created_at": T0, "created_by": "pm", "note": ""}):
        with pytest.raises(REFUSED), store.engine.begin() as c:
            c.execute(compliance_versions_t.insert().values(**values))
    row = {"version": 1, "test": "x", "label": "x", "scope": "fund", "measure": "gross", "limit_value": 1,
           "action": "trim", "enabled": True}
    for bad in ({"scope": "desk"}, {"measure": "vibes"}, {"action": "pause"}, {"limit_value": 0}):
        with pytest.raises(REFUSED), store.engine.begin() as c:
            c.execute(compliance_tests_t.insert().values(**(row | bad)))
    with pytest.raises(REFUSED), store.engine.begin() as c:
        c.execute(compliance_asset_types_t.insert().values(version=1, underlying="ada", asset_type="small"))


def test_an_underlying_with_no_group_is_named(store):
    store.create_sleeve(name="d", strategy="ma_cross", instrument="DOGE/USDT", bar_spec="1h", starting_balance=0)
    store.create_sleeve(name="b", strategy="ma_cross", instrument="XBT/USD", bar_spec="1h", starting_balance=0)
    assert store.unmapped() == ["DOGE"]


# --- decisions name the test of their version ---------------------------------------------------------------------


def _decision(eng, version, test_hit, n=1):
    with eng.begin() as c:
        c.execute(text("INSERT INTO gate_decisions (seq, sleeve_id, bar_ts, intent_id, intent, underlying, "
                       "profile_version, compliance_version, test_hit, outcome, limit_hit, requested_qty, "
                       "approved_qty, price, stage, decided_at) VALUES (:n, (SELECT id FROM sleeves WHERE name = 's'), "
                       ":t, :i, 'entry', 'BTC', 1, :v, :h, 'rejected', NULL, 1, 0, 1, 'submit', :t)"),
                  {"n": n, "t": T0, "i": f"i{n}", "v": version, "h": test_hit})


def test_a_decision_names_only_a_test_of_the_version_it_records(store):
    store.create_sleeve(name="s", strategy="ma_cross", instrument="XBT/USD", bar_spec="1h", starting_balance=0)
    _decision(store.engine, 1, "asset_stop_loss", 1)
    _decision(store.engine, 1, None, 2)
    for n, (v, hit) in enumerate(((1, "made_up"), (None, "asset_stop_loss"), (99, "asset_stop_loss")), start=3):
        with pytest.raises(REFUSED):
            _decision(store.engine, v, hit, n)
    with pytest.raises(REFUSED), store.engine.begin() as c:
        c.execute(text("UPDATE gate_decisions SET test_hit = 'made_up' WHERE seq = 1"))


# --- migration 0027 -----------------------------------------------------------------------------------------------


def _at(eng, rev):
    from alembic import command

    with eng.begin() as conn:
        command.upgrade(schema._config(conn), rev)


def _down(eng, rev):
    from alembic import command

    with eng.begin() as conn:
        command.downgrade(schema._config(conn), rev)


def test_0027_seeds_the_same_version_1_as_the_store_and_maps_every_underlying_there(engine, tmp_path):
    _at(engine, "0026")
    with engine.begin() as c:
        for u, sym in (("ADA", "ADA/USD"), ("BTC", "XBT/USD")):
            c.execute(text("INSERT INTO instruments (venue, symbol, contract, underlying, created_at) VALUES "
                           "('KRAKEN', :s, 'spot', :u, :t)"), {"s": sym, "u": u, "t": T0})
        c.execute(text("INSERT INTO portfolio_state (id, status, profile_version, updated_at) "
                       "VALUES (1, 'ok', 1, :t)"), {"t": T0})
    _at(engine, "0027")
    s = Store(engine=engine)
    got = s.compliance()
    want = Store(engine=make_engine(f"sqlite:///{tmp_path}/fresh.db")).compliance()
    assert (got.version, got.created_at, got.created_by, got.note, got.tests) == \
           (want.version, want.created_at, want.created_by, want.note, want.tests)
    assert got.asset_types == want.asset_types | {"ADA": "small"}
    with engine.connect() as c:
        assert c.execute(text("SELECT compliance_version FROM portfolio_state")).scalar() == 1
    if schema.head() == "0027":
        assert schema.report(engine)[1] == []


def test_0027_on_a_database_a_newer_store_already_opened_keeps_its_one_seed(engine):
    _at(engine, "0026")
    Store(engine=engine)
    _at(engine, "0027")
    assert _count(engine, compliance_versions_t) == 1 and _count(engine, compliance_tests_t) == 8


def test_0027_goes_down_only_while_no_one_has_edited_or_decided_under_it(engine):
    _at(engine, "0027")
    _down(engine, "0026")
    _at(engine, "0027")
    s = Store(engine=engine)
    c1 = s.compliance()
    s.add_compliance_version(_edit(c1, asset_stop_loss="0.03"), c1.asset_types, expected_version=1, note="x")
    with pytest.raises(NotImplementedError, match="never dropped"):
        _down(engine, "0026")
    assert s.compliance().version == 2
