"""P2-8 two legs, the schema (DA, 0022): a second leg is the row "<name>~2" whose parent is the first leg, held by one
CHECK on both engines, and orders carry one parent_trade_id across both legs. Migrated and create_all databases agree."""

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import schema
from sleeve_fund.store import make_engine, metadata

T = datetime(2026, 10, 14, 0, 0, tzinfo=UTC)


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    if eng.dialect.name == "postgresql":
        with eng.begin() as c:
            c.execute(text("DROP FUNCTION IF EXISTS trials_write_once() CASCADE"))


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/legs.db")
    _clean(eng)
    yield eng
    _clean(eng)


def _sleeve(conn, name, parent=None):
    conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, risk_profile,"
                      " warmup_bars, desired_state, status, status_reason, created_at, updated_at, parent) VALUES (:n, "
                      "'buy_and_hold', 'BTC/USDT', '1-DAY-LAST-EXTERNAL', '{}', 1000, 'balanced', 0, 'stopped', "
                      "'stopped', '', :t, :t, :p)"), {"n": name, "t": T, "p": parent})


def _refused(engine, name, parent):
    with pytest.raises(IntegrityError), engine.begin() as c:
        _sleeve(c, name, parent)


def _before_0022(eng):
    from alembic import command
    from alembic.script import ScriptDirectory

    before = ScriptDirectory.from_config(schema._config(None)).get_revision("0022").down_revision
    with eng.begin() as conn:
        command.upgrade(schema._config(conn), before)


def test_the_migrated_shape_keeps_its_fk_and_check_and_matches_the_code(engine):
    schema.migrate(engine, log=lambda _: None)
    assert schema.report(engine)[1] == []
    have = inspect(engine)
    assert any(fk["referred_table"] == "sleeves" and fk["constrained_columns"] == ["parent"]
               for fk in have.get_foreign_keys("sleeves"))
    assert "sleeves_leg_name" in {c["name"] for c in have.get_check_constraints("sleeves")}
    assert "orders_parent_trade_id" in {i["name"] for i in have.get_indexes("orders")}


def test_a_leg_row_is_its_parents_name_with_tilde_2_and_nothing_else_is_allowed(engine):
    schema.migrate(engine, log=lambda _: None)
    with engine.begin() as c:
        _sleeve(c, "carry")
        _sleeve(c, "carry~2", "carry")
        _sleeve(c, "other")
    _refused(engine, "carry", "carry")  # self (and a duplicate name)
    _refused(engine, "solo", "solo")  # self
    _refused(engine, "carry~2~2", "carry~2")  # a grandchild: its parent holds "~"
    _refused(engine, "carry~3", "carry")  # a second leg under another name
    _refused(engine, "other~2", "carry")  # a leg named for someone else
    _refused(engine, "top~level", None)  # a "~" on a top-level row
    with pytest.raises(IntegrityError), engine.begin() as c:
        _sleeve(c, "carry~2", "carry")  # a second leg with the fixed name: the name is unique


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="Postgres: SQLite does not enforce foreign keys here")
def test_a_parent_with_a_leg_cannot_be_deleted(engine):
    schema.migrate(engine, log=lambda _: None)
    with engine.begin() as c:
        _sleeve(c, "carry")
        _sleeve(c, "carry~2", "carry")
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(text("DELETE FROM sleeves WHERE name = 'carry'"))


def test_0022_stops_and_lists_existing_tilde_names_and_changes_nothing(engine):
    _before_0022(engine)
    with engine.begin() as c:
        c.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, "
                       "risk_profile, warmup_bars, desired_state, status, status_reason, created_at, updated_at) VALUES "
                       "('odd~name', 'buy_and_hold', 'BTC/USDT', '1-DAY-LAST-EXTERNAL', '{}', 1000, 'balanced', 0, "
                       "'stopped', 'stopped', '', :t, :t)"), {"t": T})
    with pytest.raises(Exception, match="sleeve names holding '~'.*odd~name"):
        schema.migrate(engine, log=lambda _: None)
    assert "parent" not in {c["name"] for c in inspect(engine).get_columns("sleeves")}


def test_0022_keeps_every_existing_row_and_its_children(engine):
    _before_0022(engine)
    with engine.begin() as c:
        c.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, "
                       "risk_profile, warmup_bars, desired_state, status, status_reason, created_at, updated_at) VALUES "
                       "('s1', 'buy_and_hold', 'BTC/USDT', '1-DAY-LAST-EXTERNAL', '{}', 1000, 'balanced', 0, "
                       "'stopped', 'stopped', '', :t, :t)"), {"t": T})
        c.execute(text("INSERT INTO orders (sleeve, order_id, ts, updated_at, side, order_type, qty, status, "
                       "filled_qty, fee, intent, reason, signal, message) VALUES ('s1', 'o1', :t, :t, 'BUY', 'MARKET',"
                       " 1, 'filled', 1, 0, 'entry', 'r', '{}', '')"), {"t": T})
    schema.migrate(engine, log=lambda _: None)
    with engine.connect() as c:
        assert c.execute(text("SELECT name, parent FROM sleeves")).all() == [("s1", None)]
        assert c.execute(text("SELECT order_id, parent_trade_id FROM orders")).all() == [("o1", None)]
