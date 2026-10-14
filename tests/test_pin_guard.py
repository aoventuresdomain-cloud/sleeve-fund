"""F-WED1-3 (Advisor, HoE 14 Oct): 0020 makes a strategy's venue pin insert-only in the database, as a Store's create_all
does. A venue is never changed; a pin is deleted only with its saved backtest (a bt: strategy with a backtests row); a
strategy that has an order or a fill can't be given one. QD's acceptance cell (test_pin_frozen) drives the Store; these drive a migrated database by hand,
each refused and each allowed path, and check the two builds carry the same triggers."""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import schema
from sleeve_fund.store import make_engine, metadata

T = datetime(2026, 10, 14, 3, tzinfo=timezone.utc)
PG_TRIGGERS = {"sleeve_venue_pinned"}
SQLITE_TRIGGERS = {"sleeve_venues_no_edit", "sleeve_venues_no_delete", "sleeve_venues_no_late_pin"}


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    if eng.dialect.name == "postgresql":
        with eng.begin() as c:
            c.execute(text("DROP FUNCTION IF EXISTS trials_write_once() CASCADE"))
            c.execute(text("DROP FUNCTION IF EXISTS sleeve_venue_pinned() CASCADE"))


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/p.db")
    _clean(eng)
    yield eng
    _clean(eng)


def _migrate(eng):
    from alembic import command

    with eng.begin() as conn:
        command.upgrade(schema._config(conn), "head")


def _triggers(eng):
    with eng.connect() as c:
        if eng.dialect.name == "postgresql":
            return set(c.execute(text("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal "
                                      "AND tgrelid = 'sleeve_venues'::regclass")).scalars())
        return set(c.execute(text("SELECT name FROM sqlite_master WHERE type = 'trigger' "
                                  "AND tbl_name = 'sleeve_venues'")).scalars())


def _strategy(conn, name, pin=None, order=False, fill=False, backtest=False):
    conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, risk_profile,"
                      " warmup_bars, desired_state, status, status_reason, created_at, updated_at) VALUES (:n, "
                      "'buy_and_hold', 'BTC/USDT', '1-DAY-LAST-EXTERNAL', '{}', 1000, 'balanced', 0, 'stopped', "
                      "'stopped', '', :t, :t)"), {"n": name, "t": T})
    if pin:
        conn.execute(text("INSERT INTO sleeve_venues (sleeve, venue) VALUES (:n, :v)"), {"n": name, "v": pin})
    if backtest:
        conn.execute(text("INSERT INTO backtests (id, sleeve, key, title, query, created_at, result) VALUES "
                          "(:n, :n, 'k', 't', '', :t, '{}')"), {"n": name, "t": T})
    if order:
        conn.execute(text("INSERT INTO orders (sleeve, order_id, ts, updated_at, side, order_type, qty, status, "
                          "filled_qty, fee, intent, reason, signal, message, venue) VALUES (:s, :o, :t, :t, 'BUY', "
                          "'MARKET', 1, 'submitted', 0, 0, 'entry', 'r', '{}', '', 'BINANCE')"),
                     {"s": name, "o": f"{name}-o", "t": T})
    if fill:
        conn.execute(text("INSERT INTO fills (sleeve, trade_id, order_id, ts, side, qty, price, fee) "
                          "VALUES (:s, :i, :o, :t, 'BUY', 1, 100, 0)"),
                     {"s": name, "i": f"{name}-f", "o": f"{name}-o", "t": T})


def _refused(eng, sql, **params):
    with pytest.raises(IntegrityError, match="venue is pinned"):
        with eng.begin() as conn:
            conn.execute(text(sql), params)


def test_a_migrated_database_carries_the_same_pin_triggers_as_create_all(engine, tmp_path):
    _migrate(engine)
    want = PG_TRIGGERS if engine.dialect.name == "postgresql" else SQLITE_TRIGGERS
    assert _triggers(engine) == want
    _clean(engine)
    metadata.create_all(engine)
    assert _triggers(engine) == want


def test_on_a_migrated_database_a_pin_is_never_changed_and_never_dropped_from_a_strategy(engine):
    _migrate(engine)
    with engine.begin() as conn:
        _strategy(conn, "s1", pin="BINANCE")  # pinned and not yet traded: still frozen
    _refused(engine, "UPDATE sleeve_venues SET venue = 'KRAKEN' WHERE sleeve = 's1'")
    _refused(engine, "DELETE FROM sleeve_venues WHERE sleeve = 's1'")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT venue FROM sleeve_venues WHERE sleeve = 's1'")).scalar() == "BINANCE"


@pytest.mark.parametrize("held", ["order", "fill"])
def test_on_a_migrated_database_a_strategy_that_has_traded_cant_be_given_a_pin(engine, held):
    _migrate(engine)
    with engine.begin() as conn:
        _strategy(conn, "s2", order=True, fill=held == "fill")
    _refused(engine, "INSERT INTO sleeve_venues (sleeve, venue) VALUES ('s2', 'KRAKEN')")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sleeve_venues")).scalar() == 0


def test_on_a_migrated_database_a_new_strategy_is_pinned_and_a_saved_backtests_pin_goes_with_it(engine):
    _migrate(engine)
    with engine.begin() as conn:
        _strategy(conn, "s3")
        conn.execute(text("INSERT INTO sleeve_venues (sleeve, venue) VALUES ('s3', 'BINANCE')"))  # before it trades
        _strategy(conn, "bt:1", pin="BINANCE", order=True, fill=True, backtest=True)
    with engine.begin() as conn:  # the prune's order: pins, then backtests, then the strategy
        conn.execute(text("DELETE FROM sleeve_venues WHERE sleeve = 'bt:1'"))
    with engine.connect() as conn:
        assert dict(conn.execute(text("SELECT sleeve, venue FROM sleeve_venues")).all()) == {"s3": "BINANCE"}


@pytest.mark.parametrize("name", ["paper1", "bt:2"])
def test_a_backtests_row_alone_never_frees_a_strategys_pin(engine, name):
    """HoE 14 Oct: nothing in the schema stops a backtests row naming a paper strategy (backtests.sleeve has no key and
    no check), so the delete also needs the saved-backtest name; and a bt: name with no backtests row stays pinned."""
    _migrate(engine)
    with engine.begin() as conn:
        _strategy(conn, name, pin="BINANCE", order=True, fill=True, backtest=name == "paper1")
    _refused(engine, "DELETE FROM sleeve_venues WHERE sleeve = :n", n=name)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT venue FROM sleeve_venues WHERE sleeve = :n"), {"n": name}).scalar() == "BINANCE"
