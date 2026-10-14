"""0020 (P2-4) backfills orders.venue as paper/config resolved it when each order was sent. A saved backtest's strategy
has no pin row (save_backtest never wrote one), so its orders take the venue its backtests row ran on, read from the
run's query as the dashboard read it (F-WED1-1); a strategy with neither takes the default venue of the day."""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from sleeve_fund import schema
from sleeve_fund.store import make_engine, metadata

T = datetime(2026, 10, 13, 9, tzinfo=timezone.utc)


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    if eng.dialect.name == "postgresql":
        with eng.begin() as c:
            c.execute(text("DROP FUNCTION IF EXISTS trials_write_once() CASCADE"))


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/v.db")
    _clean(eng)
    yield eng
    _clean(eng)


def _before_0020(eng):
    from alembic import command
    from alembic.script import ScriptDirectory

    before = ScriptDirectory.from_config(schema._config(None)).get_revision("0020").down_revision
    with eng.begin() as conn:
        command.upgrade(schema._config(conn), before)


def _strategy(conn, name, pin=None, queries=()):
    conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, risk_profile,"
                      " warmup_bars, desired_state, status, status_reason, created_at, updated_at) VALUES (:n, "
                      "'buy_and_hold', 'BTC/USDT', '1-DAY-LAST-EXTERNAL', '{}', 1000, 'balanced', 0, 'stopped', "
                      "'stopped', '', :t, :t)"), {"n": name, "t": T})
    if pin:
        conn.execute(text("INSERT INTO sleeve_venues (sleeve, venue) VALUES (:n, :v)"), {"n": name, "v": pin})
    for i, q in enumerate(queries):
        conn.execute(text("INSERT INTO backtests (id, sleeve, key, title, query, created_at, result) VALUES "
                          "(:i, :n, 'k', 't', :q, :t, '{}')"), {"i": f"{name}-{i}", "n": name, "q": q, "t": T})
    conn.execute(text("INSERT INTO orders (sleeve, order_id, ts, updated_at, side, order_type, qty, status, filled_qty, "
                      "fee, intent, reason, signal, message) VALUES (:s, :o, :t, :t, 'BUY', 'MARKET', 1, 'filled', 1, "
                      "0, 'entry', 'r', '{}', '')"), {"s": name, "o": f"{name}-o", "t": T})


def test_each_order_gets_the_venue_its_strategy_was_sent_to(engine):
    _before_0020(engine)
    with engine.begin() as conn:
        _strategy(conn, "bt-binance", queries=["venue=binance&pair=BTC%2FUSDT"])
        _strategy(conn, "bt-blank", queries=["pair=BTC%2FUSD"])  # no venue param: the dashboard's default
        _strategy(conn, "bt-unknown", queries=["venue=nowhere&pair=BTC%2FUSD"])  # no profile: the default too
        _strategy(conn, "bt-two", queries=["venue=binance", "venue=kraken"])  # no one venue to name
        _strategy(conn, "pinned", pin="binance", queries=["venue=kraken"])  # the pin is what paper sent to
        _strategy(conn, "plain")  # neither: the default of the day
    schema.migrate(engine, log=lambda _: None)
    with engine.connect() as c:
        got = dict(c.execute(text("SELECT sleeve, venue FROM orders")).all())
    assert got == {"bt-binance": "BINANCE", "bt-blank": "KRAKEN", "bt-unknown": "KRAKEN", "bt-two": "KRAKEN",
                   "pinned": "BINANCE", "plain": "KRAKEN"}
