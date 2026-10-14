"""Bridge A, data lane (DA spec bridge-a-data-spec.md 1-2, migration 0025): the fund's capital ledger is append-only
and ends each flow once, in id order; instruments hold one row per venue, symbol and contract. Both engines: CI runs
these against Postgres via TEST_DATABASE_URL."""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from sleeve_fund import schema
from sleeve_fund.store import Store, decisions_t, fund_flows_t, make_engine, metadata

T0 = datetime(2026, 10, 14, 21, 0, tzinfo=timezone.utc)
REFUSED = (IntegrityError, DBAPIError)


def _clean(eng):
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        if eng.dialect.name == "postgresql":
            conn.execute(text("DROP FUNCTION IF EXISTS fund_flows_write_once() CASCADE"))
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


def _count(eng, t=fund_flows_t):
    with eng.connect() as c:
        return c.execute(select(func.count()).select_from(t)).scalar()


# --- the ledger ---------------------------------------------------------------------------------------------------


def test_a_flow_is_journaled_with_its_decision_and_moves_nothing_until_the_poll_applies_it(store):
    fid = store.fund_flow("deposit", "100000", actor="pm", reason="clean slate", rebase=True, now=T0)
    assert store.fund_capital() == 0 and [f["id"] for f in store.pending_flows()] == [fid]
    with store.engine.connect() as c:
        d = c.execute(select(decisions_t).where(decisions_t.c.action == "fund_deposit")).one()
    assert d.actor == "pm" and f"flow {fid}, re-base" in d.reason
    store.settle_flow(fid, T0 + timedelta(minutes=1))
    assert store.fund_capital() == Decimal("100000") and store.pending_flows() == []
    assert store.fund_capital(at=T0) == 0  # applied after then


def test_capital_is_exact_deposits_less_withdrawals(store):
    a = store.fund_flow("deposit", "0.1", actor="pm", reason="a", now=T0)
    b = store.fund_flow("deposit", "0.2", actor="pm", reason="b", now=T0)
    w = store.fund_flow("withdraw", "0.3", actor="pm", reason="c", now=T0)
    for i, f in enumerate((a, b, w)):
        store.settle_flow(f, T0 + timedelta(minutes=i))
    assert store.fund_capital() == Decimal(0)
    assert store.fund_capital(at=T0 + timedelta(minutes=1)) == Decimal("0.3")


@pytest.mark.parametrize("kind, amount, rebase, actor, reason", [
    ("transfer", "1", False, "pm", "x"),
    ("deposit", "0", False, "pm", "x"),
    ("deposit", "-5", False, "pm", "x"),
    ("withdraw", "5", True, "pm", "x"),
    ("deposit", "5", False, "", "x"),
    ("deposit", "5", False, "pm", "  "),
])
def test_the_store_refuses_a_flow_outside_the_rules_and_writes_nothing(store, kind, amount, rebase, actor, reason):
    with pytest.raises(ValueError):
        store.fund_flow(kind, amount, actor=actor, reason=reason, rebase=rebase, now=T0)
    assert _count(store.engine) == 0 and _count(store.engine, decisions_t) == 0


@pytest.mark.parametrize("values", [
    {"kind": "transfer"},
    {"amount": 0},
    {"kind": "withdraw", "rebase": True},
    {"actor": ""},
    {"applied_at": T0, "refused_at": T0, "refuse_reason": "x"},
    {"refused_at": T0},
    {"refuse_reason": "x"},
])
def test_the_database_refuses_a_row_outside_the_rules(store, values):
    row = {"kind": "deposit", "amount": 1, "rebase": False, "actor": "pm", "reason": "r", "requested_at": T0} | values
    with pytest.raises(REFUSED), store.engine.begin() as c:
        c.execute(fund_flows_t.insert().values(**row))


@pytest.mark.parametrize("sql", [
    "DELETE FROM fund_flows",
    "UPDATE fund_flows SET amount = 2",
    "UPDATE fund_flows SET kind = 'withdraw'",
    "UPDATE fund_flows SET reason = 'other'",
    "UPDATE fund_flows SET requested_at = applied_at",
])
def test_a_flow_is_never_deleted_or_edited_at_the_database(store, sql):
    store.settle_flow(store.fund_flow("deposit", "1", actor="pm", reason="r", now=T0), T0)
    store.fund_flow("deposit", "1", actor="pm", reason="r", now=T0)  # a pending one too
    with pytest.raises(REFUSED), store.engine.begin() as c:
        c.execute(text(sql))
    assert _count(store.engine) == 2 and store.fund_capital() == 1


def test_a_flow_is_ended_once_and_never_both_ways(store):
    fid = store.fund_flow("withdraw", "5", actor="pm", reason="r", now=T0)
    store.refuse_flow(fid, T0, "more than free cash")
    with pytest.raises(ValueError):
        store.settle_flow(fid, T0)
    for sql in ("UPDATE fund_flows SET applied_at = refused_at",
                "UPDATE fund_flows SET refused_at = NULL, refuse_reason = NULL",
                "UPDATE fund_flows SET refuse_reason = 'other'"):
        with pytest.raises(REFUSED), store.engine.begin() as c:
            c.execute(text(sql))
    f = store.fund_flows()[0]
    assert f["applied_at"] is None and f["refuse_reason"] == "more than free cash"
    with pytest.raises(ValueError):
        store.refuse_flow(fid, T0, "  ")


def test_flows_end_in_order_so_an_inflow_never_overtakes_a_pending_outflow(store):
    """Advisor CAP5-RESET: a deposit behind a pending withdrawal waits until the withdrawal is ended."""
    w = store.fund_flow("withdraw", "5", actor="pm", reason="out", now=T0)
    d = store.fund_flow("deposit", "50", actor="pm", reason="in", now=T0)
    with pytest.raises(ValueError, match="in order"):
        store.settle_flow(d, T0)
    assert [f["id"] for f in store.pending_flows()] == [w, d]
    store.refuse_flow(w, T0, "more than free cash")
    store.settle_flow(d, T0)
    assert store.fund_capital() == 50


def test_the_poll_ends_a_flow_inside_its_own_transaction(store):
    fid = store.fund_flow("deposit", "7", actor="pm", reason="r", now=T0)
    with pytest.raises(RuntimeError), store.engine.begin() as c:
        store.settle_flow(fid, T0, conn=c)
        raise RuntimeError("the poll's book write failed")
    assert store.fund_capital() == 0 and len(store.pending_flows()) == 1


# --- instruments --------------------------------------------------------------------------------------------------


def test_an_instrument_is_one_row_and_never_nets_as_another_underlying(store):
    i = store.instrument_id("binance", "BTC/USDT", "perp", "BTC")
    assert store.instrument_id("BINANCE", "BTC/USDT", "perp", "BTC") == i
    assert store.instrument_id("BINANCE", "BTC/USDT", "spot", "BTC") != i
    with pytest.raises(ValueError, match="nets as BTC"):
        store.instrument_id("BINANCE", "BTC/USDT", "perp", "ETH")
    with pytest.raises(ValueError):
        store.instrument_id("BINANCE", "BTC/USDT", "option", "BTC")
    assert [r["venue"] for r in store.instruments()] == ["BINANCE", "BINANCE"]


def test_a_new_strategy_writes_its_instrument_and_a_backtest_does_not(store):
    store.create_sleeve(name="trend", strategy="ma_cross", instrument="XBT/USD", bar_spec="1h", starting_balance=0)
    store.create_sleeve(name="perp1", strategy="ma_cross", instrument="SOL/USDT", bar_spec="1h", starting_balance=0,
                        params={"market": "futures"}, venue="binance")
    store.create_sleeve(name="bt:x", strategy="ma_cross", instrument="DOGE/USD", bar_spec="1h", starting_balance=0)
    got = {(r["venue"], r["symbol"], r["contract"], r["underlying"]) for r in store.instruments()}
    assert got == {("KRAKEN", "XBT/USD", "spot", "BTC"), ("BINANCE", "SOL/USDT", "perp", "SOL")}


# --- migration 0025 -----------------------------------------------------------------------------------------------


def _at(eng, rev):
    from alembic import command

    with eng.begin() as conn:
        command.upgrade(schema._config(conn), rev)


def _down(eng, rev):
    from alembic import command

    with eng.begin() as conn:
        command.downgrade(schema._config(conn), rev)


def test_0025_seeds_instruments_from_the_strategies_there_and_matches_the_code(engine):
    _at(engine, "0024")
    with engine.begin() as c:
        for name, inst, params in (("a", "XBT/USD", "{}"), ("b", "ETH/USDT", '{"market": "futures"}'),
                                   ("c", "ETH/USDT", '{"market": "futures"}'), ("bt:z", "ADA/USD", "{}")):
            c.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, "
                           "risk_profile, warmup_bars, desired_state, status, status_reason, created_at, updated_at) "
                           "VALUES (:n, 's', :i, '1h', :p, 0, 'balanced', 0, 'stopped', 'stopped', '', :t, :t)"),
                      {"n": name, "i": inst, "p": params, "t": T0})
        c.execute(text("INSERT INTO sleeve_venues (sleeve, venue) VALUES ('b', 'binance'), ('c', 'BINANCE')"))
    _at(engine, "0025")
    with engine.connect() as c:
        got = set(c.execute(text("SELECT venue, symbol, contract, underlying FROM instruments")).all())
    assert got == {("KRAKEN", "XBT/USD", "spot", "BTC"), ("BINANCE", "ETH/USDT", "perp", "ETH")}
    if schema.head() == "0025":
        assert schema.report(engine)[1] == []


def test_0025_on_a_database_a_newer_store_already_opened_keeps_its_tables(engine):
    _at(engine, "0024")
    s = Store(engine=engine)
    s.settle_flow(s.fund_flow("deposit", "3", actor="pm", reason="r", now=T0), T0)
    _at(engine, "0025")
    assert s.fund_capital() == 3
    with pytest.raises(REFUSED), engine.begin() as c:
        c.execute(text("DELETE FROM fund_flows"))


def test_0025_writes_the_same_write_once_rules_as_the_store(engine):
    _at(engine, "0025")
    s = Store(engine=engine)
    fid = s.fund_flow("deposit", "3", actor="pm", reason="r", now=T0)
    for sql in ("UPDATE fund_flows SET amount = 4", "DELETE FROM fund_flows",
                "UPDATE fund_flows SET applied_at = :t, refused_at = :t, refuse_reason = 'x'"):
        with pytest.raises(REFUSED), engine.begin() as c:
            c.execute(text(sql), {"t": T0})
    s.settle_flow(fid, T0)
    with pytest.raises(REFUSED), engine.begin() as c:
        c.execute(text("UPDATE fund_flows SET applied_at = NULL"))


def test_0025_goes_down_only_with_no_money_in_the_ledger(engine):
    _at(engine, "0025")
    _down(engine, "0024")
    assert not {"fund_flows", "instruments"} & set(inspect(engine).get_table_names())
    _at(engine, "0025")
    Store(engine=engine).fund_flow("deposit", "1", actor="pm", reason="r", now=T0)
    with pytest.raises(NotImplementedError, match="never dropped"):
        _down(engine, "0024")
    assert "fund_flows" in inspect(engine).get_table_names() and _count(engine) == 1


# --- the command (python -m sleeve_fund.fund) ---------------------------------------------------------------------


def _cmd(store, *argv):
    said = []
    return __import__("sleeve_fund.fund", fromlist=["run"]).run(list(argv), store=store, out=said.append), said


def test_the_command_is_a_dry_run_by_default_and_writes_nothing(store):
    code, said = _cmd(store, "deposit", "25000", "--actor", "pm", "--reason", "top up")
    assert code == 0 and said == ["dry run: would journal deposit 25000.00; 0 flow(s) ahead of it; nothing written"]
    assert _count(store.engine) == 0 and _count(store.engine, decisions_t) == 0


def test_the_command_writes_on_apply_and_prints_only_kind_amount_id_and_outcome(store):
    store.fund_flow("deposit", "1", actor="pm", reason="earlier", now=T0)
    code, said = _cmd(store, "withdraw", "1250.50", "--actor", "pm", "--reason", "secret words", "--apply")
    f = store.fund_flows()[-1]
    assert code == 0 and (f["kind"], f["amount"], f["reason"], f["rebase"]) == ("withdraw", Decimal("1250.5"),
                                                                                "secret words", False)
    assert said == [f"journaled: withdraw 1250.50 as flow {f['id']}; 1 flow(s) ahead of it; the next poll ends it"]
    assert "secret" not in said[0]


@pytest.mark.parametrize("amount, actor, reason", [
    ("0", "pm", "r"), ("1.005", "pm", "r"), ("-5", "pm", "r"), ("1e6", "pm", "r"), ("5", "p m", "r"),
    ("5", "pm", "it's"), ("5", "pm", "a; rm -rf /"), ("5", "pm", " "),
])
def test_the_command_refuses_a_bad_request_and_writes_nothing(store, amount, actor, reason):
    code, said = _cmd(store, "deposit", amount, "--actor", actor, "--reason", reason, "--apply")
    assert code == 1 and said[0].startswith("refused: ") and _count(store.engine) == 0


# --- instrument on orders, fills, funding and insurance (0026) ----------------------------------------------------

FOUR = ("orders", "fills", "funding", "insurance")


def _instruments(eng, table):
    with eng.connect() as c:
        return [r[0] for r in c.execute(text(f"SELECT instrument FROM {table} ORDER BY id"))]


def test_every_writer_today_gets_its_strategys_instrument_without_naming_it(store):
    store.create_sleeve(name="p", strategy="ma_cross", instrument="ETH/USDT", bar_spec="1h", starting_balance=0,
                        params={"market": "futures"}, venue="binance")
    store.record_order("p", order_id="o1", side="BUY", qty=1, intent="entry", reason="r", ts=T0)
    store.record_fill("p", side="BUY", qty=1, price=2000, fee=1, order_id="o1", trade_id="t1", ts=T0)
    store.record_funding("p", qty=1, price=2000, rate=0.0001, amount=-0.2, ts=T0)
    store.record_insurance("p", price=1, amount=3, order_id="o1", trade_id="t1", ts=T0)
    assert {t: _instruments(store.engine, t) for t in FOUR} == {t: ["ETH/USDT"] for t in FOUR}


def test_a_raw_insert_is_filled_too_and_a_named_instrument_is_kept(store):
    store.create_sleeve(name="p", strategy="ma_cross", instrument="SOL/USDT", bar_spec="1h", starting_balance=0)
    with store.engine.begin() as c:
        for inst in ("", None, "BTC/USDT"):
            cols = "sleeve, ts, side, qty, price, fee, order_id, trade_id" + (", instrument" if inst is not None else "")
            vals = ":s, :t, 'BUY', 1, 1, 0, :o, 't'" + (", :i" if inst is not None else "")
            c.execute(text(f"INSERT INTO fills ({cols}) VALUES ({vals})"), {"s": "p", "t": T0, "o": f"o{inst}", "i": inst})
    assert _instruments(store.engine, "fills") == ["SOL/USDT", "SOL/USDT", "BTC/USDT"]


def test_0026_backfills_every_row_from_its_strategy_and_goes_down_cleanly(engine):
    _at(engine, "0025")
    with engine.begin() as c:
        for name, inst in (("a", "XBT/USD"), ("b", "ETH/USDT")):
            c.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, "
                           "risk_profile, warmup_bars, desired_state, status, status_reason, created_at, updated_at) "
                           "VALUES (:n, 's', :i, '1h', '{}', 0, 'balanced', 0, 'stopped', 'stopped', '', :t, :t)"),
                      {"n": name, "i": inst, "t": T0})
            c.execute(text("INSERT INTO orders (sleeve, order_id, ts, updated_at, side, order_type, qty, status, "
                           "filled_qty, fee, intent, reason, signal, message, venue) VALUES (:n, :o, :t, :t, 'BUY', "
                           "'MARKET', 1, 'filled', 1, 0, 'entry', 'r', '{}', '', 'KRAKEN')"), {"n": name, "o": name, "t": T0})
            c.execute(text("INSERT INTO fills (sleeve, ts, side, qty, price, fee, order_id, trade_id) VALUES "
                           "(:n, :t, 'BUY', 1, 1, 0, :n, 't')"), {"n": name, "t": T0})
            c.execute(text("INSERT INTO funding (sleeve, ts, qty, price, rate, amount, kind) VALUES "
                           "(:n, :t, 1, 1, 0.0001, 0.1, 'settled')"), {"n": name, "t": T0})
            c.execute(text("INSERT INTO insurance (sleeve, ts, price, amount) VALUES (:n, :t, 1, 1)"),
                      {"n": name, "t": T0})
    _at(engine, "0026")
    assert {t: _instruments(engine, t) for t in FOUR} == {t: ["XBT/USD", "ETH/USDT"] for t in FOUR}
    with engine.begin() as c:  # and a row written after the migration is filled as the Store's are
        c.execute(text("INSERT INTO insurance (sleeve, ts, price, amount) VALUES ('b', :t, 1, 1)"), {"t": T0})
    assert _instruments(engine, "insurance")[-1] == "ETH/USDT"
    if schema.head() == "0026":
        assert schema.report(engine)[1] == []
    _down(engine, "0025")
    assert all("instrument" not in {c["name"] for c in inspect(engine).get_columns(t)} for t in FOUR)
    with engine.begin() as c:  # nothing left behind that refers to the column
        c.execute(text("INSERT INTO insurance (sleeve, ts, price, amount) VALUES ('b', :t, 1, 1)"), {"t": T0})


def test_0026_on_a_database_a_newer_store_already_opened_keeps_its_columns(engine):
    _at(engine, "0025")
    s = Store(engine=engine)
    s.create_sleeve(name="p", strategy="ma_cross", instrument="SOL/USDT", bar_spec="1h", starting_balance=0)
    s.record_fill("p", side="BUY", qty=1, price=1, fee=0, order_id="o", trade_id="t", ts=T0)
    _at(engine, "0026")
    s.record_fill("p", side="BUY", qty=1, price=1, fee=0, order_id="o2", trade_id="t", ts=T0)
    assert _instruments(engine, "fills") == ["SOL/USDT", "SOL/USDT"]
