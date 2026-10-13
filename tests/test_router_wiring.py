"""P2-4 router v1 wiring: one resolver for a strategy's venue, every order journaled with it (orders.venue, migration
0020), and no order sent to any venue but the strategy's own, whichever submit site sends it. Cells V1-V5 are PE1's
(pe1-plans/p2-4-router.bundle, handed over 13 Oct); V6 is replaced by the 0020 pins at the end."""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import test_hub_146_qa as qa
from qa155_lib import binance_inst, paper_session, quiet, stub_contract, synth_1m
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError
from test_hub_146_qa import (  # noqa: F401 - _probe registers the probe strategy
    _probe,
    flat_prices,
)

from sleeve_fund import router, schema
from sleeve_fund.paper.config import from_store
from sleeve_fund.paper.journal import MemoryJournal
from sleeve_fund.store import Store, make_engine, metadata, orders_t
from sleeve_fund.venues import DEFAULT_VENUE

ROOT = Path(__file__).resolve().parents[1]


PERP = {"market": "perp"}


def _sleeve(store, name, venue=None, instrument="BTC/USD", params=None):
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument=instrument, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue=venue, params=params)
    return store.sleeve(name)


def _order(store, name, oid):
    store.record_order(name, order_id=oid, side="BUY", qty=1.0, intent="entry", reason="r")
    with store.engine.connect() as c:
        return c.execute(orders_t.select().where(orders_t.c.order_id == oid)).first()


@pytest.fixture
def store(tmp_path):
    return Store(engine=make_engine(f"sqlite:///{tmp_path}/j.db"))


# --- one resolver ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pin,want", [(None, DEFAULT_VENUE), ("binance", "BINANCE"), ("BINANCE_P2", "BINANCE")])
def test_venue_for_is_the_pin_or_the_default_upper_case_with_a_clone_as_its_base(pin, want):
    assert router.venue_for(pin) == want


def test_the_router_and_paper_config_resolve_every_strategy_to_the_same_venue(store):
    for name, pin in (("plain", None), ("pinned", "BINANCE")):
        s = _sleeve(store, name, pin, "BTC/USDT" if pin else "BTC/USD", PERP if pin else None)
        assert router.venue_for(s.venue) == from_store(s).venue


def test_check_takes_a_portfolio_runs_clone_as_its_base_venue_and_refuses_any_other():
    router.check("s", "BTCUSDT-PERP.BINANCE_P3", "BINANCE")
    with pytest.raises(router.VenueMismatch, match="own venue only"):
        router.check("s", "XBT/USD.KRAKEN", "BINANCE")


# --- every order row carries its venue ------------------------------------------------------------------------------

def test_every_order_row_carries_its_strategys_venue_whatever_the_caller(store):
    _sleeve(store, "plain")
    _sleeve(store, "pinned", "binance", "BTC/USDT")
    assert _order(store, "plain", "o1").venue == DEFAULT_VENUE
    assert _order(store, "pinned", "o2").venue == "BINANCE"
    with pytest.raises(TypeError):  # no caller names a venue: the router does
        store.record_order("plain", order_id="o3", side="BUY", qty=1.0, intent="entry", reason="r", venue="X")


def test_an_archived_runs_orders_keep_their_venue_and_its_copied_pin_resolves_to_it(store):
    _sleeve(store, "pinned", "BINANCE", "BTC/USDT", PERP)
    _order(store, "pinned", "before")
    store.request_reset("pinned", "Settings changed")
    run = store.split_run(store.pending_resets()[0])
    assert {o["venue"] for o in store.orders(run)} == {"BINANCE"}
    assert router.venue_for(store.sleeve(run).venue) == "BINANCE"
    assert _order(store, run, "after").venue == "BINANCE"


def test_a_backtests_journal_carries_its_runs_venue_into_the_saved_rows():
    j = MemoryJournal()
    j.create_sleeve(name="backtest", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                    starting_balance=10_000.0, venue="BINANCE_P1")
    j.record_order("backtest", order_id="b1", side="BUY", qty=1.0, intent="entry", reason="r")
    assert j.orders_["b1"]["venue"] == "BINANCE"


def test_the_database_refuses_a_venue_not_in_upper_case(store):
    _sleeve(store, "plain")
    with pytest.raises(DBAPIError), store.engine.begin() as c:
        c.execute(orders_t.insert().values(sleeve="plain", order_id="x", ts=pd.Timestamp.now(tz="UTC"),
                                           updated_at=pd.Timestamp.now(tz="UTC"), side="BUY", order_type="MARKET",
                                           qty=1, status="submitted", filled_qty=0, fee=0, intent="entry",
                                           reason="r", signal={}, message="", venue="binance"))


# --- PE1's cells: the route, the refusal, the gate order, no bypass ---------------------------------------------

def _spy(monkeypatch):
    checked = []
    real = router.check

    def check(strategy, instrument_id, pinned_venue):
        checked.append((strategy, instrument_id, pinned_venue))
        return real(strategy, instrument_id, pinned_venue)

    monkeypatch.setattr(router, "check", check)
    return checked


def _wavy(n=60):
    """A long held from minute 5, a 1% stop hit by a dip at minute 20, re-entered, and closed at minute 50."""
    return qa.shape(flat_prices(n), 20, 21, 0.985)


# V1: the paper node's execution client is the route's: its venue and its account, one per strategy.
def test_v1_paper_builds_its_execution_client_from_the_route(monkeypatch):
    from sleeve_fund.paper import node
    from sleeve_fund.paper.config import load_sleeve

    seen = []
    real = router.route

    def route(*a, **kw):
        seen.append(real(*a, **kw))
        return seen[-1]

    monkeypatch.setattr(router, "route", route)
    configs, real_config = [], node.SandboxExecutionClientConfig

    def config(**kw):
        configs.append(kw)
        return real_config(**kw)

    monkeypatch.setattr(node, "SandboxExecutionClientConfig", config)
    for k in list(__import__("os").environ):
        if k.upper().startswith("KRAKEN_"):
            monkeypatch.delenv(k)
    cfg = load_sleeve(ROOT / "configs" / "sleeves" / "ping_pong_test.toml")
    built = node.build_node(cfg, log_level="ERROR", asset_fetch=dict)
    try:
        (r,) = seen
        assert r.venue == cfg.venue and r.account_id.startswith(f"{cfg.venue}-PAPER-")
        assert r.strategy == cfg.name
        (made,) = configs
        assert str(made["account_id"]) == r.account_id  # the execution client is opened on the route's account
    finally:
        built.dispose()


# V2 + V5: a mismatched order is refused, never sent, its row marked denied and an error event written; the trade_from
# gate still runs first.
def _stub(trade_from=None):
    from sleeve_fund.strategies.base import LongFlatStrategy

    events, statuses, sent = [], [], []
    runtime = SimpleNamespace(name="rt-v2", venue="KRAKEN",
                              now=lambda: pd.Timestamp("2026-10-13", tz="UTC").to_pydatetime(),
                              on_order_status=lambda oid, st, msg="": statuses.append((oid, st, msg)),
                              store=SimpleNamespace(event=lambda *a, **k: events.append(a)))
    stub = SimpleNamespace(_cfg=SimpleNamespace(trade_from=trade_from), clock=SimpleNamespace(timestamp_ns=lambda: 0),
                           decisions={"O-1": {"intent": "entry"}}, runtime=runtime, _backtest=False,
                           log=SimpleNamespace(error=lambda _: None))
    order = SimpleNamespace(client_order_id="O-1", instrument_id="BTCUSDT-PERP.BINANCE")
    return LongFlatStrategy.submit_order, stub, order, events, statuses, sent


def test_v2_an_order_for_another_venue_is_refused_marked_denied_and_logged():
    submit, stub, order, events, statuses, _ = _stub()
    submit(stub, order)  # returns before the parent's submit: a refused order never reaches any venue
    assert statuses == [("O-1", "denied", statuses[0][2])] and "own venue only" in statuses[0][2]
    (ev,) = events
    assert ev[:3] == ("rt-v2", "error", "order_refused_venue")
    assert "O-1" not in stub.decisions


def test_v5_the_trade_from_gate_runs_before_the_router():
    submit, stub, order, events, statuses, _ = _stub(trade_from=10**18)
    submit(stub, order)
    assert events == [] and statuses == [] and "O-1" not in stub.decisions


# V3 + V4: in an engine run every order the strategy sends was checked by the router against its pinned venue, and every
# journaled order carries that venue, in paper and in backtest. Watched stops are journaled but never sent.
@pytest.mark.parametrize("perp", [False, True], ids=["spot", "perp"])
def test_v3_v4_paper_every_order_is_routed_and_journals_its_venue(monkeypatch, perp):
    checked = _spy(monkeypatch)
    run = qa.paper(_wavy(), side=1, perp=perp, profile="aggressive" if not perp else "conservative", stop=0.01,
                   enter=5, leave=50, extra={"after": 1})
    sent = [o for o in run.orders if o["order_type"] != "STOP (watched)"]
    assert len(sent) >= 2, [(o["intent"], o["order_type"]) for o in run.orders]
    assert {o["venue"] for o in run.orders} == {"KRAKEN"}, [(o["intent"], o["venue"]) for o in run.orders]
    assert checked and {c[2] for c in checked} == {"KRAKEN"} and {c[0] for c in checked} == {"q146"}
    assert len(checked) >= len(sent), (len(checked), [(o["intent"], o["order_type"]) for o in sent])


def test_v3_v4_backtest_every_order_is_routed_and_journals_its_venue(monkeypatch):
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    checked = _spy(monkeypatch)
    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    m = qa.minutes_of(_wavy())
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    res = run_backtest("probe", bars, inst, params={"enter": 5, "leave": 50, "side": 1, "after": 1, "stop_loss": 0.01},
                       starting_capital=10_000, risk_profile="aggressive", bar_minutes=1,
                       half_spread=qa.SPREAD / 2 / qa.BASE)
    orders = list(res.journal.orders_.values())
    assert len(orders) >= 2 and {o["venue"] for o in orders} == {"KRAKEN"}, [(o["intent"], o["venue"])
                                                                            for o in orders]
    assert len(checked) >= len(orders) and {c[2] for c in checked} == {"KRAKEN"}


def test_v4_the_engine_sends_orders_only_through_the_routed_submit_order():
    """No bypass: the one call to Nautilus' own submit_order is the router's override in base.py, and nothing in the
    package calls Strategy.submit_order (or submit_order_list) directly."""
    calls = []
    for path in (ROOT / "sleeve_fund").rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"super\(\)\.submit_order|Strategy\.submit_order|submit_order_list\(", code):
                calls.append((path.relative_to(ROOT).as_posix(), line.strip()))
    assert calls == [("sleeve_fund/strategies/base.py", "super().submit_order(order, *args, **kwargs)")], calls


# --- no order to another venue --------------------------------------------------------------------------------------

def _session(store, name, monkeypatch, params):
    stub_contract(monkeypatch)  # before the instrument is built: no network
    inst = binance_inst()
    m1 = synth_1m(days=1, seed=3, p0=60_000.0)
    return quiet(lambda: paper_session(store, name, inst, m1, strategy="buy_and_hold", params=params, create=False))


def test_a_strategy_handed_another_venues_market_refuses_to_start_and_sends_nothing(store, monkeypatch):
    _sleeve(store, "pinned", DEFAULT_VENUE, "BTC/USD")  # pinned to the default venue, handed another venue's market
    with pytest.raises(RuntimeError, match="won't start"):
        _session(store, "pinned", monkeypatch, PERP)
    assert store.fills("pinned") == [] and store.orders("pinned") == []
    (why,) = [e for e in store.events("pinned", limit=1000) if e["kind"] == "venue_mismatch"]
    assert "won't start" in why["message"] and why["level"] == "error"


# --- migration 0020 -------------------------------------------------------------------------------------------------

def _clean(eng):
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield eng
    _clean(eng)


def _orders_shape(eng):
    i = inspect(eng)
    return (sorted((c["name"], c["nullable"]) for c in i.get_columns("orders")),
            sorted(c["name"] for c in i.get_check_constraints("orders")))


def _at_0014_with_orders(eng):
    from alembic import command

    with eng.begin() as conn:
        command.upgrade(schema._config(conn), "0014")
        now = pd.Timestamp("2026-10-01", tz="UTC").to_pydatetime()
        for name in ("old", "pinned"):
            conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, "
                              "risk_profile, warmup_bars, desired_state, status, status_reason, created_at, updated_at)"
                              " VALUES (:n, 'buy_and_hold', 'BTC/USD', '1-DAY-LAST-EXTERNAL', '{}', 1000, 'balanced', "
                              "0, 'stopped', 'stopped', '', :t, :t)"), {"n": name, "t": now})
        conn.execute(text("INSERT INTO sleeve_venues (sleeve, venue) VALUES ('pinned', 'binance')"))
        # An archived run (split_run): its own copy of the pin.
        conn.execute(text("INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, "
                          "risk_profile, warmup_bars, desired_state, status, status_reason, created_at, updated_at)"
                          " VALUES ('pinned--20260930', 'buy_and_hold', 'BTC/USD', '1-DAY-LAST-EXTERNAL', '{}', 1000, "
                          "'balanced', 0, 'stopped', 'stopped', '', :t, :t)"), {"t": now})
        conn.execute(text("INSERT INTO sleeve_venues (sleeve, venue) VALUES ('pinned--20260930', 'BINANCE')"))
        for i, name in enumerate(("old", "pinned", "pinned", "pinned--20260930")):
            conn.execute(text("INSERT INTO orders (sleeve, order_id, ts, updated_at, side, order_type, qty, status, "
                              "filled_qty, fee, intent, reason, signal, message) VALUES (:s, :o, :t, :t, 'BUY', "
                              "'MARKET', 1, 'filled', 1, 0, 'entry', 'r', '{}', '')"),
                         {"s": name, "o": f"o{i}", "t": now})


def test_0020_backfills_each_order_with_its_strategys_venue_then_matches_create_all(engine, tmp_path):
    _at_0014_with_orders(engine)
    assert schema.migrate(engine, log=lambda _: None) == f"at migration {schema.head()}"
    with engine.connect() as c:
        got = dict(c.execute(text("SELECT order_id, venue FROM orders")).all())
    assert got == {"o0": DEFAULT_VENUE, "o1": "BINANCE", "o2": "BINANCE", "o3": "BINANCE"}
    assert schema.report(engine)[1] == []
    migrated = _orders_shape(engine)
    fresh = make_engine(f"sqlite:///{tmp_path}/fresh.db") if engine.dialect.name == "sqlite" else engine
    if fresh is engine:
        _clean(engine)
    Store(engine=fresh)
    assert migrated == _orders_shape(fresh)
    assert ("venue", False) in migrated[0] and "orders_venue_upper" in migrated[1]


def test_0020_on_a_database_create_all_already_built_adds_nothing_twice(engine):
    from alembic import command

    Store(engine=engine)
    with engine.begin() as conn:
        command.stamp(schema._config(conn), "0014")
    assert schema.migrate(engine, log=lambda _: None) == f"at migration {schema.head()}"
    assert _orders_shape(engine)[1].count("orders_venue_upper") == 1
    assert schema.report(engine)[1] == []


def test_0020_frozen_default_is_todays_default_venue():
    import importlib.util
    from pathlib import Path

    import sleeve_fund

    path = Path(sleeve_fund.__file__).parent / "migrations" / "versions" / "0020_orders_venue.py"
    spec = importlib.util.spec_from_file_location("migration_0020", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT_VENUE_THEN == DEFAULT_VENUE
    with pytest.raises(NotImplementedError):
        mod.downgrade()


def test_an_unpinned_strategy_is_held_to_the_venue_its_market_was_built_on_a_pinned_one_to_its_pin():
    """HoE 13 Oct: a strategy saved before strategies carried a venue (no pin) is checked against the venue its market
    was built on, so test harnesses that build one on another venue's market run as before; production paper builds
    every unpinned strategy on the default venue (F-P24-1 makes every new one pinned). A pinned one is held to its
    pin."""
    from nautilus_trader.trading import Strategy

    from sleeve_fund.strategies.base import LongFlatStrategy

    sent = []

    class Sink(Strategy):
        def submit_order(self, order, *a, **k):
            sent.append(order.client_order_id)

    class Bare(LongFlatStrategy, Sink):  # Sink takes the parent's submit_order: what reaches the engine is recorded
        clock = SimpleNamespace(timestamp_ns=lambda: 0)
        log = SimpleNamespace(error=lambda _: None)

    for pinned in (False, True):
        s = Bare.__new__(Bare)
        s._cfg = SimpleNamespace(trade_from=None, instrument_id=SimpleNamespace(venue=SimpleNamespace(value="BINANCE")))
        statuses = []
        s.runtime = SimpleNamespace(name="u", venue=DEFAULT_VENUE, pinned=pinned, backtest=False, now=lambda: None,
                                    on_order_status=lambda oid, st, msg="": statuses.append(st),
                                    store=SimpleNamespace(event=lambda *a, **k: None))
        s.decisions = {}
        s.submit_order(SimpleNamespace(client_order_id=f"O-{pinned}", instrument_id="BTCUSDT-PERP.BINANCE"))
    assert sent == ["O-False"] and statuses == ["denied"]
