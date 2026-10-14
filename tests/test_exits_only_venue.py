"""F-WED1-3 (Advisor 14 Oct 03:35 UK; DA rule A): an open position whose orders went to another venue than the pin
(a pre-0020 order backfilled to the default venue, or the default venue changed under an unpinned Strategy) keeps
its own venue. The journal writes its exits there, refuses an opening order, and is unchanged otherwise."""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from sleeve_fund import router
from sleeve_fund.paper.config import from_store
from sleeve_fund.store import Store, make_engine, metadata
from sleeve_fund.supervisor import VENUE_EXITS

MISMATCH = "on another venue than its pin"


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/j.db")
    _clean(eng)
    yield Store(engine=eng)
    _clean(eng)


def _held_elsewhere(store, name="s"):
    """Pinned to BINANCE, holding 1 from an entry whose row says KRAKEN (as a backfilled pre-0020 order would)."""
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE")
    store.record_order(name, order_id=f"{name}-e", side="BUY", qty=1.0, intent="entry", reason="r")
    store.record_fill(name, side="BUY", qty=1.0, price=100.0, fee=0.0, order_id=f"{name}-e", trade_id="t1")
    with store.engine.begin() as c:
        c.execute(text("UPDATE orders SET venue = 'KRAKEN' WHERE order_id = :o"), {"o": f"{name}-e"})


def _venue(store, order_id):
    with store.engine.connect() as c:
        return c.execute(text("SELECT venue FROM orders WHERE order_id = :o"), {"o": order_id}).scalar()


def _count(store, table, order_id):
    with store.engine.connect() as c:
        return c.execute(text(f"SELECT count(*) FROM {table} WHERE order_id = :o"), {"o": order_id}).scalar()


@pytest.mark.parametrize("intent", ["stop_loss", "exit", "take_profit", "risk_halt", "pm_flatten"])
def test_a_holding_mismatch_journals_its_exits_on_the_positions_venue(store, intent):
    _held_elsewhere(store)
    assert store.position_venue("s") == "KRAKEN"
    store.record_order("s", order_id="s-x", side="SELL", qty=1.0, intent=intent, reason="r")
    assert _venue(store, "s-x") == "KRAKEN"


@pytest.mark.parametrize("intent", ["entry", "rebalance"])
def test_a_holding_mismatch_refuses_an_opening_order_and_writes_nothing(store, intent):
    _held_elsewhere(store)
    with pytest.raises(ValueError, match=MISMATCH):
        store.record_order("s", order_id="s-n", side="BUY", qty=1.0, intent=intent, reason="r",
                           timing={"decided": 1_760_000_000_000_000_000})
    assert _count(store, "orders", "s-n") == 0 and _count(store, "order_timings", "s-n") == 0


def test_a_flat_mismatch_journals_on_the_pin(store):
    _held_elsewhere(store)
    store.record_order("s", order_id="s-x", side="SELL", qty=1.0, intent="exit", reason="r")
    store.record_fill("s", side="SELL", qty=1.0, price=101.0, fee=0.0, order_id="s-x", trade_id="t2")
    assert store.position_venue("s") is None
    store.record_order("s", order_id="s-n", side="BUY", qty=1.0, intent="entry", reason="r")
    assert _venue(store, "s-n") == "BINANCE"


def test_holding_on_the_pin_is_unchanged(store):
    store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE")
    assert store.position_venue("s") is None
    store.record_order("s", order_id="s-e", side="BUY", qty=1.0, intent="entry", reason="r")
    store.record_fill("s", side="BUY", qty=1.0, price=100.0, fee=0.0, order_id="s-e", trade_id="t1")
    assert store.position_venue("s") == "BINANCE"
    store.record_order("s", order_id="s-a", side="BUY", qty=1.0, intent="entry", reason="r")
    store.record_order("s", order_id="s-x", side="SELL", qty=2.0, intent="stop_loss", reason="r")
    assert _venue(store, "s-a") == _venue(store, "s-x") == "BINANCE"


# --- the start: exits only while holding on another venue, refused once flat (Advisor 14 Oct 03:35 UK, HoE) ----------

@pytest.mark.parametrize("traded,holding,mode", [
    (None, True, router.NORMAL), ("BINANCE", True, router.NORMAL), ("BINANCE_P2", False, router.NORMAL),
    ("KRAKEN", True, router.EXITS_ONLY), ("KRAKEN", False, router.REFUSE)])
def test_start_mode_is_normal_on_the_pin_exits_only_holding_elsewhere_and_refused_flat_elsewhere(traded, holding, mode):
    assert router.start_mode("binance", traded, holding) == mode


def _closed(store, name="s"):
    store.record_order(name, order_id=f"{name}-x", side="SELL", qty=1.0, intent="exit", reason="r")
    store.record_fill(name, side="SELL", qty=1.0, price=101.0, fee=0.0, order_id=f"{name}-x", trade_id="t2")


def test_venue_start_reads_the_journal_holding_elsewhere_then_flat_elsewhere(store):
    store.create_sleeve(name="new", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE")
    assert store.venue_start(store.sleeve("new")) == (router.NORMAL, None)
    _held_elsewhere(store)
    assert store.venue_start(store.sleeve("s")) == (router.EXITS_ONLY, "KRAKEN")
    _closed(store)
    assert store.venue_start(store.sleeve("s")) == (router.REFUSE, "KRAKEN")  # its exit went where the position was


def test_venue_start_takes_dust_as_flat(store):
    _held_elsewhere(store)
    store.record_order("s", order_id="s-x", side="SELL", qty=0.99999, intent="exit", reason="r")
    store.record_fill("s", side="SELL", qty=0.99999, price=101.0, fee=0.0, order_id="s-x", trade_id="t2")
    with store.engine.begin() as c:  # the exit went where the position is (rule A); only the dust is left
        assert c.execute(text("SELECT venue FROM orders WHERE order_id = 's-x'")).scalar() == "KRAKEN"
    assert store.venue_start(store.sleeve("s")) == (router.REFUSE, "KRAKEN")


def _events(store, kind, name="s"):
    return [e for e in store.events(name, limit=1000) if e["kind"] == kind]


def test_the_paper_process_builds_a_holder_on_its_positions_venue_paused_for_its_exits_only(store, capfd):
    from sleeve_fund.paper import node

    _held_elsewhere(store)
    row = node.venue_start(store, store.sleeve("s"))
    assert row.venue == "KRAKEN" and from_store(row).venue == "KRAKEN"
    s = store.sleeve("s")
    assert (s.status, s.status_reason) == ("paused", VENUE_EXITS) and row.status_reason == VENUE_EXITS
    (ev,) = _events(store, "venue_mismatch_exits_only")
    assert ev["level"] == "error" and not re.search("BINANCE|KRAKEN", ev["message"], re.I)
    assert "journal KRAKEN, pin BINANCE" in capfd.readouterr().out  # the log names them
    node.venue_start(store, store.sleeve("s"))  # the supervisor's start, or a restart: said once
    assert len(_events(store, "venue_mismatch_exits_only")) == 1


def test_the_paper_process_refuses_a_flat_strategy_last_traded_elsewhere_with_one_alert(store):
    from sleeve_fund.paper import node

    _held_elsewhere(store)
    _closed(store)  # flat, its last trade on another venue than its pin
    assert node.venue_start(store, store.sleeve("s")) is None
    (ev,) = _events(store, "venue_mismatch")
    assert ev["level"] == "error" and "won't start" in ev["message"]
    assert not re.search("BINANCE|KRAKEN", ev["message"], re.I)


def test_the_paper_process_starts_a_strategy_on_its_pin_unchanged(store):
    from sleeve_fund.paper import node

    store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE")
    row = store.sleeve("s")
    assert node.venue_start(store, row) is row and store.events("s", limit=10) == []


# --- the supervisor: the same start, and exits-only ends once flat --------------------------------------------------

def _supervised(store, monkeypatch):
    from sleeve_fund import supervisor

    started, stopped = [], []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: started.append(a[0][-1]) or
                        SimpleNamespace(pid=1, poll=lambda: None, send_signal=lambda *a: None, wait=lambda **k: 0))
    sup = supervisor.Supervisor(store)
    monkeypatch.setattr(sup, "_stop", lambda name, proc, why: stopped.append((name, why)) or
                        setattr(proc, "popen", None))
    return sup, started, stopped


def test_the_supervisor_starts_a_holder_elsewhere_for_its_exits_only_then_stops_and_refuses_it_once_flat(
        store, monkeypatch):
    _held_elsewhere(store)
    sup, started, stopped = _supervised(store, monkeypatch)
    sup.step()
    s = store.sleeve("s")
    assert started == ["s"] and (s.status, s.status_reason) == ("paused", VENUE_EXITS)
    assert [e["kind"] for e in _events(store, "start_refused") + _events(store, "incident")] == ["start_refused",
                                                                                               "incident"]
    sup.step()  # still holding: it keeps running
    assert stopped == [] and started == ["s"]
    _closed(store)
    sup.step()
    s = store.sleeve("s")
    assert [n for n, _ in stopped] == ["s"] and (s.desired_state, s.status) == ("stopped", "stopped")
    assert s.status_reason == f"not started: {router.VENUE_MOVED[router.REFUSE]}"
    (ev,) = _events(store, "venue_exits_done")
    assert ev["level"] == "info" and not re.search("BINANCE|KRAKEN", ev["message"], re.I)
    for _ in range(3):
        sup.step()
    assert started == ["s"]  # never started again on its own
    store.set_desired_state("s", "running")  # the PM's Start
    sup.step()
    assert started == ["s"] and store.sleeve("s").status == "stopped"
    assert len(_events(store, "start_refused")) == 2  # the exits-only start's, and this refusal's


def test_the_supervisor_refuses_a_flat_strategy_last_traded_elsewhere(store, monkeypatch):
    _held_elsewhere(store)
    _closed(store)
    sup, started, _ = _supervised(store, monkeypatch)
    sup.step()
    s = store.sleeve("s")
    assert started == [] and (s.desired_state, s.status) == ("stopped", "stopped")
    (ev,) = _events(store, "start_refused")
    assert ev["level"] == "error" and not re.search("BINANCE|KRAKEN", ev["message"], re.I)


# --- the refusal stops the order before any venue (DA) --------------------------------------------------------------

def test_through_papers_queued_journal_an_opening_order_is_refused_at_once_and_exits_land_on_the_positions_venue(
        store):
    from sleeve_fund.paper.queued import QueuedStore

    _held_elsewhere(store)
    q = QueuedStore(store, retry_seconds=())
    with pytest.raises(ValueError, match=MISMATCH):  # on the strategy's own call, before it sends anything
        q.record_order("s", order_id="s-n", side="BUY", qty=1.0, intent="entry", reason="r")
    q.record_order("s", order_id="s-x", side="SELL", qty=1.0, intent="stop_loss", reason="r")
    q.settle()
    assert _count(store, "orders", "s-n") == 0 and _venue(store, "s-x") == "KRAKEN"


def _submitter(on_order):
    """LongFlatStrategy._submit on a stand-in with just what a market order needs; returns (submit, sent)."""
    from decimal import Decimal

    from nautilus_trader.model import OrderSide

    from sleeve_fund.strategies.base import LongFlatStrategy

    sent = []
    s = SimpleNamespace(
        _gated=lambda side, intent: None, clock=SimpleNamespace(timestamp_ns=lambda: 0),
        instrument=SimpleNamespace(size_precision=3, price_increment=SimpleNamespace(as_double=lambda: 0.1)),
        _cfg=SimpleNamespace(maker_wait_minutes=0, instrument_id="BTCUSDT-PERP.BINANCE"), _price=lambda: 100.0,
        _outage_book=None, _deciding_bar=None, _margin=True, _book_stop_at=None, decisions={}, _unsettled=lambda: False,
        _backtest=False, fee_model=None, _deciding=None, simulated_venue=False, _sent=[], _kept={}, _maker={},
        order_factory=SimpleNamespace(market=lambda **k: SimpleNamespace(client_order_id="C-1", **k)),
        runtime=SimpleNamespace(on_order=on_order, on_timing=lambda *a, **k: None), submit_order=sent.append)
    return (lambda intent: LongFlatStrategy._submit(s, OrderSide.BUY, Decimal("1"), intent, "r", {})), sent


@pytest.mark.parametrize("intent", ["entry", "rebalance", "stop_loss", "exit"])
def test_every_order_is_journaled_before_it_is_sent_so_a_refused_row_sends_nothing(intent):
    def refuse(**_):
        raise ValueError("an opening order is refused while its open position is on another venue than its pin")

    submit, sent = _submitter(refuse)
    with pytest.raises(ValueError, match=MISMATCH):
        submit(intent)
    assert sent == []
    submit, sent = _submitter(lambda **_: None)  # control: the same order goes once its row is written
    assert submit(intent) and len(sent) == 1


# --- one holding rule, exact on both engines (DA) -------------------------------------------------------------------

def test_a_book_flat_by_fills_a_float_sum_misses_is_flat_and_an_entry_goes_on_the_pin(store):
    """0.1 + 0.2 bought and 0.3 sold sum to about 5.6e-17 in SQLite's floats: summed as the decimals each fill reads
    back as, the book is flat, so nothing about a position refuses the entry and it is written on the pin."""
    store.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, venue="BINANCE")
    for i, (side, qty) in enumerate((("BUY", 0.1), ("BUY", 0.2), ("SELL", 0.3))):
        intent = "entry" if side == "BUY" else "exit"
        store.record_order("s", order_id=f"s-{i}", side=side, qty=qty, intent=intent, reason="r")
        store.record_fill("s", side=side, qty=qty, price=100.0, fee=0.0, order_id=f"s-{i}", trade_id=f"t{i}")
    with store.engine.begin() as c:  # its orders went to another venue than its pin
        c.execute(text("UPDATE orders SET venue = 'KRAKEN' WHERE sleeve = 's'"))
    assert store.position_venue("s") is None
    store.record_order("s", order_id="s-n", side="BUY", qty=1.0, intent="entry", reason="r")
    assert _venue(store, "s-n") == "BINANCE"


def test_exits_only_close_then_refused_then_reset_then_a_normal_start_on_the_pin(store, monkeypatch):
    """DA: the whole sequence, with 0020's insert-only pin: the reset copies the pin to the run before the orders move,
    so no pin trigger trips, and the fresh run starts normally on its pin."""
    _held_elsewhere(store)
    sup, started, stopped = _supervised(store, monkeypatch)
    sup.step()
    assert store.sleeve("s").status_reason == VENUE_EXITS
    _closed(store)
    sup.step()
    assert store.sleeve("s").status == "stopped" and store.venue_start(store.sleeve("s"))[0] == router.REFUSE
    store.request_reset("s", "Reset strategy: back on its own venue")
    sup.step()  # carried out: the run so far is put away under its own name
    (run,) = [n for n in store.archived() if n.startswith("s--")]
    assert store.venue_start(store.sleeve("s")) == (router.NORMAL, None)
    assert router.venue_for(store.sleeve(run).venue) == router.venue_for(store.sleeve("s").venue) == "BINANCE"
    assert store.traded_venue(run) == "KRAKEN" and store.fills("s") == []
    store.set_desired_state("s", "running")  # the PM's Start
    sup.step()
    assert started == ["s", "s"] and _events(store, "start_refused") == []  # started, not refused (the run has those)
    store.record_order("s", order_id="s-n", side="BUY", qty=1.0, intent="entry", reason="r")
    assert _venue(store, "s-n") == "BINANCE"
