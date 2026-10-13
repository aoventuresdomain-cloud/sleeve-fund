"""RESTING-OFF (HoE, Advisor 13 Oct): paper sends no resting entry until ORPHAN-GUARD lands, which cancels one its
process can no longer watch. A Strategy whose entries rest (maker-first execution) is refused at start with the
reason; one at market starts as before; one holding a position runs for its exits only, as any refused start does."""
from __future__ import annotations

from types import SimpleNamespace

from sleeve_fund import supervisor
from sleeve_fund.store import Store

MAKER = {"market": "spot", "allow_short": False, "maker_wait_minutes": 9}


def _supervised(tmp_path, monkeypatch, sleeves):
    store = Store(f"sqlite:///{tmp_path}/r.db")
    for name, params in sleeves:
        store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                            starting_balance=1_000, venue="binance", params=params)
    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: started.append(a[0][-1]) or
                        SimpleNamespace(pid=1, poll=lambda: None, send_signal=lambda *a: None, wait=lambda **k: 0))
    return store, supervisor.Supervisor(store), started


def test_a_strategy_whose_entries_would_rest_is_refused_at_start_with_the_reason_and_one_at_market_starts(
        tmp_path, monkeypatch):
    store, sup, started = _supervised(tmp_path, monkeypatch, [("maker", MAKER), ("market", {"market": "spot"})])
    for _ in range(3):
        sup.step()
    assert started == ["market"]
    s = store.sleeve("maker")
    assert s.desired_state == "stopped" and s.status == "stopped"
    assert s.status_reason == f"not started: {supervisor.RESTING_OFF}"
    assert [e["kind"] for e in store.events("maker", limit=100)].count("start_refused") == 1  # once, no loop


def test_a_strategy_whose_entries_would_rest_still_holding_runs_for_its_exits_only(tmp_path, monkeypatch):
    store, sup, started = _supervised(tmp_path, monkeypatch, [("maker", MAKER)])
    store.record_order("maker", order_id="m-1", side="BUY", qty=0.01, intent="entry", reason="test")
    store.book_fill("maker", side="BUY", qty=0.01, price=60_000.0, fee=0.6, order_id="m-1", trade_id="t1")
    sup.step()
    s = store.sleeve("maker")
    assert started == ["maker"] and s.status == "paused"
    assert s.status_reason == f"{supervisor.EXITS_ONLY}: {supervisor.RESTING_OFF}"
