"""PM commands around a liquidation (Head of QA, 6 Oct 23:46; HoE Done-when for the stop-safety PR).
P1-D23: a liquidation order that hangs never blocks Stop, and the commands waiting behind it raise one incident.
P1-D24: a reset asked before a liquidation that lands while it waits is dropped; the liquidation halt stays."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from sleeve_fund import supervisor as sup
from sleeve_fund.paper.runtime import LIQ_STUCK_HEAD, WIPED_OUT, SleeveRuntime, liquidation_head
from sleeve_fund.store import Store

def _holding(store):
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced")
    store.record_fill("s1", side="BUY", qty=0.01, price=60_000.0, fee=0.3, order_id="e1", trade_id="e1")


def _incidents(store):
    return [e["message"] for e in store.events("s1", limit=1000) if e["kind"] == "incident"]


def test_a_stuck_liquidation_order_raises_one_incident_naming_what_waits_behind_it(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _holding(store)
    t = [datetime.now(timezone.utc)]  # after its fill, as at the venue
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0005)
    store.command("s1", "flatten", "close it")
    for _ in range(4):  # two minutes: not yet stuck
        t[0] += timedelta(seconds=30)
        rt.tick(equity=9_000, price=50_000, cash=9_400, qty=0.01, liquidating=True)
    assert not [m for m in _incidents(store) if m.startswith(LIQ_STUCK_HEAD)]
    for _ in range(12):  # past five minutes, still working
        t[0] += timedelta(seconds=30)
        rt.tick(equity=9_000, price=50_000, cash=9_400, qty=0.01, liquidating=True)
    (inc,) = [m for m in _incidents(store) if m.startswith(LIQ_STUCK_HEAD)]
    assert "the PM's flatten waits behind it" in inc and "Stop is still taken" in inc
    assert [c["command"] for c in store.pending_commands("s1")] == ["flatten"]  # parked, not lost
    t[0] += timedelta(seconds=30)
    rt.tick(equity=9_000, price=50_000, cash=9_400, qty=0.01)  # the order finished: the clock starts again
    assert rt.liq_working_since is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund.dashboard import app as app_mod

    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def _stop(c):
    return c.post("/sleeves/s1/command", data={"command": "stop", "reason": "checked"}, auth=("pm", "test-pw"),
                  headers={"origin": "http://testserver"}, follow_redirects=False)


@pytest.mark.parametrize("holding", [True, False])
def test_stop_is_always_taken_and_a_waiting_flatten_is_kept_only_while_there_is_something_to_sell(client, holding):
    c, store = client
    _holding(store)
    if not holding:
        store.record_fill("s1", side="SELL", qty=0.01, price=60_000.0, fee=0.3, order_id="x1", trade_id="x1")
    store.command("s1", "pause", "hold on")
    store.command("s1", "flatten", "close it")
    store.heartbeat("s1")  # the process is reporting
    r = _stop(c)
    assert "command_error" not in r.headers.get("location", ""), r.headers.get("location")
    assert store.sleeve("s1").desired_state == "stopped"
    # The pause lapses; the flatten is kept for the exits-only run while it holds (P1-U35), else it lapses too
    assert [x["command"] for x in store.pending_commands("s1")] == (["flatten"] if holding else [])


def test_the_supervisor_lapses_a_kept_flatten_once_it_stops_a_flat_strategy(tmp_path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _holding(store)
    store.record_fill("s1", side="SELL", qty=0.01, price=60_000.0, fee=0.3, order_id="x1", trade_id="x1")
    store.set_desired_state("s1", "stopped")
    store.command("s1", "flatten", "kept through a Stop")

    class FakePopen:
        pid, returncode = 4242, None
        poll = lambda self: None  # noqa: E731
        send_signal = wait = kill = lambda self, *a, **k: 0  # noqa: E731
    sv = sup.Supervisor(store)
    sv.procs["s1"] = sup.Proc(popen=FakePopen(), started_at=sup.utcnow())
    sv.step()
    assert sv.procs["s1"].popen is None and not store.pending_commands("s1")


def test_a_reset_asked_before_a_liquidation_is_dropped_and_the_halt_stays(tmp_path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    _holding(store)
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: pytest.fail("nothing starts"))
    store.request_reset("s1", "start again")
    sv = sup.Supervisor(store)
    sv.reset_pending()  # holding: it queues the flatten first
    assert [x["command"] for x in store.pending_commands("s1")] == ["flatten"]
    # The liquidation lands while the reset waits, and leaves it flat
    store.record_fill("s1", side="SELL", qty=0.01, price=54_000.0, fee=0.3, order_id="liq", trade_id="liq")
    halt = f"{WIPED_OUT}: 600.00, 100% of strategy equity at entry"
    store.event("s1", "error", "risk_halt", halt)
    store.set_status("s1", "halted", halt)
    sv.reset_pending()
    assert not store.pending_resets() and not store.reset_runs()  # dropped, nothing put away
    s = store.sleeve("s1")
    assert (s.status, s.status_reason) == ("halted", halt) and liquidation_head(store, "s1") == halt
    (ev,) = [e for e in store.events("s1", limit=100) if e["kind"] == "reset_dropped"]
    assert "only a reset after liquidation clears that" in ev["message"]
    assert [d for d in store.decisions("s1") if d["action"] == "drop reset"]
