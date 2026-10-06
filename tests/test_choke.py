"""CHOKE (HoE 6 Oct 20:52, Advisor): one "nothing opens" gate. entry_blocked says whether anything may open or add, and
why; the engine checks it on every order that would make the position bigger, at submit and at fill, and cancels
resting entries while it is closed; the supervisor's Stop and the dashboard's Start and Resume read the same answer.
Stops, exits, closes and liquidations are never gated. QA's invariant test is its own file."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.paper.journal import MemoryJournal
from sleeve_fund.paper.runtime import WIPED_OUT, SleeveRuntime, entry_blocked, entry_blocked_in
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import LongFlatStrategy
from test_sleeve_runtime import store  # noqa: F401  (Postgres when TEST_DATABASE_URL is set)

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def _state(status="running", reason="", paused_until=None, desired="running"):
    return SimpleNamespace(status=status, status_reason=reason, paused_until=paused_until, desired_state=desired)


@pytest.mark.parametrize("state,kw,why", [
    (_state(), {}, None),
    (_state(), {"liquidated": WIPED_OUT}, "only a reset after liquidation clears that"),
    (_state("halted", f"{WIPED_OUT}: 3,328.70, 33% of strategy equity at entry"), {}, "only a reset after liquidation"),
    (_state("halted", "drawdown 21% from peak"), {}, "it is halted, and only a resume clears that"),
    (_state("paused", "daily loss", NOW + timedelta(hours=12)), {}, "only the next 00:00 UTC roll clears that"),
    (_state("paused", "daily loss", NOW - timedelta(minutes=1)), {}, None),  # past its roll: the runtime lifts it
    (_state("paused", "paused by PM"), {}, "it is paused"),
    (_state("paused", "paused by PM"), {"starting": True}, None),  # Start and Resume act on it themselves
    (_state("stopped", desired="stopped"), {}, "it is stopped, and only Start clears that"),
    (_state(desired="stopped"), {}, "it is stopped"),  # a Stop the supervisor hasn't acted on yet
    (_state("stopped", desired="stopped"), {"starting": True}, None),  # Start is what clears a stop
    (_state(), {"holds": {"funding": "no funding rate for the next settlement"}}, "no funding rate"),
])
def test_entry_blocked_names_what_alone_clears_each_state(state, kw, why):
    blocked, said = entry_blocked(state, NOW, **kw)
    assert blocked == (why is not None) and (said is None if why is None else why in said), said


def test_a_paper_runtime_is_blocked_by_a_stop_not_yet_acted_on_and_by_a_hold(store):
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced")
    rt = SleeveRuntime(store, "s1", now=lambda: NOW)
    rt.on_start(0.008)
    assert rt.entry_blocked() == (False, None)
    rt.holds["data"] = "the last candle is stale"
    assert rt.entry_blocked() == (True, "the last candle is stale")
    rt.holds.clear()
    store.set_desired_state("s1", "stopped")
    assert rt.entry_blocked() == (True, "it is stopped, and only Start clears that")


def test_nothing_opens_or_adds_while_the_gate_is_closed_and_exits_still_run(prices, instrument, monkeypatch):
    """A backtest whose gate closes half way: after that no entry or rebalance is sent (one note says why), and the
    position it held is still closed by its own exit."""
    first = run_backtest("ping_pong", prices, instrument, {"rise": 0.01, "dip": 0.005}, risk_profile="balanced")
    entries = sorted(o["ts"] for o in first.journal.orders_.values() if o["intent"] == "entry")
    assert len(entries) > 4
    cut = entries[len(entries) // 2] - timedelta(seconds=1)  # just before an entry the open gate sends

    def gate(self):
        return (True, "held for the test") if self.now() >= cut else (False, None)
    monkeypatch.setattr(SleeveRuntime, "entry_blocked", gate)
    res = run_backtest("ping_pong", prices, instrument, {"rise": 0.01, "dip": 0.005}, risk_profile="balanced")
    after = [o for o in res.journal.orders_.values() if o["ts"] >= cut]
    assert not [o for o in after if o["intent"] in ("entry", "rebalance")], after[:3]
    before = [o for o in res.journal.orders_.values() if o["ts"] < cut]
    if before[-1]["intent"] == "entry":  # it held a position when the gate closed: its exit still went
        assert after and after[0]["intent"] != "entry"
    (note,) = [e for e in res.journal.events_ if e["kind"] == "entry_gated"]
    assert note["message"].startswith("Order not sent: it would open or add to the position, and held for the test")


def test_a_fill_that_adds_while_the_gate_is_closed_is_kept_and_opens_one_incident():
    """An entry sent before the gate closed and filled after it: kept (never flattened), one incident per order."""
    j = MemoryJournal()
    rt = SimpleNamespace(store=j, name="s", now=lambda: NOW,
                         entry_blocked=lambda: (True, "it is halted, and only a resume clears that"))
    me = SimpleNamespace(runtime=rt, _gated_fills=set(), decisions={"e1": {"intent": "entry"}, "x1": {"intent": "exit"}})
    for coid in ("e1", "e1", "x1"):  # two slices of the entry, and an exit
        LongFlatStrategy._gated_fill(me, coid, 0.1, 60_000.0)
    (inc,) = [e for e in j.events_ if e["kind"] == "incident"]
    assert inc["level"] == "error" and "filled while nothing may open (it is halted" in inc["message"]
    assert "kept with its stop, not closed" in inc["message"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund.dashboard import app as app_mod

    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced", desired_state="stopped")
    return TestClient(app_mod.create_app(store)), store


def _cmd(c, command, reason="checked"):
    r = c.post("/sleeves/s1/command", data={"command": command, "reason": reason}, auth=("pm", "test-pw"),
               headers={"origin": "http://testserver"}, follow_redirects=False)
    return r.headers.get("location", "")


def test_start_reads_the_journal_so_a_liquidation_overwritten_by_a_stop_is_still_refused(client):
    c, store = client
    store.event("s1", "error", "risk_halt", f"{WIPED_OUT}: 3,328.70, 33% of strategy equity at entry")
    store.set_status("s1", "stopped", "stopped by PM")  # an older supervisor wrote over the halt
    assert entry_blocked_in(store, "s1", starting=True) == (
        True, "it was liquidated, and only a reset after liquidation clears that")
    assert "command_error" in _cmd(c, "start") and store.sleeve("s1").desired_state == "stopped"
