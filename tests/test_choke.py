"""CHOKE (HoE 6 Oct 20:52, Advisor): one "nothing opens" gate. entry_blocked says whether anything may open or add, and
why; the engine checks it on every order that would make the position bigger, at submit and at fill, and cancels
resting entries while it is closed; the supervisor's Stop and the dashboard's Start and Resume read the same answer.
Stops, exits, closes and liquidations are never gated. QA's invariant test is its own file."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import test_replay
from fastapi.testclient import TestClient

from sleeve_fund.paper.journal import MemoryJournal
from sleeve_fund.paper.runtime import WIPED_OUT, SleeveRuntime, blocked_state, entry_blocked
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store, utcnow
from sleeve_fund.strategies.base import LongFlatStrategy
from test_sleeve_runtime import store  # noqa: F401  (Postgres when TEST_DATABASE_URL is set)

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def _state(status="running", reason="", paused_until=None, desired="running"):
    return SimpleNamespace(status=status, status_reason=reason, paused_until=paused_until, desired_state=desired)


@pytest.mark.parametrize("state,kw,why", [
    (_state(), {}, None),
    (_state(), {"liquidated": WIPED_OUT}, "only a reset after liquidation clears that"),
    (_state("halted", f"{WIPED_OUT}: 3,328.70, 33% of strategy equity at entry"), {}, "only a reset after liquidation"),
    (_state("halted", "drawdown 21% from peak"), {}, "it is halted (drawdown 21% from peak), and only a resume clears"),
    (_state("paused", "daily loss", NOW + timedelta(hours=12)), {}, "only the next 00:00 UTC roll clears that"),
    (_state("paused", "daily loss", NOW - timedelta(minutes=1)), {}, None),  # past its roll: the runtime lifts it
    (_state("paused", "paused by PM"), {}, "it is paused"),
    (_state("paused", "paused by PM"), {"starting": True}, None),  # Start and Resume act on it themselves
    (_state("stopped", desired="stopped"), {}, "it is stopped, and only Start clears that"),
    (_state(desired="stopped"), {}, "it is stopped"),  # a Stop the supervisor hasn't acted on yet
    (_state("stopped", desired="stopped"), {"starting": True}, None),  # Start is what clears a stop
    (_state(), {"holds": {"funding": "no funding rate for the next settlement"}}, "no funding rate"),
    (_state("paused", "paused by PM"), {"holds": {"data": "the latest candle is degraded"}}, "candle is degraded"),
    (_state(), {"archived": True}, "it is retired (archived), and only Restore clears that"),
    (_state("stopped", desired="stopped"), {"archived": True, "starting": True}, "retired"),
])
def test_blocked_state_names_what_alone_clears_each_state(state, kw, why):
    blocked, said = blocked_state(state, NOW, **kw)
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
    refused = [d for d in res.journal.decisions_ if d["action"] == "entry_refused"]
    assert refused and all(d["reason"].endswith("would open or add to the position, and held for the test")
                           for d in refused), refused[:2]


def test_a_fill_that_adds_while_the_gate_is_closed_is_kept_and_opens_one_incident():
    """An entry sent before the gate closed and filled after it: kept (never flattened), one incident per order."""
    j = MemoryJournal()
    rt = SimpleNamespace(store=j, name="s", now=lambda: NOW,
                         entry_blocked=lambda: (True, "it is halted, and only a resume clears that"))
    me = SimpleNamespace(runtime=rt, _gated_fills=set(), _slices={"s1": "k1", "s2": "k1"},
                         decisions={"e1": {"intent": "entry"}, "x1": {"intent": "exit"}, "s1": {"intent": "entry"},
                                    "s2": {"intent": "entry"}})
    for coid in ("e1", "e1", "x1", "s1", "s2"):  # two fills of an entry, an exit, two slices of one kept entry
        LongFlatStrategy._gated_fill(me, coid, 0.1, 60_000.0)
    inc, kept = [e for e in j.events_ if e["kind"] == "incident"]  # one for the entry, one for the kept order
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
    assert entry_blocked(store, "s1", starting=True) == (
        True, "it was liquidated, and only a reset after liquidation clears that")
    assert "command_error" in _cmd(c, "start") and store.sleeve("s1").desired_state == "stopped"


def _stopped_holder(store, status="running", reason=""):
    from sleeve_fund import supervisor as sup

    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced")
    store.record_fill("s1", side="BUY", qty=0.01, price=60_000.0, fee=0.3, order_id="raced", trade_id="raced")
    store.set_desired_state("s1", "stopped")
    store.set_status("s1", status, reason)
    started = []

    class FakePopen:
        pid, returncode = 4242, None
        poll = lambda self: None  # noqa: E731
        send_signal = wait = kill = lambda self, *a, **k: 0  # noqa: E731
    sv = sup.Supervisor(store)
    return sv, started, FakePopen


def test_a_stopped_strategy_still_holding_runs_for_its_exits_only_with_one_incident(store, monkeypatch):
    """P1-U35 (Advisor 20:56, HoE): a raced remainder held by a STOPPED strategy is never left unwatched. The
    supervisor starts it for its exits only (its stop, or a safety stop, still closes it; nothing opens); the process
    it starts writes the one incident with its safety stop (test_restart_safety_stop), so the supervisor writes none;
    once flat it stops; the PM's Start then restarts it to trade."""
    from sleeve_fund import supervisor as sup
    from sleeve_fund.strategies.base import EXITS_ONLY

    sv, started, FakePopen = _stopped_holder(store, "stopped", "stopped by PM")
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: started.append(1) or FakePopen())
    sv.step()
    s = store.sleeve("s1")
    assert started == [1] and s.desired_state == "stopped"
    assert (s.status, s.status_reason) == ("paused", f"{EXITS_ONLY}: {sup.STOPPED_HOLDING}")
    assert entry_blocked(store, "s1")[0]  # nothing opens
    sv.step()
    sv.step()
    assert not [e for e in store.events("s1", limit=100) if e["kind"] == "incident"]
    assert started == [1]
    store.record_fill("s1", side="SELL", qty=0.01, price=60_100.0, fee=0.3, order_id="stop", trade_id="stop")
    sv.step()  # flat: stopped
    assert sv.procs["s1"].popen is None and store.sleeve("s1").status == "stopped"
    sv.step()
    assert started == [1]


def test_the_pms_stop_on_a_holder_restarts_it_for_its_exits_only_and_start_lets_it_trade_again(store, monkeypatch):
    from sleeve_fund import supervisor as sup

    sv, started, FakePopen = _stopped_holder(store)
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: started.append(1) or FakePopen())
    sv.procs["s1"] = sup.Proc(popen=FakePopen(), started_at=sup.utcnow())  # running when the PM pressed Stop
    sv.step()
    assert started == [1] and store.sleeve("s1").status == "paused"
    store.set_desired_state("s1", "running")  # the PM's Start
    sv.step()
    assert started == [1, 1] and store.sleeve("s1").status == "stopped"  # the new process's start sets it running


@pytest.mark.parametrize("status,reason", [("halted", "drawdown 21% from peak"), ("paused", "paused by PM")])
def test_a_stopped_holder_that_is_halted_or_paused_keeps_its_process_and_status_with_one_incident(
        store, monkeypatch, status, reason):
    from sleeve_fund import supervisor as sup

    sv, started, FakePopen = _stopped_holder(store, status, reason)
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: started.append(1) or FakePopen())
    sv.procs["s1"] = sup.Proc(popen=FakePopen(), started_at=sup.utcnow())
    sv.step()
    sv.step()
    s = store.sleeve("s1")
    assert started == [] and sv.procs["s1"].popen is not None and (s.status, s.status_reason) == (status, reason)
    assert len([e for e in store.events("s1", limit=100) if e["kind"] == "incident"]) == 1


def test_a_deploy_before_the_stopped_holder_is_flat_restarts_it_without_a_second_incident(store, monkeypatch):
    """A deploy restarts the exits-only process; the restarted process's safety-stop incident is written once per
    position (runtime.incident_once), so a second process start adds none."""
    from sleeve_fund import supervisor as sup

    sv, started, FakePopen = _stopped_holder(store, "stopped", "stopped by PM")
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: started.append(1) or FakePopen())
    sv.step()
    rt = SleeveRuntime(store, "s1", now=utcnow)  # the process it started writes its incident (after the fill)
    rt.incident_once("Incident, s1: the open long", "Incident, s1: the open long position of 0.01 ...")
    sup.Supervisor(store).step()  # a deploy: a new supervisor, its process gone with the old one
    SleeveRuntime(store, "s1", now=utcnow).incident_once("Incident, s1: the open long", "again")
    assert started == [1, 1]
    assert len([e for e in store.events("s1", limit=100) if e["kind"] == "incident"]) == 1


def test_a_block_episode_is_one_alert_and_one_cleared_event_with_its_refusals_and_the_gate_reads_it(store):
    """Advisor 22:29: one decision row per refused order, one alert when the block starts, one cleared event when it
    ends carrying the refusal count, no alert per bar; the store-level gate sees the engine's open episode."""
    from sleeve_fund.paper.runtime import BLOCK_CLEARED, BLOCK_STARTED

    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced")
    rt = SleeveRuntime(store, "s1", now=lambda: NOW)
    rt.on_start(0.008)
    rt.holds["data"] = "the latest candle is degraded"
    for _ in range(3):
        rt.entry_blocked()
    rt.refused("the latest candle is degraded", "entry buy 0.1 would open or add to the position")
    rt.refused("the latest candle is degraded", "entry buy 0.1 would open or add to the position")
    assert entry_blocked(store, "s1") == (True, "the latest candle is degraded")  # no process needed
    assert entry_blocked(store, "s1", starting=True) == (False, None)
    rt.holds.clear()
    assert rt.entry_blocked() == (False, None)
    kinds = [e["kind"] for e in store.events("s1", limit=50) if e["kind"] in (BLOCK_STARTED, BLOCK_CLEARED)]
    assert kinds == [BLOCK_CLEARED, BLOCK_STARTED]  # newest first: one alert, one cleared
    cleared = store.last_event("s1", (BLOCK_CLEARED,))
    assert cleared["message"].endswith("Orders refused while it held: 2")
    assert entry_blocked(store, "s1") == (False, None)


def test_an_archived_strategy_refuses_start_and_resume_naming_it_retired(client):
    c, store = client
    store.archive("s1")
    assert "retired" in _cmd(c, "start") and store.sleeve("s1").desired_state == "stopped"
    assert "retired" in _cmd(c, "resume")


@pytest.mark.parametrize("status,reason,until", [
    ("halted", "drawdown 21% from peak", None),
    ("paused", "daily loss 3.1% hit the 3% limit", datetime(2099, 1, 1, tzinfo=timezone.utc)),
])
def test_a_strategys_own_reset_never_clears_its_halt_or_daily_pause(store, status, reason, until):
    """Advisor 22:29: a per-strategy Reset never clears a halt or the daily pause (their own action does: a resume, or
    the 00:00 UTC roll); the gate stays closed through it, naming that action."""
    from sleeve_fund import supervisor as sup

    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced", desired_state="stopped")
    store.set_status("s1", status, reason, until)
    store.request_reset("s1", "testing finished", actor="PM")
    sup.Supervisor(store).reset_pending()
    s = store.sleeve("s1")
    assert s.status == status and store.pending_reset("s1") is None
    blocked, why = entry_blocked(store, "s1", starting=True)
    assert blocked and ("only a resume" in why if status == "halted" else "00:00 UTC" in why), why


@pytest.mark.no_open_risk_limit(reason="guards off: the 2x short must open in session 1 to be carried")
@pytest.mark.parametrize("holder", ["stopped_holder", "pm_paused"])
def test_sg5_a_holder_still_guarded_by_its_limits_never_opens_an_entry(tmp_path, holder, monkeypatch):
    """HoE 23:22 (SG5 done-when 2): the exits-only and PM-paused holders of QA's SG5 pins (test_gate_845_u35, where the
    drawdown halt, the daily-loss flatten, the liquidation check and the stop each still fire) never open an entry:
    across later sessions that trade both ways, quiet enough to stay inside every limit, nothing that grows the
    position is sent or filled."""
    import dataclasses

    from sleeve_fund import risk
    from test_gate_845_u35 import ENTRY, _deploys

    for nm, p in list(risk.PROFILES.items()):  # QA's whole_equity fixture, inline
        monkeypatch.setitem(risk.PROFILES, nm, dataclasses.replace(p, max_position_pct=1.0))
    swings = [(3, -0.004), (3, 0.006), (4, -0.004), (3, 0.005)]  # dips and rises ping_pong would trade on
    store = _deploys(tmp_path, holder, [(ENTRY, swings), (ENTRY, swings)])
    name = "ping-pong-test"
    orders = sorted(store.orders(name, limit=1000), key=lambda o: o["ts"])
    s2 = datetime.fromtimestamp(test_replay.START / 1e9, timezone.utc) + timedelta(hours=2)  # the holder's sessions
    held = [o for o in orders if o["ts"] < s2 and o["status"] == "filled"]
    after = [o for o in orders if o["ts"] >= s2]
    assert not [o for o in after if o["intent"] in ("entry", "rebalance")], [(o["ts"], o["intent"], o["side"])
                                                                            for o in after]
    carried = abs(sum(o["filled_qty"] * (1 if o["side"] == "BUY" else -1) for o in held))
    assert abs(store.journal_book(name, 10_000)["qty"]) <= carried + 1e-12
    kinds = {e["kind"] for e in store.events(name, limit=1000)}
    assert "entry_blocked" in kinds, kinds  # it did try to trade, and the gate refused it
