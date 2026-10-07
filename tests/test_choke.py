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
from sleeve_fund.paper.runtime import CODES, LABELS, WIPED_OUT, SleeveRuntime, blocked_state, entry_blocked
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store, utcnow
from sleeve_fund.strategies.base import LongFlatStrategy
from test_sleeve_runtime import store  # noqa: F401  (Postgres when TEST_DATABASE_URL is set)

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def _state(status="running", reason="", paused_until=None, desired="running"):
    return SimpleNamespace(status=status, status_reason=reason, paused_until=paused_until, desired_state=desired)


@pytest.mark.parametrize("state,kw,codes,words", [
    (_state(), {}, (), None),
    (_state(), {"liquidated": WIPED_OUT}, ("liquidated",),
     "Liquidated: position margin lost. Only a reset after liquidation clears it."),
    (_state("halted", f"{WIPED_OUT}: 3,328.70, 33% of strategy equity at entry"), {}, ("liquidated",),
     "Liquidated: position margin lost, 3,328.70, 33% of strategy equity at entry. Only a reset after liquidation"),
    (_state("halted", "drawdown 15.3% hit the 15% limit"), {"since": NOW - timedelta(hours=2)}, ("drawdown_halt",),
     "Drawdown halt: down 15.3% against a 15% limit since 10:00 UTC. Only you can clear it, with Resume."),
    (_state("halted", "invalid equity data"), {}, ("halted",), "Halted: invalid equity data. Only you can clear it, with Resume."),
    (_state("paused", "daily loss 3.2% hit the 3% limit", NOW + timedelta(hours=12)), {}, ("daily_pause",),
     "Daily loss pause: down 3.2% today against a 3% limit. It clears at the 00:00 UTC roll on 07 Oct."),
    (_state("paused", "daily loss", NOW - timedelta(minutes=1)), {}, (), None),  # past its roll: the runtime lifts it
    (_state("paused", "paused by PM: news"), {}, ("paused",), "Paused: paused by PM: news. Only Resume clears it."),
    (_state("paused", "paused by PM"), {"starting": True}, (), None),  # Start and Resume act on it themselves
    (_state("paused", "exits only: stopped"), {}, ("exits_only",), "Exits only: it still holds a position"),
    (_state("stopped", desired="stopped"), {}, ("stopped",), "Stopped: only Start clears it."),
    (_state(desired="stopped"), {}, ("stopped",), "Stopped"),  # a Stop the supervisor hasn't acted on yet
    (_state("stopped", desired="stopped"), {"starting": True}, (), None),  # Start is what clears a stop
    (_state(), {"holds": {"funding": "no funding rate for the next settlement"}}, ("funding_missing",),
     "No funding rate: no funding rate for the next settlement."),
    (_state("paused", "paused by PM"), {"holds": {"data": "the latest candle is degraded"}},
     ("paused", "degraded_candle"), "Paused: paused by PM. Only Resume clears it. Degraded candle: the latest candle"),
    (_state(), {"holds": {"data": "the last candle is stale"}}, ("stale_data",), "Stale data: the last candle is stale."),
    (_state(), {"archived": True}, ("retired",), "Retired: cannot be started."),
    (_state("stopped", desired="stopped"), {"archived": True, "starting": True}, ("retired",), "Retired"),
    # Every cause that applies, in the codes' order (Advisor 00:20)
    (_state("halted", "drawdown 21% hit the 20% limit", desired="stopped"), {"archived": True,
     "holds": {"funding": "none yet"}}, ("drawdown_halt", "retired", "stopped", "funding_missing"), "Drawdown halt"),
])
def test_blocked_state_lists_every_cause_with_its_code_figure_and_what_clears_it(state, kw, codes, words):
    blocked, said = blocked_state(state, NOW, **kw)
    assert blocked == bool(codes) and (said is None if not codes else said.codes == codes), said
    if codes:
        assert said.code == codes[0] and said.startswith(words), said
        assert all(f"{LABELS[c]}:" in said for c in codes) and set(codes) <= set(CODES)


def test_a_paper_runtime_is_blocked_by_a_stop_not_yet_acted_on_and_by_a_hold(store):
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced")
    rt = SleeveRuntime(store, "s1", now=lambda: NOW)
    rt.on_start(0.008)
    assert rt.entry_blocked() == (False, None)
    rt.holds["data"] = "the last candle is stale"
    assert rt.entry_blocked() == (True, "Stale data: the last candle is stale.")
    rt.holds.clear()
    store.set_desired_state("s1", "stopped")
    assert rt.entry_blocked() == (True, "Stopped: only Start clears it.")


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
    refused = [d for d in res.journal.decisions_ if d["action"] == "entry_blocked"]
    assert refused and all(d["reason"].endswith("would open or add to the position. held for the test")
                           for d in refused), refused[:2]


def test_a_fill_that_adds_while_the_gate_is_closed_is_kept_and_opens_one_incident():
    """An entry sent before the gate closed and filled after it: kept (never flattened), one incident per order."""
    j = MemoryJournal()
    rt = SimpleNamespace(store=j, name="s", now=lambda: NOW,
                         entry_blocked=lambda: (True, "it is halted, and only a resume clears that"))
    me = SimpleNamespace(runtime=rt, _gated_fills=set(), _slices={"s1": "k1", "s2": "k1"}, _ms_since_block=lambda why: 40,
                         decisions={"e1": {"intent": "entry"}, "x1": {"intent": "exit"}, "s1": {"intent": "entry"},
                                    "s2": {"intent": "entry"}})
    for coid in ("e1", "e1", "x1", "s1", "s2"):  # two fills of an entry, an exit, two slices of one kept entry
        LongFlatStrategy._gated_fill(me, coid, 0.1, 60_000.0)
    inc, kept = [e for e in j.events_ if e["kind"] == "incident"]  # one for the entry, one for the kept order
    assert inc["level"] == "error" and "filled while nothing may open: 0.1 at 60,000. it is halted" in inc["message"]
    assert "kept with its stop, not closed" in inc["message"]
    # Advisor 7 Oct 05:01: each raced order is on record with the ms after the block began, for fills-vs-model
    raced = [e["message"] for e in j.events_ if e["kind"] == "raced_fill"]
    assert raced == ["Raced fill: 0.1 at 60,000, 40 ms after nothing could open any more (it is halted, and only a "
                     "resume clears that)."] * 2, raced


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
    blocked, why = entry_blocked(store, "s1", starting=True)
    assert blocked and why.codes == ("liquidated",) and why.endswith("Only a reset after liquidation clears it.")
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
    assert entry_blocked(store, "s1") == (True, "Degraded candle: the latest candle is degraded.")  # no process needed
    assert entry_blocked(store, "s1")[1].codes == ("degraded_candle",)
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
    assert "Retired" in _cmd(c, "start") and store.sleeve("s1").desired_state == "stopped"
    assert "Retired" in _cmd(c, "resume")


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
    assert blocked and ("Only you can clear it" in why if status == "halted" else "00:00 UTC" in why), why


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


def test_the_pm_controls_act_on_a_strategy_whose_reset_is_under_way(client):
    """P1-KR-1, P1-KR-3 (Head of QA; Advisor 7 Oct: one mechanism, every PM control kept as the reset's hold). With a
    reset's flatten waiting, Flatten was refused and the kill switch skipped the strategy, and a Stop was overridden
    by the reset's restart: each fresh run traded. Now Flatten and the kill switch are kept as the fresh run's pause
    without a second sale, and a Stop leaves it stopped."""
    from sleeve_fund.supervisor import Supervisor

    c, store = client
    pm = {"auth": ("pm", "test-pw"), "headers": {"origin": "http://testserver"}, "follow_redirects": False}
    for name in ("s2", "s3"):
        store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                            starting_balance=10_000, risk_profile="balanced")
    for name in ("s1", "s2", "s3"):
        store.set_desired_state(name, "running")
        store.set_status(name, "running")
        store.record_fill(name, side="BUY", qty=0.01, price=60_000.0, fee=0.3, order_id="o1", trade_id="t1")
        assert c.post(f"/sleeves/{name}/reset", data={"reason": "Test finished"}, **pm).status_code == 303
    sup = Supervisor(store, python="true")
    sup.reset_pending()  # each reset's flatten waits for its process
    flat = c.post("/sleeves/s1/command", data={"command": "flatten", "reason": "Hold it"}, **pm).headers["location"]
    assert "command_error" not in flat
    assert c.post("/book/flatten", data={"reason": "Drawdown"}, **pm).headers["location"] == "/risk?killed=3"
    assert "command_error" not in c.post("/sleeves/s3/command", data={"command": "stop", "reason": "Done"},
                                         **pm).headers["location"]
    sup.reset_pending()  # s3's flatten went with its Stop: queued again, started only to sell
    for name in ("s1", "s2", "s3"):
        flattens = [cmd for cmd in store.pending_commands(name) if cmd["command"] == "flatten"]
        assert len(flattens) == 1, (name, flattens)  # one sale each, never a second
        store.mark_applied(flattens[0]["id"])
        store.record_fill(name, side="SELL", qty=0.01, price=60_000.0, fee=0.3, order_id="o2", trade_id="t2")
    sup.reset_pending()
    assert store.pending_resets() == []
    s1, s2, s3 = (store.sleeve(n) for n in ("s1", "s2", "s3"))
    assert s1.status == "paused" and s1.status_reason.startswith("flattened by PM: Book kill switch: Drawdown")
    assert s2.status == "paused" and s2.status_reason.startswith("flattened by PM: Book kill switch: Drawdown")
    assert s3.desired_state == "stopped"
