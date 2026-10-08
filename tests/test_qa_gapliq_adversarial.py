"""QA regression cells ported from the #155 adversarial v2 probes, as QA agent sgz adapted them for the stop-safety
branch (PR #182): ADV-7 (P1-U34), ADV-8 (P1-D22), ADV-9 (P1-D24) and ADV-10 (P1-D23). Every finding is closed, so the
strict xfail marks are gone and each cell is a plain test. Assertions, expected values and set-ups are the source's.

Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/gate-stop-choke-gapliq-scripts/test_155_adv_v2_adapted_sgz.py
(adapted from degraded-155-scripts/test_155_delta_adversarial_v2.py; built on tests/test_degraded_155_qa.py, whose
autouse _reg fixture applies here too).

  ADV-7  P1-U34: a Resume, Flatten or Start queued while paused, landing on the liquidating tick: never running,
         nothing opens, and the command is taken up afterwards (no deadlock).
  ADV-8  P1-D22: a restart after downtime finds the price past the liquidation price (no stop resting).
  ADV-9  P1-D24: a Reset asked while paused and holding, carried out after the liquidation, must not clear the halt.
  ADV-10 P1-D23: a liquidation order that never fills: the PM's commands wait; Stop must still work.
"""
import dataclasses
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import test_degraded_155_qa as q  # noqa: E402
from test_degraded_155_qa import _reg  # noqa: E402,F401  (autouse fixture: the master's harness)
from sleeve_fund import risk  # noqa: E402
from sleeve_fund.store import Store  # noqa: E402

PERP = q.PERP
NAME = "ping-pong-test"


def _incidents(store, name):
    return [e for e in store.events(name, limit=5000) if e["kind"] == "incident"]


# ---- ADV-7 (P1-U34, from the #164 round, coordinator's required probe) ----------------------------------------------

class RunningForATick(AssertionError):
    """P1-U34: status 'running' written on the liquidating tick."""


# Only the cells P1-U34's mark covers are ported: the isolated resume cells, red on 81e6d6f (the SHA the mark names).
# The other ten (flatten, start and every full-margin cell) carry no finding and are never red on a pre-fix SHA
# (tests/QA_CELLS.md); they stay in QA's master.
ADV7 = [pytest.param(0.1, s, "resume", id=f"isolated-10pct-guard-path-{sid}-resume")
        for s, sid in (("ping_pong", "ping_pong"), ("qa_t", "always-short"))]


@pytest.mark.parametrize("pct, strategy, command", ADV7)
@pytest.mark.no_open_risk_limit  # QA sgz: guards off, liquidation mechanics only (full-margin set-up)
def test_adv7_p1_u34_a_command_queued_while_paused_lands_on_the_liquidating_tick(tmp_path, monkeypatch, pct,
                                                                               strategy, command):
    """P1-U34: the PM paused the strategy holding a short; a command is queued and is still pending at the tick that
    liquidates it (injected just before the first tick that sees the gap). It must not lift or override the
    liquidation halt, nothing may open after the liquidation, it ends halted in the ruled words, and the command is
    taken up once the liquidation has filled (no deadlock).

    Source: quant-review/v2-p1/gate-stop-choke-gapliq-scripts/test_155_adv_v2_adapted_sgz.py (ADV-7); finding P1-U34."""
    import test_replay
    from sleeve_fund.paper import runtime as rt_mod
    from sleeve_fund.research.replay import replay
    for nm, p in list(risk.PROFILES.items()):
        risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=pct)
    params = ({"rise": 0.01, "dip": 0.005, **PERP} if strategy == "ping_pong"
              else {**PERP, "at": 0, "side": -1, "tag": "u34"})  # qa_t: wants a short on every bar

    def meta(bal):
        m = q._session_meta(bal, params)
        m["sleeve"]["strategy"] = strategy
        return m
    store = Store(f"sqlite:///{tmp_path}/u34.db")
    statuses = []
    real_tick = rt_mod.SleeveRuntime.tick
    injected = []

    def tick(self, **kw):
        if kw.get("price", 0) > 60_000.0 * 1.015 * 1.3 and not injected:
            injected.append(self.now())
            if command == "start":  # the dashboard's Start on a paused strategy: desired state only
                self.store.set_desired_state(self.name, "running")
                self.store.decide("PM", "start", "QA U34: start while paused", self.name)
            else:
                self.store.command(self.name, command, f"QA U34: {command} queued while paused")
        out = real_tick(self, **kw)
        statuses.append((self.now(), self.status, kw.get("qty")))
        return out
    monkeypatch.setattr(rt_mod.SleeveRuntime, "tick", tick)
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        q._record_session(s1, meta(10_000), [(5, 0.0), (20, 0.015), (2, 0.0)])
        replay(s1, store=store)
        book = store.journal_book(NAME, 10_000)
        assert book["qty"] < 0  # harness: short held
        store.command(NAME, "pause", "QA: paused while holding")
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        test_replay.START = start0 + 2 * 3600 * 10**9
        s2 = tmp_path / "s2.jsonl.gz"
        bal = book["cash"] + book["qty"] * book["entry_px"]
        # after the gap the price dips and rises, so a running ping_pong would trade
        q._record_session(s2, meta(bal), [(5, 0.0), (0, 0.6), (3, -0.02), (3, 0.03), (4, -0.02)],
                          px=60_000.0 * 1.015)
        replay(s2, store=store)
    finally:
        test_replay.START = start0
    assert injected, statuses[-5:]  # harness: the command was pending at the gap tick
    orders = sorted(store.orders(NAME, limit=100_000), key=lambda o: (pd.Timestamp(o["ts"]), str(o["order_id"])))
    liq_at = [i for i, o in enumerate(orders) if o["intent"] == "liquidation"]
    assert liq_at, [(o["intent"], o["side"]) for o in orders]
    after = [(o["intent"], o["side"]) for o in orders[liq_at[-1] + 1:] if o["intent"] != "liquidation"
             and not (o["order_type"] == "STOP (watched)" and o["status"] == "canceled" and not o["filled_qty"])]
    assert not after, after  # nothing opens (or is sent) after the liquidation
    s = store.sleeve(NAME)
    assert s.status == "halted" and s.status_reason.startswith("Position margin lost (liquidated): "), \
        (s.status, s.status_reason)
    stuck = [c["command"] for c in store.pending_commands(NAME)]
    assert not stuck, stuck  # the queued command was taken up once the liquidation filled (no deadlock)
    running_after = [x for x in statuses if x[0] >= injected[0] and x[1] == "running"]
    if running_after:  # never running, not even for a tick
        raise RunningForATick(running_after[:3])


# ---- ADV-8 (the L18 hook's case without #146: a restart after downtime finds the price past liquidation) ------------

# [full-margin] carries no finding and is never red on a pre-fix SHA (tests/QA_CELLS.md); it stays in QA's master.
@pytest.mark.parametrize("pct", [pytest.param(0.1, id="isolated-10pct")])
@pytest.mark.no_open_risk_limit  # QA sgz: guards off, liquidation mechanics only
def test_adv8_a_restart_after_downtime_past_the_liquidation_price_liquidates_halts_and_opens_one_incident(tmp_path, pct):
    """No stop resting (so not GAP-LIQ). The short is opened, the process is down while the price moves 60% against
    it, and the restarted process's first prices are past liquidation: it is liquidated in the ruled words, one
    incident naming the equity left, nothing opens after it.

    Source: quant-review/v2-p1/gate-stop-choke-gapliq-scripts/test_155_adv_v2_adapted_sgz.py (ADV-8); finding P1-D22."""
    import test_replay
    from sleeve_fund.research.replay import replay
    for nm, p in list(risk.PROFILES.items()):
        risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=pct)
    params = {"rise": 0.01, "dip": 0.005, **PERP}
    store = Store(f"sqlite:///{tmp_path}/l18.db")
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        q._record_session(s1, q._session_meta(10_000, params), [(5, 0.0), (20, 0.015), (2, 0.0)])
        replay(s1, store=store)
        book = store.journal_book(NAME, 10_000)
        assert book["qty"] < 0
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        test_replay.START = start0 + 6 * 3600 * 10**9
        s2 = tmp_path / "s2.jsonl.gz"
        bal = book["cash"] + book["qty"] * book["entry_px"]
        q._record_session(s2, q._session_meta(bal, params), [(3, 0.0), (3, -0.02), (3, 0.03), (4, -0.02)],
                          px=60_000.0 * 1.015 * 1.6)
        orders = replay(s2, store=store)
    finally:
        test_replay.START = start0
    intents = [(o["intent"], o["side"]) for o in orders]
    li = [i for i, o in enumerate(orders) if o["intent"] == "liquidation"]
    assert li, intents
    assert not [x for x in intents[li[-1] + 1:] if x[0] != "liquidation"], intents
    s = store.sleeve(NAME)
    assert s.status == "halted" and s.status_reason.startswith("Position margin lost (liquidated): "), s.status_reason
    inc = _incidents(store, NAME)
    left = store.journal_book(NAME, 10_000)["cash"]
    assert len(inc) == 1, [e["message"] for e in inc]
    m = re.search(r"; ([\d,]+\.\d\d) of equity left", inc[0]["message"])
    assert m and float(m.group(1).replace(",", "")) == pytest.approx(max(left, 0.0), abs=0.011), (inc[0]["message"], left)


# ---- ADV-9 (U34 with a Reset): a per-strategy reset requested while paused and holding, carried out by the
# supervisor after the liquidation ------------------------------------------------------------------------------------

class FakePopen:
    pid, returncode = 4242, None
    poll = lambda self: None  # noqa: E731
    send_signal = wait = kill = lambda self, *a, **k: 0  # noqa: E731


def test_adv9_a_reset_requested_before_a_liquidation_does_not_clear_the_liquidation_halt(tmp_path, monkeypatch):
    """Advisor 20:41 (U27): an ordinary per-strategy Reset is refused while liquidated (it points to RAL). Here the PM
    asked for the reset while the strategy was paused and holding; the supervisor's flatten for it is still pending on
    the liquidating tick; once flat the supervisor would carry the reset out. It must not clear the liquidation halt:
    after the supervisor steps and a new process runs the next day, still halted in the ruled words, nothing trades.

    Source: quant-review/v2-p1/gate-stop-choke-gapliq-scripts/test_155_adv_v2_adapted_sgz.py (ADV-9); finding P1-D24.
    Set-up re-pinned by the Head of QA (the reset is asked before the liquidation); assertions unchanged."""
    import test_replay
    from sleeve_fund import supervisor as sup
    from sleeve_fund.research.replay import replay
    for nm, p in list(risk.PROFILES.items()):
        risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=0.1)
    params = {"rise": 0.01, "dip": 0.005, **PERP}
    store = Store(f"sqlite:///{tmp_path}/r.db")
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: FakePopen())
    from sleeve_fund.paper import runtime as rt_mod
    real_tick, injected, box = rt_mod.SleeveRuntime.tick, [], {}

    def tick(self, **kw):  # the supervisor's flatten for the reset lands on the liquidating tick
        if kw.get("price", 0) > 60_000.0 * 1.015 * 1.3 and not injected:
            injected.append(self.now())
            box["sv"].reset_pending()  # holding: it queues the flatten first
        return real_tick(self, **kw)
    monkeypatch.setattr(rt_mod.SleeveRuntime, "tick", tick)
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        q._record_session(s1, q._session_meta(10_000, params), [(5, 0.0), (20, 0.015), (2, 0.0)])
        replay(s1, store=store)
        book = store.journal_book(NAME, 10_000)
        assert book["qty"] < 0
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        store.command(NAME, "pause", "QA: paused while holding")
        # Re-pinned (HoQA 7 Oct 00:55 UK): the reset is asked here, paused and holding, before the liquidation is
        # booked; only the supervisor's step for it waits for the liquidating tick. The source asked it on that tick,
        # where main now refuses a reset of a liquidated strategy (U27), so the D24 path was no longer reached.
        store.request_reset(NAME, "QA: reset asked while paused and holding")
        sv = box["sv"] = sup.Supervisor(store)
        sv.procs[NAME] = sup.Proc()
        sv.procs[NAME].popen, sv.procs[NAME].started_at = FakePopen(), sup.utcnow()
        test_replay.START = start0 + 2 * 3600 * 10**9
        s2 = tmp_path / "s2.jsonl.gz"
        bal = book["cash"] + book["qty"] * book["entry_px"]
        q._record_session(s2, q._session_meta(bal, params), [(5, 0.0), (0, 0.6), (5, 0.0)], px=60_000.0 * 1.015)
        orders = replay(s2, store=store)
        intents = [(o["intent"], o["side"]) for o in orders]
        liquidated = any(o["intent"] == "liquidation" for o in orders)
        before = (store.sleeve(NAME).status, store.sleeve(NAME).status_reason)
        refused = None
        try:
            sv.reset_pending()  # flat now: the supervisor carries the reset out, or refuses it
        except ValueError as exc:
            refused = str(exc)
        n = len(store.orders(NAME, limit=100_000))
        test_replay.START = start0 + 26 * 3600 * 10**9  # the next day, past 00:00 UTC
        s3 = tmp_path / "s3.jsonl.gz"
        left = store.journal_book(NAME, 10_000)["cash"]
        q._record_session(s3, q._session_meta(left, params), [(3, -0.02), (3, 0.03), (4, -0.02)],
                          px=60_000.0 * 1.015 * 1.6)
        replay(s3, store=store)
        s = store.sleeve(NAME)
        status3 = (s.status, s.status_reason)
        n3 = len(store.orders(NAME, limit=100_000))
        store.command(NAME, "resume", "QA: a plain resume after the reset")  # a liquidation halt would refuse it
        test_replay.START = start0 + 28 * 3600 * 10**9
        s4 = tmp_path / "s4.jsonl.gz"
        q._record_session(s4, q._session_meta(left, params), [(3, -0.02), (3, 0.03), (4, -0.02)],
                          px=60_000.0 * 1.015 * 1.6)
        replay(s4, store=store)
        after_resume = [(o["intent"], o["side"]) for o in store.orders(NAME, limit=100_000)][: len(store.orders(
            NAME, limit=100_000)) - n3]
    finally:
        test_replay.START = start0
    s = store.sleeve(NAME)
    assert not after_resume, ("traded after a plain resume", after_resume, status3)
    events = [(e["kind"], e["message"][:100]) for e in sorted(store.events(NAME, limit=5000), key=lambda e: e["id"])][-12:]
    assert injected and liquidated, intents  # harness: the reset landed on the gap tick, which liquidated it
    allo = store.orders(NAME, limit=100_000)  # newest first
    new = allo[: len(allo) - n]
    assert not new, ([(o["intent"], o["side"]) for o in new], before, refused, events)
    assert s.status == "halted" and s.status_reason.startswith("Position margin lost (liquidated): "), \
        (s.status, s.status_reason, before, refused, events)


# ---- ADV-10 (U34's "commands wait": a liquidation order that never fills) -------------------------------------------

# [resume-waiting] carries no finding and is never red at an assertion (tests/QA_CELLS.md); it stays in QA's master.
@pytest.mark.parametrize("queued", [pytest.param(("resume", "flatten"), id="flatten-waiting")])
def test_adv10_a_liquidation_order_that_never_fills_parks_the_pms_commands_but_stop_still_works(tmp_path, queued):
    """The runtime skips every PM command while a liquidation order is working. If that order never fills (stuck at
    the venue), say what happens: the commands stay queued (none lost), and the PM's Stop is still taken (Advisor
    18:17: Stop always works), even with a Flatten waiting behind the stuck order.

    Source: quant-review/v2-p1/gate-stop-choke-gapliq-scripts/test_155_adv_v2_adapted_sgz.py (ADV-10); finding P1-D23."""
    from datetime import datetime, timedelta, timezone
    from fastapi.testclient import TestClient
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.paper.runtime import SleeveRuntime
    os.environ.setdefault("DASHBOARD_PASSWORD", "qa-pw")
    os.environ.setdefault("TEARSHEET_DIR", str(tmp_path))
    store = Store(f"sqlite:///{tmp_path}/s.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000, risk_profile="balanced",
                        params={"rise": 0.01, "dip": 0.005, **PERP})
    t = [datetime.now(timezone.utc) - timedelta(minutes=30)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0005)
    mark = {"cash": 0.0, "qty": -0.1}
    rt.tick(equity=10_000, price=60_000, **mark)
    store.command("s1", "pause", "holding")
    t[0] += timedelta(minutes=1)
    rt.tick(equity=10_000, price=60_000, **mark)
    c = TestClient(app_mod.create_app(store))
    auth, same = ("pm", os.environ["DASHBOARD_PASSWORD"]), {"origin": "http://testserver"}
    for cmd in queued:
        store.command("s1", cmd, f"QA: {cmd}")
    for _ in range(20):  # ten minutes of ticks with the liquidation order stuck working
        t[0] += timedelta(seconds=30)
        rt.tick(equity=9_000, price=66_000, liquidating=True, **mark)
    pending = [x["command"] for x in store.pending_commands("s1")]
    assert pending == list(queued), pending  # parked, not lost
    assert store.sleeve("s1").status == "paused"
    store.heartbeat("s1")  # the process is reporting
    r = c.post("/sleeves/s1/command", data={"command": "stop", "reason": "QA stop while the liquidation is stuck"},
               auth=auth, headers=same, follow_redirects=False)
    s = store.sleeve("s1")
    assert s.desired_state == "stopped", (r.status_code, r.headers.get("location"), r.text[:300])
