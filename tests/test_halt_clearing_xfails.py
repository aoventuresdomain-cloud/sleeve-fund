"""Halt clearing (BLOCKER on the #155 start gate, Advisor 6 Oct 18:17): strict xfails written by QA BEFORE the build.
The engineer makes each one pass (and removes its mark) before #155's start gate opens.

Source [R18:17] (advisor-rulings.md, "RAL + halts", Advisor 6 Oct 18:17 to QA): "HALTS (BLOCKER on #155 start gate): a
halt is cleared only by its own action (DD halt: resume; daily pause: next 00:00 UTC roll; liquidation: RAL);
Stop/Start, deploy restart, supervisor restart never clear it; Start refuses while halted naming the clearing action;
Stop always works. ... Falsifier: every halt state shows its clearing action on the dashboard."

QA's finding on 3be572a (and ba4f533) that this file grew from: a PM Stop then Start clears ANY halt. The supervisor
writes status "stopped" over "halted" or "paused" when it stops the process, and the next start sets it running. A
drawdown halt or a liquidation then halts again only because the first tick measures against the old high-water mark,
and a daily pause pauses again only because the day's open is still the first mark of the day: a fresh halt, not the
original one.

ASSUMED INTERFACES (adapt the names, never the assertions): as test_ral_xfails.py (whose helpers and fixtures this file
imports: copy both into tests/). Start refused while halted comes back from POST /sleeves/{name}/command as
command_error naming the clearing action: "resume" (drawdown halt), "00:00 UTC" (daily pause), "reset after
liquidation" (liquidation). A daily pause's paused_until is the next 00:00 UTC. The strategy page's halt banner (the
role="alert" block that begins "Halted:" or "Paused:") names the clearing action in those words.
Restarts use existing entry points: the supervisor (its subprocess replaced by a stand-in), the dashboard (its clock
frozen at the test's 2025 time where it posts Start, as it judges a pause against that clock), and a fresh SleeveRuntime
started and ticked on the journal (the paper process's own start). Every strategy places a 2% stop, so the stop-safety
gate (stopless capped at 1x) lets the supervisor start it.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from sleeve_fund.store import utcnow
from test_ral_xfails import (AUTH, GAP_PX, NAME, _liquidate, _ordinary, _post, _ral_events, _runtime,  # noqa: F401
                             _tick, client, store)

REASON = "QA halt clearing: BLOCKER on the #155 start gate (Advisor 18:17)"
xf = pytest.mark.xfail(strict=True, reason=REASON)

HALTS = ["drawdown_halt", "daily_pause", "liquidation"]
HALT_KINDS = ("risk_halt", "risk_pause")
CLEARED_BY = {"drawdown_halt": "resume", "daily_pause": "00:00 utc", "liquidation": "reset after liquidation"}
T = datetime(2025, 10, 3, 10, 0, tzinfo=timezone.utc)  # the ordinary halts happen here
ROLL = datetime(2025, 10, 4, 0, 0, tzinfo=timezone.utc)  # the next 00:00 UTC


class _Popen:
    """The paper process, for the supervisor: alive until signalled."""
    pid = 1

    def __init__(self, *a, **k):
        self.alive, self.returncode = True, None

    def poll(self):
        return None if self.alive else self.returncode

    def send_signal(self, _):
        self.alive, self.returncode = False, 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive, self.returncode = False, -9


def _halt(store, tmp_path, kind):
    """A strategy in one of the three halt states, its process running under a supervisor. `at` is a time before the
    next 00:00 UTC roll for the restarts; equity and price are what it marks at."""
    if kind == "liquidation":
        f = _liquidate(tmp_path, store)
        at, equity, price = f.liq["ts"] + timedelta(minutes=30), f.rem, GAP_PX
    else:
        _ordinary(store, kind, t=T)
        at, equity, price = T + timedelta(hours=2), 7_900.0 if kind == "drawdown_halt" else 9_400.0, 60_000.0
    s = store.sleeve(NAME)
    return SimpleNamespace(kind=kind, status=s.status, reason=s.status_reason, until=s.paused_until, at=at,
                           equity=equity, price=price, halts=_halts(store))


def _halts(store):
    return len([e for e in store.events(NAME, limit=10_000) if e["kind"] in HALT_KINDS])


def _dashboard_clock(monkeypatch, h):
    """The dashboard judges a pause against its own clock: frozen at the test's 2025 time (harness only)."""
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "utcnow", lambda: h.at)


def _supervised(store, monkeypatch):
    from sleeve_fund import supervisor

    monkeypatch.setattr(supervisor.subprocess, "Popen", _Popen)
    sup = supervisor.Supervisor(store)
    sup.step()
    assert sup.procs[NAME].alive  # setup: its process runs, halted (passes on 3be572a)
    return sup


def _process_start(store, h):
    """The paper process starting on the journal, ticked once at the halt's own equity and price."""
    return _runtime(store, h.at, h.equity, price=h.price)


def _kept(store, h, rt=None):
    """The original halt holds: same status, reason and pause end, no fresh halt or pause standing in for it, and
    nothing may open."""
    s = store.sleeve(NAME)
    assert (s.status, s.status_reason, s.paused_until) == (h.status, h.reason, h.until), (s.status, s.status_reason)
    assert _halts(store) == h.halts, [e["message"] for e in store.events(NAME, limit=10) if e["kind"] in HALT_KINDS]
    if rt is not None:
        assert not rt.can_open()


def _error(r):
    assert r.status_code == 303, r.status_code
    return parse_qs(urlparse(r.headers["location"]).query).get("command_error", [None])[0]


# --- Stop/Start, deploy restart, supervisor restart never clear a halt [R18:17] -----------------------------------

@pytest.mark.parametrize("kind", HALTS)
def test_stop_then_start_never_clears_a_halt(store, tmp_path, client, monkeypatch, kind):
    """[R18:17] "Stop/Start ... never clear it": the PM's Stop then Start on the dashboard, carried out by the
    supervisor, leaves the original halt in place (refused Start, or a start that keeps it). On 3be572a the
    supervisor writes "stopped" over it and the next start runs the strategy until a fresh halt or pause."""
    h = _halt(store, tmp_path, kind)
    _dashboard_clock(monkeypatch, h)
    sup = _supervised(store, monkeypatch)
    for command in ("stop", "start"):
        _post(client, {"command": command, "reason": f"{command} it"})
        sup.step()
    rt = _process_start(store, h) if store.sleeve(NAME).desired_state == "running" else None
    _kept(store, h, rt)


@pytest.mark.parametrize("kind", HALTS)
def test_guard_a_deploy_restart_never_clears_a_halt(store, tmp_path, monkeypatch, kind):
    """GUARD (passes on 3be572a) for [R18:17] "deploy restart never clears it": the supervisor shuts down (stopping
    every process, which runs its own on_stop), a new one starts every strategy, and the original halt holds."""
    from sleeve_fund import supervisor
    from sleeve_fund.paper.runtime import SleeveRuntime

    h = _halt(store, tmp_path, kind)
    old = _supervised(store, monkeypatch)
    for name, proc in old.procs.items():  # as Supervisor.run does when it is stopped
        old._stop(name, proc, "supervisor shutting down")
    SleeveRuntime(store, NAME, now=lambda: h.at).on_stop()  # the paper process's own shutdown
    new = supervisor.Supervisor(store)
    new.step()
    assert new.procs[NAME].alive
    _kept(store, h, _process_start(store, h))


@pytest.mark.parametrize("why", ["crash", "stale_heartbeat"])
@pytest.mark.parametrize("kind", HALTS)
def test_guard_a_supervisor_restart_never_clears_a_halt(store, tmp_path, monkeypatch, kind, why):
    """GUARD (passes on 3be572a) for [R18:17] "supervisor restart never clears it": the supervisor restarts a process
    that crashed or went quiet, and the original halt holds."""
    h = _halt(store, tmp_path, kind)
    sup = _supervised(store, monkeypatch)
    proc = sup.procs[NAME]
    if why == "crash":
        proc.popen.alive, proc.popen.returncode = False, 1
        sup.step()  # crashed: back-off
        proc.next_start = utcnow() - timedelta(seconds=1)
    else:
        proc.started_at = utcnow() - timedelta(hours=1)
        store._update_sleeve(NAME, heartbeat_at=utcnow() - timedelta(hours=1))
    sup.step()
    assert sup.procs[NAME].alive
    _kept(store, h, _process_start(store, h))


@pytest.mark.parametrize("action", ["stop_start", "deploy_restart", "supervisor_restart", "resume", "0000_utc_roll"])
def test_guard_nothing_but_ral_writes_a_liquidation_reset(store, tmp_path, client, monkeypatch, action):
    """GUARD (passes on 3be572a) for #164 (e14acec), which judges "liquidated" from the journal: a liquidation stays
    live until an event of kind "liquidation_reset" is written after it, and only RAL may write one [R18:17] "a halt
    is cleared only by its own action". After a liquidation, a PM Stop then Start, a deploy restart, a supervisor
    restart, a PM resume (and the restart that applies it) or the next 00:00 UTC roll writes none. (RAL writing
    exactly one: test_ral_writes_exactly_one_liquidation_reset_event_..., in test_ral_xfails.py.)"""
    from sleeve_fund import supervisor
    from sleeve_fund.paper.runtime import SleeveRuntime

    h = _halt(store, tmp_path, "liquidation")
    sup = _supervised(store, monkeypatch)
    if action == "stop_start":
        _dashboard_clock(monkeypatch, h)
        for command in ("stop", "start"):
            _post(client, {"command": command, "reason": f"{command} it"})
            sup.step()
        _process_start(store, h)
    elif action == "deploy_restart":
        for name, proc in sup.procs.items():
            sup._stop(name, proc, "supervisor shutting down")
        SleeveRuntime(store, NAME, now=lambda: h.at).on_stop()
        supervisor.Supervisor(store).step()
        _process_start(store, h)
    elif action == "supervisor_restart":
        proc = sup.procs[NAME]
        proc.popen.alive, proc.popen.returncode = False, 1
        sup.step()
        proc.next_start = utcnow() - timedelta(seconds=1)
        sup.step()
        _process_start(store, h)
    elif action == "resume":
        try:
            store.command(NAME, "resume", "carry on", actor="PM")
        except ValueError:
            pass  # refused in words: also fine
        _tick(_process_start(store, h), h.at + timedelta(minutes=1), h.equity, price=h.price)
    else:
        rt = _process_start(store, h)
        _tick(rt, ROLL + timedelta(seconds=30), h.equity, price=h.price)
        _runtime(store, ROLL + timedelta(minutes=1), h.equity, price=h.price)
    assert not _ral_events(store), [(e["kind"], e["message"]) for e in _ral_events(store)]


# --- Start refuses while halted; Stop always works [R18:17] -------------------------------------------------------

@pytest.mark.parametrize("kind", ["drawdown_halt", "daily_pause", pytest.param("liquidation", marks=xf)])
def test_start_refuses_while_halted_naming_the_clearing_action(store, tmp_path, client, monkeypatch, kind):
    """[R18:17] "Start refuses while halted naming the clearing action": stopped while halted, Start is refused in
    words that name what clears it (resume; the next 00:00 UTC; reset after liquidation), and it stays stopped."""
    h = _halt(store, tmp_path, kind)
    _dashboard_clock(monkeypatch, h)
    sup = _supervised(store, monkeypatch)
    assert _error(_post(client, {"command": "stop", "reason": "stop it"})) is None
    sup.step()
    err = _error(_post(client, {"command": "start", "reason": "start it"}))
    assert err and CLEARED_BY[kind] in err.lower(), err
    assert store.sleeve(NAME).desired_state == "stopped"
    _kept(store, h)


@pytest.mark.parametrize("kind", HALTS)
def test_guard_stop_always_works(store, tmp_path, client, monkeypatch, kind):
    """GUARD (passes on 3be572a) for [R18:17] "Stop always works": in every halt state Stop is taken, the strategy is
    marked stopped, and the supervisor stops its process."""
    _halt(store, tmp_path, kind)
    sup = _supervised(store, monkeypatch)
    assert _error(_post(client, {"command": "stop", "reason": "stop it"})) is None
    assert store.sleeve(NAME).desired_state == "stopped"
    popen = sup.procs[NAME].popen
    sup.step()
    assert not popen.alive and sup.procs[NAME].popen is None
    assert store.last_event(NAME, ("process_stop",)) is not None


# --- each halt's own clearing action [R18:17] ---------------------------------------------------------------------

@pytest.mark.parametrize("at", [T + timedelta(minutes=5), datetime(2025, 10, 3, 23, 50, tzinfo=timezone.utc)])
def test_the_daily_pause_clears_at_the_next_0000_utc_roll_and_not_before(store, at):
    """[R18:17] "daily pause: next 00:00 UTC roll": a pause at 10:05 or at 23:50 ends at 00:00 UTC on 4 Oct (not 24
    hours later): still paused 30 seconds before (a restart then keeps it), trading 30 seconds after."""
    rt = _ordinary(store, "daily_pause", t=at)
    assert store.sleeve(NAME).paused_until == ROLL, store.sleeve(NAME).paused_until
    _tick(rt, ROLL - timedelta(seconds=30), 9_400.0, price=60_000.0)
    assert not rt.can_open() and store.sleeve(NAME).status == "paused"
    rt2 = _runtime(store, ROLL - timedelta(seconds=20), 9_400.0, price=60_000.0)
    assert not rt2.can_open()
    _tick(rt2, ROLL + timedelta(seconds=30), 9_400.0, price=60_000.0)
    assert rt2.can_open() and store.sleeve(NAME).status == "running", store.sleeve(NAME).status_reason


def test_a_resume_does_not_clear_a_daily_pause(store):
    """[R18:17] "a halt is cleared only by its own action (... daily pause: next 00:00 UTC roll ...)": the PM's resume
    is refused in words or ignored; it stays paused until the roll."""
    rt = _ordinary(store, "daily_pause", t=T)
    until = store.sleeve(NAME).paused_until
    try:
        store.command(NAME, "resume", "carry on")
    except ValueError:
        pass
    _tick(rt, T + timedelta(hours=1), 9_400.0, price=60_000.0)
    s = store.sleeve(NAME)
    assert s.status == "paused" and s.paused_until == until and not rt.can_open(), (s.status, s.status_reason)


def test_guard_a_resume_clears_a_drawdown_halt(store):
    """GUARD (passes on 3be572a) for [R18:17] "DD halt: resume": the PM's resume lifts a drawdown halt."""
    rt = _ordinary(store, "drawdown_halt", t=T)
    store.command(NAME, "resume", "checked, carry on")
    _tick(rt, T + timedelta(minutes=1), 7_900.0, price=60_000.0)
    assert store.sleeve(NAME).status == "running" and rt.can_open()


# --- the falsifier: the dashboard shows each halt state's clearing action [R18:17] ----------------------------------

def _banner(page):
    for part in page.split('role="alert">')[1:]:
        text = part.split("</div>")[0]
        if "Halted:" in text or "Paused:" in text:
            return " ".join(text.split())
    return ""


@pytest.mark.parametrize("kind", ["drawdown_halt", "daily_pause", pytest.param("liquidation", marks=xf)])
def test_the_dashboard_shows_each_halt_states_clearing_action(store, tmp_path, client, kind):
    """[R18:17] Falsifier: "every halt state shows its clearing action on the dashboard". The strategy page's halt
    banner names it: "resume" for a drawdown halt (a GUARD: it passes on 3be572a); the next "00:00 UTC" for a daily
    pause, never resume; "Reset after liquidation" for a liquidation, never "until you resume"."""
    _halt(store, tmp_path, kind)
    banner = _banner(client.get(f"/sleeves/{NAME}", auth=AUTH).text)
    low = banner.lower()
    assert banner, "no halt banner"
    assert CLEARED_BY[kind] in low, banner
    wrong = "reset after liquidation" if kind == "drawdown_halt" else "until you resume"
    assert wrong not in low, banner
