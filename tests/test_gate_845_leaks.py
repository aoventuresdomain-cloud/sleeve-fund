"""QA leak repros 2 and 3 on the stop-safety + CHOKE + GAP-LIQ gate (845df4d), from the leaks QA found on main/eb737bd
(qa-state 22:33). Leaks 1 and 4 are covered by the exposure-gate set's replay cells (stopped/retired x every path;
partial/kept x stopped), run read-only alongside; the raced fill (leak 4 / P1-SG7) per the Advisor's 23:05 ruling is
in test_gate_845_u35_sg7.py. Leak 2 is a strict xfail (raises=AssertionError) where 845df4d fails (P1-SG1: Start
accepted on an archived strategy and the supervisor starts it); leak 3 is a plain regression guard (passes on 845df4d). Run with PYTHONPATH=<checkout>:<checkout>/tests (imports test_sleeve_runtime.store).

(2) Archive (Retire = Stop + Archive, Advisor 18:17) then Start must not open: Start is refused, nothing is started.
(3) A daily-loss pause is cleared only by the next 00:00 UTC roll (Advisor 18:17 HC): the PM's Resume (dashboard or
    store), a Stop/Start, a deploy restart all leave it paused; it lifts at the roll, not 24 h after it began.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.paper.runtime import SleeveRuntime
from test_sleeve_runtime import store  # noqa: F401  (Postgres when TEST_DATABASE_URL is set)

AUTH = ("pm", "qa-pw")
# A head without the gate branch (main 9ae1f8a): the leak-3 guard fails there (no 00:00 roll, no entry_blocked).
PRE_GATE = not hasattr(SleeveRuntime, "entry_blocked")


def _blocked(rt) -> bool:
    if not hasattr(rt, "entry_blocked"):
        raise AssertionError("not built: SleeveRuntime.entry_blocked (CHOKE)")
    return rt.entry_blocked()[0]
SAME = {"origin": "http://testserver"}


@pytest.fixture
def client(store, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("DASHBOARD_PASSWORD", AUTH[1])
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund.dashboard import app as app_mod

    return TestClient(app_mod.create_app(store)), store


def _cmd(c, command, reason="QA"):
    r = c.post("/sleeves/s1/command", data={"command": command, "reason": reason}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    return r.headers.get("location", "")


class FakePopen:
    pid, returncode = 4242, None
    poll = lambda self: None  # noqa: E731
    send_signal = wait = kill = lambda self, *a, **k: 0  # noqa: E731


# PE2: P1-SG1 passes on this head: its xfail mark removed
@pytest.mark.parametrize("how", ["archive_stopped", "retire_running"])
def test_leak2_archive_then_start_is_refused_and_nothing_is_started(client, monkeypatch, how):
    from sleeve_fund import supervisor as sup

    c, store = client
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced",
                        desired_state="running" if how == "retire_running" else "stopped")
    if how == "retire_running":
        assert "command_error" not in _cmd(c, "stop", "retire it")
    r = c.post("/sleeves/s1/archive", data={"action": "archive", "reason": "retired"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303, r.text
    assert "s1" in store.archived()
    loc = _cmd(c, "start", "start the archived one")
    started = []
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: started.append(1) or FakePopen())
    sup.Supervisor(store).step()
    got = {"start_refused": "command_error" in loc, "desired": store.sleeve("s1").desired_state,
           "processes_started": len(started)}
    assert got == {"start_refused": True, "desired": "stopped", "processes_started": 0}, (got, loc)


D = datetime(2026, 10, 6, 0, 30, tzinfo=timezone.utc)


def _paused(store, t):
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced")
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    mark = {"cash": 0.0, "qty": 1.0}
    assert rt.tick(equity=10_000, price=10_000, **mark) is None
    t[0] = D.replace(hour=9)
    assert rt.tick(equity=9_450, price=9_450, **mark) == "flatten"  # down 5.5%: paused for the day
    return rt


@pytest.mark.xfail(PRE_GATE, strict=True, raises=AssertionError, reason="HC (Advisor 18:17) + CHOKE not on this head")
def test_leak3_a_daily_pause_ends_at_the_next_0000_roll_and_no_resume_stop_start_or_restart_lifts_it(client,
                                                                                                    monkeypatch):
    from sleeve_fund import store as store_mod
    from sleeve_fund.dashboard import app as app_mod

    c, store = client
    t = [D]
    monkeypatch.setattr(app_mod, "utcnow", lambda: t[0])
    monkeypatch.setattr(store_mod, "utcnow", lambda: t[0])
    rt = _paused(store, t)
    roll = datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc)
    flat = {"cash": 9_450.0, "qty": 0.0}
    seen = {"paused_until": store.sleeve("s1").paused_until}
    t[0] = D.replace(hour=9, minute=5)
    loc = _cmd(c, "resume", "carry on")
    seen["dashboard_resume_refused"] = "command_error" in loc and "00%3A00" in loc
    store.command("s1", "resume", "carry on (direct)")  # a resume that got past the dashboard
    rt.tick(equity=9_450, price=9_450, **flat)
    seen["after_store_resume"] = (store.sleeve("s1").status, rt.can_open())
    store.set_desired_state("s1", "running")
    loc = _cmd(c, "stop", "stop it")
    loc2 = _cmd(c, "start", "start it again")
    seen["start_refused"] = "command_error" in loc2 and "00%3A00" in loc2
    seen["status_after_stop_start"] = store.sleeve("s1").status
    store.set_desired_state("s1", "running")
    t[0] = roll - timedelta(seconds=30)
    rt2 = SleeveRuntime(store, "s1", now=lambda: t[0])  # a deploy/supervisor restart just before the roll
    rt2.on_start(0.008)
    rt2.tick(equity=9_450, price=9_450, **flat)
    seen["restart_before_roll"] = (store.sleeve("s1").status, rt2.can_open(), _blocked(rt2))
    t[0] = roll + timedelta(seconds=30)
    rt2.tick(equity=9_450, price=9_450, **flat)
    opens = rt2.can_open()  # the runtime lifts an expired pause on its next entry check (can_open / entry_blocked)
    seen["after_roll"] = (store.sleeve("s1").status, opens, _blocked(rt2))
    want = {"paused_until": roll, "dashboard_resume_refused": True, "after_store_resume": ("paused", False),
            "start_refused": True, "status_after_stop_start": "paused",
            "restart_before_roll": ("paused", False, True), "after_roll": ("running", True, False)}
    assert seen == want, seen
