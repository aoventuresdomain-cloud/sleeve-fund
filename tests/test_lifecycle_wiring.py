"""P2-6 wiring: the PM's lifecycle (sleeve_fund.lifecycle) through the store's one writer, the runtime's entry gate, the
wind-down stop, the 75% notice, the deadline's flatten and retire once flat, the supervisor and the PM-only route.
Design: pe1-plans/p2-6-wiring.md (HoE OK 13 Oct). Rulings: Independent Quant Advisor 6 Oct 14:50/14:55."""

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import lifecycle
from sleeve_fund.paper.runtime import (
    WIND_FLATTEN,
    WIND_NOTICE,
    WIND_RETRY,
    WIND_STOP,
    SleeveRuntime,
    entry_blocked,
)
from sleeve_fund.store import Store
from sleeve_fund.supervisor import Proc, decide

T0 = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


def _sleeve(store, name="s1", bar_spec="1-HOUR-LAST-EXTERNAL"):
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USD", bar_spec=bar_spec,
                        starting_balance=10_000)


def _hold(store, name="s1", qty=0.05, at=T0 - timedelta(days=2), n=1):
    store.record_fill(name, side="BUY", qty=qty, price=60_000, fee=1.0, order_id=f"o{n}", trade_id=f"t{n}", ts=at)


def _close(store, name="s1", qty=0.05, at=T0, n=2):
    store.record_fill(name, side="SELL", qty=qty, price=61_000, fee=1.0, order_id=f"o{n}", trade_id=f"t{n}", ts=at)


def _runtime(store, t, name="s1"):
    rt = SleeveRuntime(store, name, now=lambda: t[0])
    rt.on_start(0.008)
    return rt


def _kinds(store, name="s1"):
    return [e["kind"] for e in store.events(name, limit=1000)]


# --- the one writer --------------------------------------------------------------------------------------------


def test_a_wind_down_sets_its_start_and_deadline_and_journals_the_move_and_the_holds(store):
    _sleeve(store)
    s = store.set_lifecycle("s1", "wind_down", "pm", "rotating out", now=T0)
    assert (s.lifecycle, s.wind_down_started, s.wind_down_deadline) == ("winding_down", T0, T0 + timedelta(days=3))
    ev = store.last_event("s1", ("lifecycle",))
    assert "must be flat by 16 Oct 2026 12:00 UTC (13:00 UK)" in ev["message"]
    assert "Window 3 days" in ev["message"] and "0 closed paper trades" in ev["message"]
    d = store.decisions("s1", limit=1)[0]
    assert (d["actor"], d["action"], d["reason"]) == ("PM", "wind_down", "rotating out")


def test_the_deadline_comes_from_the_strategys_own_paper_holds(store):
    _sleeve(store)
    _hold(store, at=T0 - timedelta(days=9))
    _close(store, at=T0 - timedelta(days=5))  # one 4-day hold: under 20 backtest trades, so 2 x the longest
    s = store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    assert s.wind_down_deadline == T0 + timedelta(days=8)
    assert "1 closed paper trade" in store.last_event("s1", ("lifecycle",))["message"]


@pytest.mark.parametrize("actor", ["system", "supervisor", "PM", "advisor"])
def test_only_the_pm_moves_a_strategy_and_nothing_is_written_otherwise(store, actor):
    _sleeve(store)
    with pytest.raises(lifecycle.NotPM):
        store.set_lifecycle("s1", "wind_down", actor, "r", now=T0)
    assert store.sleeve("s1").lifecycle == "active" and "lifecycle" not in _kinds(store)


def test_retire_while_holding_is_refused_and_points_to_close_now(store):
    _sleeve(store)
    _hold(store)
    with pytest.raises(lifecycle.LifecycleError, match="close it now"):
        store.set_lifecycle("s1", "retire", "pm", "r", now=T0)
    assert store.sleeve("s1").lifecycle == "active"


def test_retiring_a_flat_strategy_stops_and_archives_it_and_nothing_brings_it_back(store):
    _sleeve(store)
    s = store.set_lifecycle("s1", "retire", "pm", "done with it", now=T0)
    assert (s.lifecycle, s.desired_state) == ("retired", "stopped") and "s1" in store.archived()
    with pytest.raises(ValueError, match="can't be brought back"):
        store.unarchive("s1")
    for action in ("wind_down", "retire", "close_now"):
        with pytest.raises(lifecycle.LifecycleError, match="retired"):
            store.set_lifecycle("s1", action, "pm", "r", now=T0)


def test_close_now_on_a_holder_winds_it_down_with_its_deadline_now(store):
    _sleeve(store)
    _hold(store)
    s = store.set_lifecycle("s1", "close_now", "pm", "get out", now=T0)
    assert (s.lifecycle, s.wind_down_started, s.wind_down_deadline) == ("winding_down", T0, T0)
    assert "flattened at market on the first fresh price" in store.last_event("s1", ("lifecycle",))["message"]


def test_close_now_while_winding_down_keeps_the_start_and_moves_the_deadline_to_now(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    s = store.set_lifecycle("s1", "close_now", "pm", "now", now=T0 + timedelta(hours=5))
    assert (s.wind_down_started, s.wind_down_deadline) == (T0, T0 + timedelta(hours=5))


def test_close_now_on_a_flat_strategy_retires_it_at_once(store):
    _sleeve(store)
    assert store.set_lifecycle("s1", "close_now", "pm", "r", now=T0).lifecycle == "retired"


def test_a_saved_backtest_and_an_unknown_action_are_refused(store):
    _sleeve(store)
    with pytest.raises(ValueError, match="bad lifecycle action"):
        store.set_lifecycle("s1", "pause", "pm", "r")
    with pytest.raises(ValueError, match="saved backtest"):
        store.set_lifecycle("bt:abc", "wind_down", "pm", "r")


def test_complete_retire_needs_winding_down_and_flat(store):
    _sleeve(store)
    assert not store.complete_retire("s1", T0)  # active
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    assert not store.complete_retire("s1", T0)  # still holding
    _close(store, at=T0 + timedelta(hours=1))
    assert store.complete_retire("s1", T0 + timedelta(hours=2), "flat")
    s = store.sleeve("s1")
    assert (s.lifecycle, s.desired_state, s.wind_down_deadline) == ("retired", "stopped", T0 + timedelta(days=3))
    assert store.last_event("s1", ("lifecycle",))["message"].startswith("Retired (flat)")
    assert not store.complete_retire("s1", T0 + timedelta(hours=3))  # once


def test_set_lifecycle_has_one_caller_the_pm_route():
    root = Path(__file__).resolve().parents[1] / "sleeve_fund"
    callers = [str(p.relative_to(root)) for p in root.rglob("*.py") if ".set_lifecycle(" in p.read_text()]
    assert callers == ["dashboard/app.py"]


# --- the database's own checks (0017) -------------------------------------------------------------------------


def test_the_database_refuses_an_unknown_lifecycle_and_a_wind_down_without_its_times(store):
    _sleeve(store)
    for sql in ("UPDATE sleeves SET lifecycle = 'paused' WHERE name = 's1'",
                "UPDATE sleeves SET lifecycle = 'winding_down' WHERE name = 's1'",
                ("UPDATE sleeves SET lifecycle = 'winding_down', wind_down_started = :t, "
                 "wind_down_deadline = :before WHERE name = 's1'")):
        with pytest.raises(IntegrityError), store.engine.begin() as c:
            c.execute(text(sql), {"t": T0, "before": T0 - timedelta(seconds=1)})
    assert store.sleeve("s1").lifecycle == "active"


# --- the entry gate --------------------------------------------------------------------------------------------


def test_a_winding_down_strategy_opens_nothing_and_start_or_resume_does_not_clear_it(store):
    _sleeve(store)
    t = [T0]
    rt = _runtime(store, t)
    assert rt.can_open()
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    rt.tick(equity=10_000, cash=10_000, qty=0, price=60_000)
    assert not rt.can_open()
    blocked, why = rt.entry_blocked()
    assert blocked and why.code == "winding_down" and why.startswith("Winding down: ")
    assert entry_blocked(store, "s1", T0, starting=True)[1].code == "winding_down"


def test_a_retired_strategy_reads_retired_for_good_not_the_archives_restore(store):
    _sleeve(store)
    store.set_lifecycle("s1", "retire", "pm", "r", now=T0)
    blocked, why = entry_blocked(store, "s1", T0, starting=True)
    assert blocked and why.code == "retired" and "never trades again" in why and "Restore" not in why


# --- the notice, the deadline and retire -----------------------------------------------------------------------


def test_the_notice_comes_once_at_75_percent_and_a_restart_does_not_repeat_it(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)  # 3-day window: 75% at T0 + 54 h
    t = [T0 + timedelta(hours=53)]
    rt = _runtime(store, t)
    assert rt.lifecycle_due(t[0], flat=False, busy=False) is None and WIND_NOTICE not in _kinds(store)
    t[0] = T0 + timedelta(hours=54)
    assert rt.lifecycle_due(t[0], flat=False, busy=False) is None
    assert store.last_event("s1", (WIND_NOTICE,))["level"] == "warning"
    rt.lifecycle_due(t[0] + timedelta(hours=1), flat=False, busy=False)
    rt = _runtime(store, t)
    rt.lifecycle_due(t[0] + timedelta(hours=2), flat=False, busy=False)
    assert _kinds(store).count(WIND_NOTICE) == 1


def test_at_the_deadline_it_flattens_through_the_exit_path_each_bar_until_flat_and_alerts_the_retry(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    deadline = T0 + timedelta(days=3)
    t = [deadline - timedelta(hours=1)]
    rt = _runtime(store, t)
    assert rt.lifecycle_due(t[0], flat=False, busy=False) is None  # not yet (the notice went)
    t[0] = deadline
    intent, reason = rt.lifecycle_due(t[0], flat=False, busy=False)
    assert intent == WIND_FLATTEN == "wind_down" and "deadline 16 Oct 2026 12:00 UTC" in reason
    assert rt.lifecycle_due(t[0] + timedelta(hours=1), flat=False, busy=True) is None  # its order still working
    assert WIND_RETRY not in _kinds(store)
    assert rt.lifecycle_due(t[0] + timedelta(hours=2), flat=False, busy=False)[0] == "wind_down"  # rejected: again
    assert store.last_event("s1", (WIND_RETRY,))["level"] == "error"
    rt.lifecycle_due(t[0] + timedelta(hours=3), flat=False, busy=False)
    assert _kinds(store).count(WIND_RETRY) == 1  # one alert for the episode, a decision row per attempt
    assert len(store.decisions("s1", action="flatten")) == 3


def test_retired_only_once_flat_and_journaled_by_how(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    t = [T0 + timedelta(days=3)]
    rt = _runtime(store, t)
    assert rt.lifecycle_due(t[0], flat=False, busy=False)[0] == "wind_down"
    assert rt.lifecycle_due(t[0], flat=True, busy=False) is None  # the engine says flat; the journal doesn't yet
    assert store.sleeve("s1").lifecycle == "winding_down"
    _close(store, at=t[0])
    rt.lifecycle_due(t[0] + timedelta(hours=1), flat=True, busy=False)
    s = store.sleeve("s1")
    assert (s.lifecycle, s.status, rt.lifecycle) == ("retired", "stopped", "retired")
    assert store.last_event("s1", ("lifecycle",))["message"].startswith("Retired at the deadline")


def test_a_wind_down_that_goes_flat_on_its_own_is_retired_flat(store):
    _sleeve(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    t = [T0 + timedelta(hours=1)]
    _runtime(store, t).lifecycle_due(t[0], flat=True, busy=False)
    assert store.last_event("s1", ("lifecycle",))["message"].startswith("Retired (flat)")


def test_close_now_is_flattened_at_the_next_bar_close_and_retired_as_close_now(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "close_now", "pm", "out", now=T0)
    t = [T0 + timedelta(hours=1)]
    rt = _runtime(store, t)
    assert rt.lifecycle_due(t[0], flat=False, busy=False)[0] == "wind_down"
    _close(store, at=t[0])
    rt.lifecycle_due(t[0] + timedelta(hours=1), flat=True, busy=False)
    assert store.last_event("s1", ("lifecycle",))["message"].startswith("Retired by close now")


def test_a_stale_feed_past_the_deadline_is_flattened_by_the_tick(store):
    """No bar close for over a bar after the deadline: the tick sends the flatten the bar close would have (stale
    data too, Advisor 14:55), and again a bar later while it still holds."""
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    held = {"equity": 10_000, "cash": 7_000, "qty": 0.05, "price": 60_000}
    t = [T0 + timedelta(days=3, minutes=30)]
    rt = _runtime(store, t)
    rt.lifecycle_due(T0 + timedelta(days=3) - timedelta(minutes=30), flat=False, busy=False)  # the last bar close
    assert rt.tick(**held) is None  # within a bar (and grace) of the last bar close
    t[0] = T0 + timedelta(days=3, hours=1, minutes=35)
    assert rt.tick(**held) == "flatten" and rt.flatten_why[0] == "wind_down"
    t[0] += timedelta(minutes=1)
    assert rt.tick(**held) is None  # a bar later, not every tick
    t[0] += timedelta(hours=1, minutes=5)
    assert rt.tick(**held) == "flatten"


HELD = {"equity": 10_000, "cash": 7_000, "qty": 0.05, "price": 60_000}


@pytest.mark.parametrize("winding", [False, True], ids=["from-active", "from-winding-down"])
def test_p26_2_close_now_flattens_at_market_on_the_first_fresh_tick_from_either_state(store, winding):
    """P26-2 (Advisor 13 Oct 20:30 UK, MUST): the first fresh tick after the PM's close now sends the flatten, not
    the next bar close; from winding down the deadline only moves, so this process must see that move itself."""
    _sleeve(store)
    _hold(store)
    if winding:
        store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0 - timedelta(days=1))
    t = [T0 - timedelta(minutes=30)]
    rt = _runtime(store, t)
    rt.holds.pop("stale_data", None)  # prices are arriving
    rt.lifecycle_due(t[0], flat=False, busy=False)  # the last bar close before the PM's move
    assert rt.tick(**HELD) is None
    store.set_lifecycle("s1", "close_now", "pm", "out", now=T0)
    t[0] = T0 + timedelta(seconds=5)
    assert rt.tick(**HELD) == "flatten" and rt.flatten_why == (WIND_FLATTEN, rt.flatten_why[1])
    assert "Close now by the PM" in rt.flatten_why[1]
    t[0] += timedelta(seconds=5)
    assert rt.tick(**HELD) is None  # once: a bar close resends it while it still holds (the deadline's retry)


def test_p26_2_on_stale_data_close_now_waits_for_a_fresh_tick_and_says_so_once_after_a_bar(store):
    _sleeve(store)
    _hold(store)
    t = [T0 - timedelta(minutes=30)]
    rt = _runtime(store, t)
    rt.holds["stale_data"] = "last price 600 s old"
    store.set_lifecycle("s1", "close_now", "pm", "out", now=T0)
    for minutes in (1, 30, 61):  # within a bar and its grace: waits, quietly
        t[0] = T0 + timedelta(minutes=minutes)
        assert rt.tick(**HELD) is None
    assert "close_now_waiting" not in _kinds(store)
    for minutes in (63, 64):
        t[0] = T0 + timedelta(minutes=minutes)
        assert rt.tick(**HELD) is None
    assert _kinds(store).count("close_now_waiting") == 1
    rt.holds.pop("stale_data")  # a fresh price
    t[0] += timedelta(seconds=5)
    assert rt.tick(**HELD) == "flatten" and "Close now by the PM" in rt.flatten_why[1]


def _dust(store):
    """Held 0.05, sold 0.04999999: 1e-8 left, worth 0.0006, below every venue's smallest order."""
    _hold(store)
    _close(store, qty=0.04999999, at=T0 - timedelta(hours=1))


def test_p26_1_dust_counts_as_flat_for_the_pms_retire_and_close_now(store):
    """P26-1 (HoQA MUST FIX): the engine reads below its smallest order as flat, and so does the lifecycle (is_dust,
    as a reset): dust no order can close never blocks a retire."""
    _sleeve(store)
    _dust(store)
    assert store.set_lifecycle("s1", "retire", "pm", "done", now=T0).lifecycle == "retired"
    _sleeve(store, "s2")
    store.record_fill("s2", side="BUY", qty=0.05, price=60_000, fee=1.0, order_id="o1", trade_id="t1",
                      ts=T0 - timedelta(days=2))
    store.record_fill("s2", side="SELL", qty=0.04999999, price=61_000, fee=1.0, order_id="o2", trade_id="t2",
                      ts=T0 - timedelta(hours=1))
    assert store.set_lifecycle("s2", "close_now", "pm", "out", now=T0).lifecycle == "retired"


def test_p26_1_a_wind_down_whose_flatten_leaves_dust_is_retired_once_and_worth_an_order_is_never_dust(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    _close(store, qty=0.04999999, at=T0 + timedelta(days=3))  # the deadline's flatten, rounded down to the lot
    t = [T0 + timedelta(days=3, hours=1)]
    rt = _runtime(store, t)
    rt.lifecycle_due(t[0], flat=True, busy=False)
    assert store.sleeve("s1").lifecycle == "retired"
    assert store.complete_retire("s1", t[0]) is False  # once
    _sleeve(store, "s2")
    store.record_fill("s2", side="BUY", qty=0.0001, price=60_000, fee=1.0, order_id="o1", trade_id="t1",
                      ts=T0 - timedelta(days=2))  # 6.00: above the dust line, still a holding
    with pytest.raises(lifecycle.LifecycleError):
        store.set_lifecycle("s2", "retire", "pm", "r", now=T0)
    store.set_lifecycle("s2", "wind_down", "pm", "r", now=T0)
    assert store.complete_retire("s2", T0 + timedelta(days=4)) is False


def test_f26_a_a_pm_move_never_writes_over_a_retire_the_runtime_completed_meanwhile(store, monkeypatch):
    """CR F26-A: close now judged on winding down, while the runtime's complete_retire lands first: refused, and the
    strategy stays retired and archived."""
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    _close(store, at=T0 + timedelta(hours=1))
    read, raced = store.sleeve, []

    def racing(name):  # the PM's read sees winding down; the runtime retires it before the write
        row = read(name)
        if row.lifecycle == "winding_down" and not raced:
            raced.append(True)
            assert store.complete_retire(name, T0 + timedelta(hours=2))
        return row

    monkeypatch.setattr(store, "sleeve", racing)
    with pytest.raises(lifecycle.LifecycleError, match="changed meanwhile"):
        store.set_lifecycle("s1", "close_now", "pm", "out", now=T0 + timedelta(hours=2))
    monkeypatch.undo()
    assert store.sleeve("s1").lifecycle == "retired" and "s1" in store.archived()


def test_a_backtest_has_no_pm_so_its_lifecycle_never_acts():
    rt = SleeveRuntime.for_backtest(strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                    starting_balance=10_000, risk_profile="balanced")
    assert rt.lifecycle == "active" and rt.lifecycle_due(T0, flat=False, busy=False) is None
    assert not rt.wind_down_overdue(T0)


# --- the wind-down stop ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("side,price,atr,stop_atr,liq,level", [
    (1, 60_000.0, 1_000.0, None, None, 57_000.0),  # 3 x Wilder ATR below
    (-1, 60_000.0, 1_000.0, 2.0, None, 62_000.0),  # its own stop_atr above, for a short
    (1, 60_000.0, 1_000.0, None, 56_000.0, 58_000.0),  # held at half way to liquidation
])
def test_the_wind_down_stop_is_journaled_once_and_a_restart_reads_it_back_unchanged(store, side, price, atr,
                                                                                    stop_atr, liq, level):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    t = [T0 + timedelta(hours=1)]
    rt = _runtime(store, t)
    assert rt.wind_stop is None
    assert rt.place_wind_stop(side, price, atr, stop_atr=stop_atr, liquidation=liq) == level
    ev = store.last_event("s1", (WIND_STOP,))
    assert ev["message"].startswith(f"Wind-down stop at {level!r} for the {'long' if side > 0 else 'short'}")
    assert ("half the distance to liquidation" in ev["message"]) == (liq is not None)
    assert _runtime(store, t).wind_stop == (side, level)  # read back, not recomputed
    assert rt.wind_stop_hit(level) and not rt.wind_stop_hit(level + side * 1.0)


def test_a_new_wind_down_never_reads_an_earlier_wind_downs_stop(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    t = [T0 + timedelta(hours=1)]
    _runtime(store, t).place_wind_stop(1, 60_000.0, 1_000.0, stop_atr=None, liquidation=None)
    with store.engine.begin() as c:  # as if a later wind-down (a fresh start) on the same row
        c.execute(text("UPDATE sleeves SET wind_down_started = :s, wind_down_deadline = :d WHERE name = 's1'"),
                  {"s": T0 + timedelta(hours=2), "d": T0 + timedelta(days=4)})
    assert _runtime(store, [T0 + timedelta(hours=3)]).wind_stop is None


# --- the supervisor --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("desired,holds", [("running", False), ("running", True), ("stopped", True)])
def test_the_supervisor_never_starts_a_retired_strategy(store, desired, holds):
    _sleeve(store)
    store.set_lifecycle("s1", "retire", "pm", "r", now=T0)
    s = store.sleeve("s1")
    s.desired_state = desired
    assert decide(s, Proc(), T0, holds=holds) != "start"


def test_the_supervisor_keeps_a_stopped_winding_down_holder_running_for_its_exits(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    store.set_desired_state("s1", "stopped")
    assert decide(store.sleeve("s1"), Proc(), T0, holds=True) == "start"


# --- the PM-only route -----------------------------------------------------------------------------------------


@pytest.fixture
def client(store, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    return TestClient(dash.create_app(store)), dash


def test_the_route_moves_a_strategy_for_the_pm_and_needs_a_reason_and_the_login(store, client):
    c, _ = client
    _sleeve(store)
    hdr = {"Origin": "http://testserver"}
    r = c.post("/sleeves/s1/lifecycle", data={"action": "wind_down", "reason": ""}, auth=("pm", "pw"), headers=hdr,
               follow_redirects=False)
    assert r.status_code == 400 and store.sleeve("s1").lifecycle == "active"
    r = c.post("/sleeves/s1/lifecycle", data={"action": "wind_down", "reason": "rotate"}, auth=("pm", "wrong"),
               headers=hdr, follow_redirects=False)
    assert r.status_code == 401 and store.sleeve("s1").lifecycle == "active"
    r = c.post("/sleeves/s1/lifecycle", data={"action": "wind_down", "reason": "rotate"}, auth=("pm", "pw"),
               headers=hdr, follow_redirects=False)
    assert r.status_code == 303 and store.sleeve("s1").lifecycle == "winding_down"
    _hold(store)
    r = c.post("/sleeves/s1/lifecycle", data={"action": "retire", "reason": "x"}, auth=("pm", "pw"), headers=hdr,
               follow_redirects=False)
    assert r.status_code == 400 and re.search("close it now", r.text)


# --- the engine, end to end (a restart carries the position into the wind-down) ---------------------------------


def test_the_engine_places_the_stop_flattens_at_the_deadline_through_the_exit_path_and_retires_once_flat(store,
                                                                                                         instrument):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.runner import run_backtest

    _sleeve(store, bar_spec="1-DAY-LAST-EXTERNAL")
    prices = synthetic_ohlcv(days=40, seed=3)
    run_backtest("buy_and_hold", prices.iloc[:20], instrument, runtime=SleeveRuntime(store, "s1", tick_seconds=21_600))
    assert store.held_qty("s1") > 0
    start = prices.index[19].to_pydatetime()
    store.set_lifecycle("s1", "wind_down", "pm", "rotating out", now=start)  # 3-day floor: due at day 22
    run_backtest("buy_and_hold", prices.iloc[20:], instrument, runtime=SleeveRuntime(store, "s1", tick_seconds=21_600))
    stop = store.last_event("s1", (WIND_STOP,))
    assert stop is not None and stop["ts"] < start + timedelta(days=3)
    sells = [o for o in store.orders("s1", limit=100) if o["side"] == "SELL"]
    assert [o["intent"] for o in sells] == ["wind_down"] and "Wind-down deadline" in sells[0]["reason"]
    assert sells[0]["ts"] >= start + timedelta(days=3)
    assert store.held_qty("s1") == 0
    s = store.sleeve("s1")
    assert (s.lifecycle, s.desired_state) == ("retired", "stopped") and "s1" in store.archived()
    assert store.last_event("s1", ("lifecycle",))["message"].startswith("Retired at the deadline")
    assert [f["side"] for f in reversed(store.fills("s1"))] == ["BUY", "SELL"]  # nothing opened while winding down


# --- Advisor 13 Oct 16:40 ---------------------------------------------------------------------------------------


def test_retire_once_flat_names_the_pm_and_the_system_in_its_decision(store):
    _sleeve(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    store.complete_retire("s1", T0 + timedelta(hours=1))
    d = store.decisions("s1", limit=1)[0]
    assert d["actor"] == "PM" and d["reason"].startswith("Triggered by the system once flat, ending the PM's wind-down")


def _engine(stop_atr=None, own_ready=False, bars=0, daily=0.02, wind=None):
    """Just what the wind-down ATR and the replay's stop read of the engine."""
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    store = SimpleNamespace(sleeve=lambda n: SimpleNamespace(venue=None, instrument="BTC/USD"))
    rt = SimpleNamespace(store=store, name="s1", now=lambda: T0, wind_stop=wind)
    eng = SimpleNamespace(_cfg=SimpleNamespace(stop_atr=stop_atr, atr_bars=20), runtime=rt,
                          _atr=SimpleNamespace(initialized=own_ready, value=900.0),
                          _wind_bars=[(101.0 + i % 3, 99.0 - i % 2, 100.0) for i in range(bars)])
    eng._wind_atr = LongFlatStrategy._wind_atr.__get__(eng)
    eng._replayed_stop = LongFlatStrategy._replayed_stop.__get__(eng)
    return eng


def test_the_wind_down_atr_is_its_own_then_wilder_on_its_candles_then_the_daily_one_said_so(monkeypatch):
    from sleeve_fund import open_risk

    monkeypatch.setattr(open_risk, "history_atr_pct", lambda *a, **k: 0.02)
    assert _engine(stop_atr=2.0, own_ready=True)._wind_atr(60_000.0) == (900.0, 2.0, "")
    atr, own, note = _engine(stop_atr=2.0, own_ready=False, bars=20)._wind_atr(60_000.0)
    assert own is None and 2.0 < atr < 4.0 and "own ATR unavailable" in note  # 3 x Wilder on its candles
    atr, own, note = _engine(bars=20)._wind_atr(60_000.0)
    assert own is None and note == ""
    atr, own, note = _engine(bars=5)._wind_atr(60_000.0)
    assert (atr, own) == (1_200.0, None) and "5 of 14 candles" in note and "own ATR unavailable" in note
    monkeypatch.setattr(open_risk, "history_atr_pct", lambda *a, **k: None)
    with pytest.raises(ValueError, match="no ATR yet"):
        _engine(bars=5)._wind_atr(60_000.0)


@pytest.mark.parametrize("side,own,wind,want", [
    (1, None, (1, 57_000.0), 57_000.0),
    (1, 58_000.0, (1, 57_000.0), 58_000.0),  # a long's higher stop is reached first
    (1, 56_000.0, (1, 57_000.0), 57_000.0),
    (-1, 62_000.0, (-1, 63_000.0), 62_000.0),  # a short's lower stop is reached first
    (1, 58_000.0, None, 58_000.0),
    (1, 58_000.0, (-1, 63_000.0), 58_000.0),  # a stop for the other side is not this position's
])
def test_the_outage_replay_rests_the_wind_down_stop_as_the_strategys_own(side, own, wind, want):
    assert _engine(wind=wind)._replayed_stop(side, own) == want


@pytest.mark.xfail(strict=True, reason="CASH-4 (DA): a true-up arriving after retire must still book (Advisor MUST)")
def test_l15_a_true_up_arriving_after_retire_is_still_booked(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "close_now", "pm", "r", now=T0)
    _close(store, at=T0 + timedelta(hours=1))
    store.complete_retire("s1", T0 + timedelta(hours=2), "close_now")
    store.record_true_up("s1", ts=T0 - timedelta(hours=4), amount=-1.25, arrived=T0 + timedelta(hours=3))
    assert any(r["kind"] == "true_up" for r in store.funding("s1"))


@pytest.mark.parametrize("side,price,liq,level", [
    (1, 60_000.0, None, 30_000.0),  # spot or a 1x long: half way to zero
    (1, 60_000.0, 40_000.0, 50_000.0),  # half the remaining distance to liquidation
    (-1, 60_000.0, 90_000.0, 75_000.0),
])
def test_with_no_atr_an_interim_safety_stop_goes_on_at_once_and_a_tighter_atr_stop_replaces_it_once(
        store, side, price, liq, level):
    """Advisor 13 Oct 17:05, MUST: no ATR means the safety stop (S3 to S5) is placed on the first close; the ATR stop
    replaces it later only if tighter, and from then the level is fixed."""
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    t = [T0 + timedelta(hours=1)]
    rt = _runtime(store, t)
    assert rt.place_interim_wind_stop(side, price, liq) == level
    assert rt.wind_stop == (side, level) and rt.wind_stop_interim
    again = _runtime(store, t)
    assert again.wind_stop == (side, level) and again.wind_stop_interim  # a restart keeps it, still interim
    tighter = again.place_wind_stop(side, price, 1_000.0, stop_atr=None, liquidation=liq)
    assert side * (tighter - level) > 0 and not again.wind_stop_interim
    assert _runtime(store, t).wind_stop == (side, tighter) and not _runtime(store, t).wind_stop_interim


def test_an_interim_stop_tighter_than_the_atr_one_is_kept_and_fixed(store):
    _sleeve(store)
    _hold(store)
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=T0)
    t = [T0 + timedelta(hours=1)]
    rt = _runtime(store, t)
    rt.place_interim_wind_stop(1, 60_000.0, 50_000.0)  # 55,000
    rt.keep_wind_stop("the ATR stop at 45,000 would be further from the price")
    back = _runtime(store, t)
    assert back.wind_stop == (1, 55_000.0) and not back.wind_stop_interim


def test_the_engine_places_the_interim_stop_on_the_first_close_when_there_is_no_atr(store, instrument, monkeypatch):
    from sleeve_fund import open_risk
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.runner import run_backtest

    monkeypatch.setattr(open_risk, "history_atr_pct", lambda *a, **k: None)
    _sleeve(store, bar_spec="1-DAY-LAST-EXTERNAL")
    prices = synthetic_ohlcv(days=40, seed=3)
    run_backtest("buy_and_hold", prices.iloc[:20], instrument, runtime=SleeveRuntime(store, "s1", tick_seconds=21_600))
    start = prices.index[19].to_pydatetime()
    store.set_lifecycle("s1", "wind_down", "pm", "r", now=start)
    run_backtest("buy_and_hold", prices.iloc[20:], instrument, runtime=SleeveRuntime(store, "s1", tick_seconds=21_600))
    stops = [e for e in reversed(store.events("s1", limit=1000)) if e["kind"] == WIND_STOP]
    assert stops and "interim safety stop" in stops[0]["message"] and stops[0]["ts"] == prices.index[20]
    assert _kinds(store).count("wind_down_stop_missing") == 1


# --- the migration (DA 13 Oct: 0017, named CHECKs, the batch rebuild keeps every child row) -----------------------


@pytest.fixture
def engine(tmp_path):
    from sleeve_fund.store import make_engine, metadata

    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/m.db")

    def clean():
        with eng.begin() as c:
            c.execute(text("DROP TABLE IF EXISTS alembic_version"))
        metadata.drop_all(eng)

    clean()
    yield eng
    clean()


def _sleeves_shape(eng):
    from sqlalchemy import inspect

    i = inspect(eng)
    cols = {c["name"]: (str(c["type"]).split("(")[0], c["nullable"]) for c in i.get_columns("sleeves")}
    return cols, {c["name"] for c in i.get_check_constraints("sleeves")}


def test_0017_gives_sleeves_exactly_what_create_all_does(engine, tmp_path):
    from sleeve_fund import schema
    from sleeve_fund.store import make_engine

    schema.migrate(engine, log=lambda _: None)
    migrated = _sleeves_shape(engine)
    assert {"sleeves_lifecycle", "sleeves_winding_down"} <= migrated[1]
    if engine.dialect.name == "sqlite":
        fresh = make_engine(f"sqlite:///{tmp_path}/fresh.db")
    else:
        from sleeve_fund.store import metadata

        with engine.begin() as c:
            c.execute(text("DROP TABLE IF EXISTS alembic_version"))
        metadata.drop_all(engine)
        fresh = engine
    Store(engine=fresh)
    assert _sleeves_shape(fresh) == migrated


def test_0017_upgrades_an_existing_book_keeping_every_child_row_and_reads_it_active(engine):
    from alembic import command

    from sleeve_fund import schema

    with engine.connect() as conn, conn.begin():
        command.upgrade(schema._config(conn), "0013")
    now = T0.isoformat()
    rows = {
        "sleeves": "INSERT INTO sleeves (name, strategy, instrument, bar_spec, params, starting_balance, risk_profile, "
                   "warmup_bars, desired_state, status, status_reason, created_at, updated_at) VALUES ('s1', "
                   "'buy_and_hold', 'BTC/USD', '1-HOUR-LAST-EXTERNAL', '{}', 10000, 'balanced', 0, 'running', "
                   "'running', '', :t, :t)",
        "fills": "INSERT INTO fills (sleeve, ts, side, qty, price, fee, order_id, trade_id) VALUES "
                 "('s1', :t, 'BUY', 0.05, 60000, 1, 'o1', 't1')",
        "equity": "INSERT INTO equity (sleeve, ts, equity, cash, qty, price, benchmark) VALUES "
                  "('s1', :t, 10000, 7000, 0.05, 60000, 10000)",
        "orders": "INSERT INTO orders (sleeve, order_id, ts, updated_at, side, order_type, qty, status, filled_qty, fee, "
                  "intent, reason, signal, message) VALUES ('s1', 'o1', :t, :t, 'BUY', 'MARKET', 0.05, 'filled', "
                  "0.05, 1, 'entry', 'r', '{}', '')",
        "sleeve_archive": "INSERT INTO sleeve_archive (sleeve, archived_at) VALUES ('s1', :t)",
        "events": "INSERT INTO events (sleeve, ts, level, kind, message) VALUES ('s1', :t, 'info', 'start', 'm')",
    }
    with engine.begin() as c:
        for sql in rows.values():
            c.execute(text(sql), {"t": now})
    schema.migrate(engine, log=lambda _: None)
    with engine.connect() as c:
        assert {t: c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() for t in rows} == {t: 1 for t in rows}
    s = Store(engine=engine).sleeve("s1")
    assert (s.lifecycle, s.wind_down_started, s.wind_down_deadline) == ("active", None, None)
