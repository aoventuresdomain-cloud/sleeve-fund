"""F-T-2 (Advisor BGL-PLAN 5, 14 Oct ~21:45 UK; HoE drop 2): a fill handler that fails holds new entries under its own
flag, and the most restrictive state wins: halt > stopped > exits-only > daily-loss pause > fill-handler hold > active.

- The hold never overwrites a stricter state (T-FIX8 wrote the PM's pause over a stop, an exits-only start and a
  daily-loss pause, and a Resume then lifted them).
- A Resume while held clears the hold and nothing else; the state under it keeps its own way out.
- The hold has no end time: the 00:00 roll that lifts a daily-loss pause leaves it, and it survives a restart.
- The raced-fill flatten keys on the flattening states (a halt, a live daily-loss pause), never on the hold.

Red first on main 76d7053 (T-FIX8's _set over the status). The failure goes through the strategy's own _report."""

import logging
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from sleeve_fund.paper.runtime import EXITS_ONLY, SleeveRuntime
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import FILL_HANDLER_PAUSE, LongFlatStrategy

T0 = datetime(2024, 3, 1, 23, 0, tzinfo=timezone.utc)
HELD = {"cash": 0.0, "qty": 1.0}
FLAT = {"cash": 10_000.0, "qty": 0.0}


@pytest.fixture
def store(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


def _runtime(store, t):
    if "s1" not in {s.name for s in store.sleeves()}:
        store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                            starting_balance=10_000, risk_profile="balanced")
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    return rt


def _fill_handler_fails(rt):
    """The strategy's on_order_filled raising, as its _reporting wrapper hands it to _report (paper)."""
    strategy = SimpleNamespace(handler_error_count=0, handler_errors=[], log=logging.getLogger("t"), runtime=rt,
                               _failed_handlers=set())
    LongFlatStrategy._report(strategy, "on_order_filled", RuntimeError("fill handler broke"))


def _state(store):
    s = store.sleeve("s1")
    return s.status, s.status_reason, s.paused_until


def _resume(store, rt, t, mark=HELD, equity=10_000):
    store.command("s1", "resume", "checked the fill")
    t[0] += timedelta(minutes=1)
    return rt.tick(equity=equity, price=equity, **mark)


def _prior(store, rt, t, prior):
    """Put the strategy in `prior`, holding a position; returns the equity it is marked at."""
    if prior == "halt":
        rt._set("halted", "drawdown 21.0% past the 20% limit")
    elif prior == "stopped":
        rt._set("stopped", "process stopped")
    elif prior == "exits_only":
        rt._set("paused", f"{EXITS_ONLY}: started for its exits only")
    elif prior == "pm_pause":
        store.command("s1", "pause", "PM wants it held")
        rt.tick(equity=10_000, price=10_000, **HELD)
    elif prior == "daily_pause":
        rt.tick(equity=10_000, price=10_000, **HELD)
        t[0] += timedelta(hours=10)
        assert rt.tick(equity=9_450, price=9_450, **HELD) == "flatten"  # down 5.5% on the day
        return 9_450
    return 10_000


PRIORS = ["halt", "stopped", "exits_only", "pm_pause", "daily_pause"]
FLATTENS = {"halt", "daily_pause"}  # the states whose own rule sells a raced fill (Advisor 23:05, SG7)


@pytest.mark.parametrize("prior", PRIORS)
def test_a_raced_fill_whose_handler_fails_keeps_the_stricter_state_and_one_resume_leaves_it(store, prior):
    t = [T0]
    rt = _runtime(store, t)
    equity = _prior(store, rt, t, prior)
    before = _state(store)
    t[0] += timedelta(minutes=1)
    rt.raced = "filled 120 ms after the block"  # the entry that raced the block's cancel
    _fill_handler_fails(rt)
    assert _state(store) == before, "the hold overwrote a stricter state"
    sent = rt.tick(equity=equity, price=equity, **HELD)
    assert (sent == "flatten") == (prior in FLATTENS), sent  # the hold never decides a flatten
    assert _state(store) == before
    _resume(store, rt, t, equity=equity)
    assert _state(store) == before, "a Resume while held lifted the state under the hold"
    assert not rt.can_open()


def test_the_hold_alone_blocks_entries_keeps_a_raced_fill_and_one_resume_clears_it(store):
    t = [T0]
    rt = _runtime(store, t)
    rt.tick(equity=10_000, price=10_000, **FLAT)
    rt.raced = "filled 80 ms after the block"
    _fill_handler_fails(rt)
    assert _state(store) == ("paused", FILL_HANDLER_PAUSE, None) and not rt.can_open()
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=10_000, price=10_000, **HELD) is None  # kept with its stop: the hold only blocks entries
    assert _resume(store, rt, t) is None
    assert _state(store)[:2] == ("running", "") and rt.can_open()


def test_the_hold_then_a_daily_loss_breach_both_hold_and_each_has_its_own_way_out(store):
    t = [T0]
    rt = _runtime(store, t)
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(hours=10)
    _fill_handler_fails(rt)
    assert rt.tick(equity=9_450, price=9_450, **HELD) == "flatten"  # the day's loss limit still acts while held
    status, _, until = _state(store)
    assert status == "paused" and until == datetime(2024, 3, 3, tzinfo=timezone.utc)  # the daily pause, its roll
    _resume(store, rt, t, mark=FLAT, equity=9_450)  # clears the hold only
    assert _state(store)[2] == until and not rt.can_open()
    t[0] = until + timedelta(seconds=30)
    rt.tick(equity=9_450, price=9_450, cash=9_450, qty=0.0)
    assert rt.can_open() and _state(store)[0] == "running"  # the roll lifts the daily pause


@pytest.mark.parametrize("restart", [False, True])
def test_a_daily_pause_then_the_hold_the_roll_leaves_the_hold_and_a_restart_keeps_it(store, restart):
    t = [T0]
    rt = _runtime(store, t)
    _prior(store, rt, t, "daily_pause")
    until = _state(store)[2]
    _fill_handler_fails(rt)
    t[0] = until + timedelta(seconds=30)
    if restart:
        rt = _runtime(store, t)
    rt.tick(equity=9_450, price=9_450, cash=9_450, qty=0.0)
    assert not rt.can_open() and _state(store) == ("paused", FILL_HANDLER_PAUSE, None)  # the hold has no end time
    _resume(store, rt, t, mark={"cash": 9_450, "qty": 0.0}, equity=9_450)
    assert rt.can_open() and _state(store)[:2] == ("running", "")


def test_the_hold_survives_a_restart_and_a_resume_after_it_clears_it(store):
    t = [T0]
    rt = _runtime(store, t)
    rt.tick(equity=10_000, price=10_000, **FLAT)
    _fill_handler_fails(rt)
    t[0] += timedelta(minutes=5)
    rt = _runtime(store, t)  # the restart
    assert _state(store) == ("paused", FILL_HANDLER_PAUSE, None) and not rt.can_open()
    _resume(store, rt, t, mark=FLAT)
    assert _state(store)[:2] == ("running", "") and rt.can_open()


def test_a_resume_that_clears_the_hold_under_a_halt_keeps_the_halts_owed_flatten_across_a_restart(store):
    """The hold's Resume is not the halt's: a restart after it still owes the halt's flatten (S-3)."""
    t = [T0]
    rt = _runtime(store, t)
    rt.tick(equity=10_000, price=10_000, **HELD)
    store.event("s1", "error", "risk_halt", "drawdown 21.0% past the 20% limit; flattened, PM must resume", ts=t[0])
    rt._set("halted", "drawdown 21.0% past the 20% limit")
    _fill_handler_fails(rt)
    _resume(store, rt, t)
    rt = _runtime(store, t)
    assert _state(store)[0] == "halted" and rt._owed_flatten is not None


@pytest.mark.parametrize("order,held", [(("fill_hold", "fill_hold_cleared"), False),
                                        (("fill_hold_cleared", "fill_hold"), True)])
def test_a_restart_restores_the_hold_by_journal_order_not_by_time(store, order, held):
    """Advisor F-T-2-DESIGN: two hold events in the same instant are read in the order the journal wrote them."""
    t = [T0]
    _runtime(store, t)
    for kind in order:
        store.event("s1", "info", kind, kind, ts=t[0])
    assert _runtime(store, t).fill_hold is held
