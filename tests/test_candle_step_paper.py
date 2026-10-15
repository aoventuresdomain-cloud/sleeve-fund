"""The candle-step test Strategy in the paper runtime (pe1-plans/candle-step-design.md; Advisor CANDLE-STEP; HoE: "a
restart mid-cycle resumes or closes cleanly, never doubles a position"). Run A goes straight through; run B is the same
feed killed 30 s before a step's candle close and restarted 30 s after it, on its journal (test_r_i5_2's harness)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

import pytest

from paper_scenarios import NAME
from test_long_short import PERP
from test_prop_inv import HOUR0, _at, _start_ns
from test_r_i5_1_exit_lock import _ticks
from test_r_i5_2_missed_candle import _ab
from sleeve_fund.strategies.candle_step import step_at

PARAMS = {**PERP, "size_share": 0.6}
FLAT = _ticks([(20, 0.0, 0.05)])  # 20 flat minutes: the step clock alone trades, no stop is reached


def _orders(store):
    rows = [o for o in store.orders(NAME, limit=1000) if (o["filled_qty"] or 0) > 0]
    return sorted(rows, key=lambda o: (o["ts"], o["id"]))


def _path(store):
    """The signed position after each filled order, in order."""
    held, out = D(0), []
    for o in _orders(store):
        held += D(str(o["filled_qty"])) * (1 if o["side"] == "BUY" else -1)
        out.append(held)
    return out


def _step_close(n, step=2):
    """The close of the n-th candle that is `step` (by default 2, the flip) after the feed starts."""
    t0 = _at(_start_ns(HOUR0))
    hits = [t0 + timedelta(minutes=m) for m in range(1, 20) if step_at(t0 + timedelta(minutes=m), 1)[1] == step]
    return hits[n]


def test_paper_runs_the_cycle_as_the_backtest_does():
    a, _, _, _ = _ab("candle_step", PARAMS, FLAT, lambda a: _step_close(1))
    orders = _orders(a)
    q = D(str(orders[0]["filled_qty"]))
    steps = [(o["side"], D(str(o["filled_qty"])), o["intent"]) for o in orders[:7]]
    assert steps == [("BUY", q, "entry"), ("SELL", q / 2, "exit"), ("SELL", q / 2, "exit"), ("SELL", q, "entry"),
                     ("BUY", q / 2, "exit"), ("BUY", q / 2, "exit"), ("BUY", q, "entry")]
    assert max(abs(h) for h in _path(a)) == q


def test_a_restart_over_the_flip_never_doubles_and_carries_on():
    a, b, close, (restart, _) = _ab("candle_step", PARAMS, FLAT, lambda a: _step_close(1))
    qa = D(str(_orders(a)[0]["filled_qty"]))
    assert max(abs(h) for h in _path(b)) <= qa, ("a position above Q after the restart", _path(b))
    # Down over the flip (step 2) and the cover (step 3, a no-op on a long), B carries on in the same cycle from the
    # journal: step 4's open makes up only what is missing to +Q, never more.
    after = [(o["side"], D(str(o["filled_qty"])), o["intent"], o["signal"]["step"]) for o in _orders(b)
             if o["ts"] >= restart]
    assert after[0] == ("BUY", qa / 2, "entry", 4)
    # From the next cycle on, B trades as A does: the restart cost it at most the steps it was down for.
    nxt = close + timedelta(minutes=4)  # step 0 of the next cycle
    tail = lambda s: [(o["side"], o["filled_qty"], o["intent"]) for o in _orders(s) if o["ts"] >= nxt]
    assert tail(b) == tail(a)


def test_a_restart_before_sell_half_resumes_the_carried_cycle_and_never_doubles():
    """drop-b-18 risk (HoE 01:44 UK): a Drop B deploy restarts the engine mid-cycle while the timed test runs. From the
    second cycle on, step 0 carries the last cycle's long in and sends nothing, so "sell half" (step 1) is the cycle's
    first order. B is down over that close and back before the flip's ("sell the rest and short"): it resumes the same
    cycle with the same Q, the flip closes what is held and shorts exactly Q, the position never passes Q on either
    side, and from the flip on it trades as the straight-through run does. With test_a_restart_over_the_flip_* (down
    over the flip, after sell half) this brackets a restart between the two."""
    a, b, close, (restart, _) = _ab("candle_step", PARAMS, FLAT, lambda a: _step_close(1, step=1))
    qa = D(str(_orders(a)[0]["filled_qty"]))
    assert max(abs(h) for h in _path(b)) <= qa, ("a position above Q after the restart", _path(b))
    cycle = step_at(close, 1)[0]
    mine = lambda s: [o for o in _orders(s) if o["ts"] >= restart and o["signal"].get("cycle") == cycle]
    flip = [(o["side"], D(str(o["filled_qty"])), o["intent"]) for o in mine(b) if o["signal"]["step"] == 2]
    assert flip, ("B sat the cycle out after the restart: no flip", [(o["ts"], o["side"]) for o in _orders(b)])
    assert flip[-1] == ("SELL", qa, "entry") and all(side == "SELL" for side, _, _ in flip), flip
    assert sum(q for _, q, _ in flip) == 2 * qa  # closes the long Q it held (the missed half included), shorts Q
    assert all(D(str(o["signal"]["q"])) == qa for o in mine(b))
    flip_close = close + timedelta(minutes=1)
    tail = lambda s: [(o["side"], o["filled_qty"], o["intent"]) for o in _orders(s) if o["ts"] > flip_close]
    assert tail(b) == tail(a)


def test_a_restart_after_the_run_window_opens_nothing():
    """The window counts from the first start, journaled with every step: B, restarted after it has passed, only
    closes what it holds and opens nothing (design: run_for_hours)."""
    hours = 9 / 60  # ends before the second flip's close; B is down over that close
    a, b, close, (restart, _) = _ab("candle_step", {**PARAMS, "run_for_hours": hours}, FLAT, lambda a: _step_close(1))
    opened = [o for o in _orders(b) if o["ts"] >= restart and o["intent"] == "entry"]
    assert opened == [], [(o["ts"], o["side"], o["filled_qty"]) for o in opened]
    assert _path(b)[-1] == 0  # flat once the window is over
    assert _path(a)[-1] == 0


@pytest.mark.parametrize("code", ["exits_only", "paused", "daily_pause", "halted", "portfolio_halt", "stale_data"])
def test_once_entries_are_held_only_the_reduce_and_the_close_run(monkeypatch, code):
    """Design (e), (f), (h): under exits-only, a pause, the daily brake, a halt or stale data, the cycle's reduce
    steps and closes still run and nothing opens. Held from just after step 0's buy: step 1 sells half, step 2's flip
    closes the rest and its open (the short) is held, and every later open is held too, so it ends flat."""
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.strategies.candle_step import CandleStep, CandleStepConfig

    class Held(CandleStep):
        _q_sent = False

        def entry_sent(self, qty, values):
            super().entry_sent(qty, values)
            self._q_sent = True
            if self.runtime is not None:
                self.runtime.holds[code] = "held for the cell"

        def _market_seen(self):  # stale data clears on a fresh print: here the feed stays stale once it goes
            super()._market_seen()
            if code == "stale_data" and self._q_sent:
                self.runtime.holds[code] = "last price 400 s old. It clears when data resumes"

    monkeypatch.setitem(REGISTRY, "candle_step", (Held, CandleStepConfig))
    a, _, _, _ = _ab("candle_step", PARAMS, FLAT, lambda a: _step_close(1))
    orders = _orders(a)
    q = D(str(orders[0]["filled_qty"]))
    assert [(o["side"], D(str(o["filled_qty"])), o["intent"]) for o in orders] == [
        ("BUY", q, "entry"), ("SELL", q / 2, "exit"), ("SELL", q / 2, "exit")]
    assert _path(a)[-1] == 0
