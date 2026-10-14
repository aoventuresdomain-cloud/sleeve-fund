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


def _step_close(n):
    """The close of the n-th candle that is step 2 (the flip) after the feed starts."""
    t0 = _at(_start_ns(HOUR0))
    hits = [t0 + timedelta(minutes=m) for m in range(1, 20) if step_at(t0 + timedelta(minutes=m), 1)[1] == 2]
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
