"""QA #203 (HoE ask 21:58 UK, CR203-1): a ping_pong short held through a restart in a flat market stays open.

Run A goes straight through; run B is the same feed killed 30 s before a candle close and restarted 30 s after it,
while the short leg is on and the market is flat (test_r_i5_2_missed_candle's harness). Nothing in the flat market
says exit, so the restart must not close the short: no fill in B after the restart, and B ends as short as A.
A set-up failure raises SetUpError, never AssertionError."""

from __future__ import annotations

from datetime import timedelta

import pytest

from paper_scenarios import NAME
from test_long_short import PERP
from test_r_i5_1_exit_lock import SetUpError, _ticks
from test_prop_inv import HOUR0, _at, _start_ns
from test_r_i5_2_missed_candle import _ab, _fills, _first_close

FEED = _ticks([(5, 0.0, 0.05), (12, 0.024, 0.05), (25, 0.0, 0.05)])  # buy, +1% sells and opens the short, then flat
FLAT = _at(_start_ns(HOUR0)) + timedelta(minutes=17)  # the rise ends here
ALL = ("entry", "exit", "stop_loss", "take_profit", "liquidation", "flatten")


@pytest.mark.parametrize("after_min", [3, 8])
def test_a_ping_pong_short_held_through_a_restart_in_a_flat_market_stays_open(after_min):
    def pick(a):
        if _first_close(a, ("exit",)) >= FLAT:
            raise SetUpError("A's long isn't sold before the flat stretch")
        return FLAT + timedelta(minutes=after_min)

    a, b, close, (restart, _) = _ab("ping_pong", dict(PERP), FEED, pick)
    qa, qb = a.last_equity(NAME)["qty"], b.last_equity(NAME)["qty"]
    if not qa < 0:
        raise SetUpError("A doesn't end short", qa)
    if [f for f in _fills(a, ALL) if f["ts"] >= close - timedelta(seconds=60)]:
        raise SetUpError("A trades in the flat stretch", close)
    late = [(f["ts"], f["intent"], f["qty"], f["price"]) for f in _fills(b, ALL) if f["ts"] >= restart]
    assert late == [], ("the restart traded the held short in a flat market", restart, late)
    assert abs(qb - qa) < 1e-12, ("B doesn't end as short as A", qa, qb)
