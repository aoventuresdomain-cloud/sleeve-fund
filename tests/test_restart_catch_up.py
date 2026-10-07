"""Restarts and late bars (Advisor 7 Oct 21:42 UK; R-I5-2, m13-E3 retired on the restart path): a restart catches up
exits on the candles it missed, never entries; while running, a late bar opens under 90 s late and is skipped past it.
On the hub-fed paper node replayed, with test_hub_146_qa's probe strategy."""

from datetime import timezone

import pytest
from test_hub_146_qa import _guard_marks, _probe  # noqa: F401 - its autouse fixtures: the probe, the guard marks


def _restarted_flat(prices, down, back):
    """test_hub_146_qa.restart without a position: down from minute `down` (its last heartbeat) to `back`."""
    from test_hub_146_qa import M, START, paper, stored_history

    return paper(prices, gone=frozenset(range(0, int(back * 60))), lost={START + k * M for k in range(int(back) + 1)},
                 heartbeat=START + int(down * M), history=stored_history(prices, START + int(back * M)), warmup=30,
                 enter=10, leave=60, stop=None)


def test_restart_a_missed_candle_30_s_late_with_an_entry_signal_enters_nothing():
    """Pin 1: down from 00:09:30 to 00:10:30, over the 00:10 close that says long: no entry on it. The next live
    close, 00:11, decides the entry."""
    from test_hub_146_qa import flat_prices, hhmm, timing_of

    run = _restarted_flat(flat_prices(20), 9.5, 10.5)
    assert hhmm(timing_of(run, "entry")["bar_close"]) == "00:11"


def test_restart_a_missed_candle_with_an_exit_signal_exits_once_at_market_as_a_late_exit():
    """Pin 2: long since 00:05, down from 00:10 to 00:20:30; the 00:15 close said exit: one market exit as the restart
    comes back, journalled "Late exit, missed candle 00:15", before the next close."""
    from test_hub_146_qa import M, START, flat_prices, restart

    run = restart(flat_prices(30), 10, 20.5, leave=15, warmup=30, stop=None)
    exits = [o for o in run.orders if o["intent"] == "exit"]
    assert len(exits) == 1 and exits[0]["reason"] == "Late exit, missed candle 00:15", exits
    sent = exits[0]["ts"] if exits[0]["ts"].tzinfo else exits[0]["ts"].replace(tzinfo=timezone.utc)
    assert sent.timestamp() * 1e9 < START + 21 * M  # before the next close
    assert [e for e in run.kinds("late_exit") if "missed candle 00:15" in e["message"]]


@pytest.mark.parametrize("perp, side", [(False, 1), (True, -1)])
def test_restart_killed_10_s_after_a_close_before_its_decision_exits_once_on_that_candle(perp, side):
    """Pin 5 (R203-1, Advisor regrade): the last heartbeat, 00:15:10, came after the 00:15 close but before its bar
    was decided, and that close said exit. Back at 00:15:40: one market exit, "Late exit, missed candle 00:15", before
    the 00:16 close, and no entry caught up."""
    from test_hub_146_qa import M, START, flat_prices, restart

    run = restart(flat_prices(30), 10, 15 + 40 / 60, heartbeat=15 + 10 / 60, side=side, perp=perp, leave=15,
                  warmup=30, stop=None)
    exits = [o for o in run.orders if o["intent"] == "exit"]
    assert len(exits) == 1 and exits[0]["reason"] == "Late exit, missed candle 00:15", exits
    sent = exits[0]["ts"] if exits[0]["ts"].tzinfo else exits[0]["ts"].replace(tzinfo=timezone.utc)
    assert sent.timestamp() * 1e9 < START + 16 * M
    assert not [o for o in run.orders if o["intent"] == "entry" and o["order_id"] != "O-held"]


@pytest.mark.parametrize("lag_s, opens", [(30, True), (120, False)])
def test_running_a_late_bar_opens_under_90_s_and_is_skipped_past_it(lag_s, opens):
    """Pins 3 and 4: no restart, every bar `lag_s` late."""
    from test_hub_146_qa import S, flat_prices, paper

    run = paper(flat_prices(60), enter=15, leave=50, lag=lag_s * S, stop=None)
    assert bool([o for o in run.orders if o["intent"] == "entry"]) is opens
    assert bool(run.kinds("late_entry_skipped")) is not opens
