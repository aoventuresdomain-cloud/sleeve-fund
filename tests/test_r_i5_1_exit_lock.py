"""R-I5-1 (QA Tester 2, 7 Oct 2026; HoQA graded MUST FIX before strategy testing): a paper restart after a stop
loses the exit lock, so the restarted strategy enters again where the uninterrupted run stays out.

After a stop or target closes a position, the model may not enter that side again until its signal has moved off it
(base.py _exit_lock). A restart rebuilds the lock from the journal only for models that override resume_leg
(_plan_resume); for the rest the lock is gone after a restart.

One scenario for every model, spot, with a 2% stop: the price rises for long enough to put the model long, then one
trade 2.5% down goes through the stop and the next trade is back where it was, inside the same minute, so the candle
closes where the signal still says long. Run A goes straight through and stays out (locked). Run B is the same feed,
killed at the first candle close after the stop and restarted on its journal (the paper node's restart balances); it
must make no entry A doesn't. Same recorder and replay as PROP-INV (test_prop_inv.py).

Cells (main 1709cd9)
  test_r_i5_1_lost_lock[buy_and_hold|ping_pong|trend_filter]  strict xfail: no resume_leg, so the restarted run enters
      again where A stays out
  test_r_i5_1_kept_lock_rsi_cross       guard, green: rsi_cross overrides resume_leg and stays out after the restart
  test_r_i5_1_kept_lock_guard_is_live   green: with the lock's rebuild switched off the restarted rsi_cross does enter,
      so the guard passes because of the lock
Every cell first checks its set-up (A entered, the stop filled, A made no entry after it); a failed set-up raises
SetUpError, never AssertionError, so a strict xfail can't hide it.

Not cells, with why (HoQA asked for all four models without resume_leg and a dip_buy guard):
  rsi_pullback: its entry is a one-candle event (RSI under its level, close over its EMA, a volume spike), so the lock
      only matters when the event repeats on the candles after the stop; a restart here has no history to warm up
      from, so the restarted run can't decide for the EMA's length. No divergence found over 8 feeds. Lost lock
      inferred from the code only.
  donchian: decides on daily candles and needs at least 10 days of returns for its volatility, both before the stop
      and again after the restart (no history here): weeks of feed per cell. Lost lock inferred from the code only.
  dip_buy guard: its daily trend needs days of candles, again after the restart, so a restarted run is kept out by its
      warm-up, not the lock, and a cell would pass either way. rsi_cross (also a resume_leg model) is the guard.
"""

from __future__ import annotations

import pathlib
import tempfile
from datetime import timedelta

import pytest

from paper_scenarios import NAME, replay_into
from sleeve_fund.store import Store
from test_long_short import _meta
from test_prop_inv import HOUR0, START, _at, _aware, _prices, _record_prices, _restart_meta, _start_ns

STOP = {"stop_loss": 0.02}
SPIKE = [(0, -0.025, 0.05), (0, 1 / 0.975 - 1, 0.05)]  # one trade through the 2% stop, the next straight back
RISE = [(5, 0.0, 0.05), (30, 0.004, 0.05), *SPIKE, (10, 0.001, 0.05)]  # every trend or breakout signal turns long


def _ticks(legs, px=60_000.0):
    """A tick a second, as test_prop_inv._prices, with each leg's trade size: (prices, tick seconds, sizes)."""
    prices, sizes = [], []
    for minutes, move, size in legs:
        prices += _prices([(minutes, move)], prices[-1] if prices else px)
        sizes += [size] * (minutes * 60 or 1)
    return prices, None, sizes


MODELS = {
    # model: (strategy params, the feed)
    "buy_and_hold": ({}, _ticks(RISE)),
    "ping_pong": ({"rise": 0.4, "dip": 0.005}, _ticks([*RISE[:-1], (10, -0.06, 0.05)])),  # never rises 40%: long leg
    "trend_filter": ({"fast": 2, "slow": 3}, _ticks(RISE)),
    # a leg model: an RSI(2) cross back above 30 opens a long leg that runs to RSI 99 or 48 candles
    "rsi_cross": ({"rsi_period": 2, "long_entry": 30, "long_exit": 99, "short_entry": 99.5, "short_exit": 98},
                  _ticks([(5, 0.0, 0.05), (3, -0.01, 0.05), (2, 0.004, 0.05), *SPIKE,
                          *[(1, 0.002 if i % 2 else -0.001, 0.05) for i in range(12)]])),  # RSI stays under 99
}


class SetUpError(Exception):
    pass


def _run(model):
    params, (prices, at, sizes) = MODELS[model]
    secs = list(range(len(prices))) if at is None else at
    meta = _meta(START, {**STOP, **params})
    meta["sleeve"]["strategy"] = model
    a, b = Store.in_memory(), Store.in_memory()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _record_prices(tmp / "a.jsonl.gz", meta, prices, HOUR0, at=at, sizes=sizes)
        replay_into(a, tmp / "a.jsonl.gz")
        stops = [o for o in a.orders(NAME, limit=1000) if o["intent"] == "stop_loss" and o["filled_qty"] > 0]
        if not stops:  # a set-up failure is never an AssertionError, so the strict xfail can't swallow it
            raise SetUpError(model, "the spike filled no stop in A",
                             [(o["ts"], o["side"], o["intent"], o["status"]) for o in a.orders(NAME, limit=1000)])
        stopped = max(_aware(o["ts"]) for o in stops)
        close = int((stopped - _at(_start_ns(HOUR0))).total_seconds()) // 60 * 60 + 60  # the next candle close
        kill = next(i for i, t in enumerate(secs) if t >= close) + 1  # killed once its first trade is seen
        _record_prices(tmp / "b0.jsonl.gz", meta, prices[:kill], HOUR0, at=at, sizes=sizes)
        replay_into(b, tmp / "b0.jsonl.gz")
        restart = _restart_meta(b, {**STOP, **params}, 1)
        restart["sleeve"]["strategy"] = model
        _record_prices(tmp / "b1.jsonl.gz", restart, prices, HOUR0, first=kill, at=at, sizes=sizes)
        replay_into(b, tmp / "b1.jsonl.gz")
    after = stopped + timedelta(microseconds=1)
    entries = {n: [(_aware(o["ts"]), o["side"]) for o in s.orders(NAME, limit=1000)
                   if o["intent"] == "entry" and o["filled_qty"] > 0 and _aware(o["ts"]) > after]
               for n, s in (("A", a), ("B", b))}
    if entries["A"]:
        raise SetUpError(model, "A entered again after its stop", entries["A"])
    return entries["B"]


@pytest.mark.parametrize("model", ["buy_and_hold", "ping_pong", "trend_filter"])
def test_r_i5_1_lost_lock(model):
    assert _run(model) == [], (model, "B entered again after the restart, where A stays out")


def test_r_i5_1_kept_lock_rsi_cross():
    """Guard: rsi_cross overrides resume_leg, so the restart puts its long leg and the stop's lock back."""
    assert _run("rsi_cross") == []


def test_r_i5_1_kept_lock_guard_is_live(monkeypatch):
    """The guard above proves something: with the lock's rebuild switched off, the restarted rsi_cross does enter
    again (its leg is back on long), so its passing is the lock's doing, not a warm-up that keeps it out."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    monkeypatch.setattr(LongFlatStrategy, "_lock_resumed", lambda self, r: r.__setitem__("lock_ns", None))
    assert _run("rsi_cross") != []
