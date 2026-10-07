"""R-I5-2 (QA Tester 2, 7 Oct 2026; the Advisor re-graded it MUST FIX, 21:15 UK): a paper process killed inside a
candle misses that candle's close, so an exit decided on that close comes a candle late after the restart.

The rule (Advisor): on a restart, exits on candles closed while the process was down are caught up; entries are
never replayed. Each cell: run A goes straight through; run B is the same feed (test_prop_inv's recorder), killed 30 s
before a candle close and restarted 30 s after it, on its journal (the paper node's restart balances), so the candle
closes while B is down.

Cells (main 1709cd9)
  test_r_i5_2_missed_close_exit_is_caught_up[ping_pong|trend_filter|rsi_cross]  strict xfail: A exits on a candle's
      close; B, down over that close, must book the exit on its first decision after the restart: the same qty, at the
      restart's price within SLIP, before the next candle closes (no candle of delay beyond the restart)
  test_r_i5_2_missed_close_entry_is_not_replayed  guard, green: a candle closed while B is down and flat, whose close
      enters in A, opens nothing on the restart (no entry before the next candle closes)
  test_r_i5_2_stop_unaffected  guard, green: a stop hit after the restart, inside the candle the kill cut, fills in B
      as in A (same qty, price within SLIP)
A set-up failure raises SetUpError, never AssertionError, so a strict xfail can't hide it.
"""

from __future__ import annotations

import inspect
import pathlib
import tempfile
from datetime import timedelta

import pytest

from paper_scenarios import NAME, replay_into
from sleeve_fund.store import Store
from test_long_short import _meta
from test_prop_inv import HOUR0, SLIP, START, _at, _aware, _record_prices, _restart_meta, _start_ns
from test_r_i5_1_exit_lock import SPIKE, STOP, SetUpError, _ticks

OUT = 30  # seconds down on each side of the candle close

EXITS = {
    # model: (strategy params, the feed); each model exits on a candle's close
    "ping_pong": ({}, _ticks([(5, 0.0, 0.05), (12, 0.024, 0.05), (5, 0.0, 0.05)])),  # sells 1% above its buy
    "trend_filter": ({"fast": 2, "slow": 3}, _ticks([(5, 0.0, 0.05), (10, 0.02, 0.05), (6, -0.02, 0.05)])),
    "rsi_cross": ({"rsi_period": 2}, _ticks([(5, 0.0, 0.05), (3, -0.01, 0.05), (2, 0.004, 0.05),
                                            (6, 0.02, 0.05)])),  # long on RSI back above 30, out at 55
}


def _ab(model, params, feed, pick):
    """Runs A, picks the candle close `pick(a)` (an aware datetime) to be missed, then runs B killed OUT s before it and
    restarted OUT s after it. Returns (a, b, the close, the restart's first tick time and price)."""
    prices, _, sizes = feed
    meta = _meta(START, {**STOP, **params})
    meta["sleeve"]["strategy"] = model
    t0 = _at(_start_ns(HOUR0))
    a, b = Store.in_memory(), Store.in_memory()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _record_prices(tmp / "a.jsonl.gz", meta, prices, HOUR0, sizes=sizes)
        replay_into(a, tmp / "a.jsonl.gz")
        close = pick(a)
        s = int((close - t0).total_seconds())
        if s % 60 or not OUT < s < len(prices) - OUT - 60:
            raise SetUpError(model, "the picked candle close is not a whole minute inside the feed", close)
        _record_prices(tmp / "b0.jsonl.gz", meta, prices[:s - OUT], HOUR0, sizes=sizes)
        replay_into(b, tmp / "b0.jsonl.gz")
        restart = _restart_meta(b, {**STOP, **params}, 1)
        restart["sleeve"]["strategy"] = model
        _record_prices(tmp / "b1.jsonl.gz", restart, prices, HOUR0, first=s + OUT, sizes=sizes)
        # The restart warms up from the history the hub kept while the process was down, as the node does: up to the
        # restart only, never later ticks (no look-ahead). A build without the `history` hook restarts cold (still red).
        _record_prices(tmp / "h.jsonl.gz", meta, prices[:s + OUT], HOUR0, sizes=sizes)
        if "history" in inspect.signature(replay_into).parameters:
            replay_into(b, tmp / "b1.jsonl.gz", history=tmp / "h.jsonl.gz")
        else:
            replay_into(b, tmp / "b1.jsonl.gz")
    return a, b, close, (close + timedelta(seconds=OUT), prices[s + OUT])


def _fills(store, intents):
    intent = {o["order_id"]: o["intent"] for o in store.orders(NAME, limit=1000)}
    return sorted(({**f, "ts": _aware(f["ts"]), "intent": intent.get(f["order_id"])}
                   for f in store.fills(NAME, limit=1000) if intent.get(f["order_id"]) in intents),
                  key=lambda f: (f["ts"], f["id"]))


def _first_close(store, intents, after=None):
    fills = [f for f in _fills(store, intents) if after is None or f["ts"] > after]
    if not fills:
        raise SetUpError("A made no fill of", intents, "after", after)
    return fills[0]["ts"].replace(second=0, microsecond=0)  # the close that decided it (sent just after it)


@pytest.mark.parametrize("model", list(EXITS))
def test_r_i5_2_missed_close_exit_is_caught_up(model):
    params, feed = EXITS[model]
    a, b, close, (restart, px) = _ab(model, params, feed, lambda a: _first_close(a, ("exit",)))
    want = [f for f in _fills(a, ("exit",)) if close <= f["ts"] < close + timedelta(seconds=60)]
    if not want:
        raise SetUpError(model, "A's exit isn't on the picked close", close)
    got = [f for f in _fills(b, ("exit",)) if f["ts"] >= restart]
    qty = sum(f["qty"] for f in want)
    on_time = [f for f in got if f["ts"] < close + timedelta(seconds=60)]
    assert abs(sum(f["qty"] for f in on_time) - qty) < 1e-12, \
        (model, "the exit on the missed close isn't booked on the restart's first decision", close, restart,
         [(f["ts"], f["qty"], f["price"]) for f in want], [(f["ts"], f["qty"], f["price"]) for f in got])
    assert all(abs(f["price"] - px) <= SLIP * px for f in on_time), (model, "caught-up exit price", px, on_time)


def test_r_i5_2_missed_close_entry_is_not_replayed():
    """ping_pong sells 1% above its buy, then buys back 0.5% below that sale: B is down over the buy-back's close."""
    params, feed = {}, _ticks([(5, 0.0, 0.05), (12, 0.024, 0.05), (8, -0.03, 0.05), (5, 0.0, 0.05)])
    a, b, close, (restart, _) = _ab("ping_pong", params, feed,
                                    lambda a: _first_close(a, ("entry",), after=_first_close(a, ("exit",))))
    replayed = [(f["ts"], f["qty"], f["price"]) for f in _fills(b, ("entry",))
                if restart <= f["ts"] < close + timedelta(seconds=60)]
    assert replayed == [], ("the entry on the missed close was replayed on the restart", close, replayed)


def test_r_i5_2_stop_unaffected():
    """A stop hit after the restart, inside the candle the kill cut: B's stop fills as A's. The spike is 45 s into the
    minute after the 35th close: B is killed 30 s before that close and restarted 30 s after it, so before the spike."""
    prices, _, sizes = _ticks([(5, 0.0, 0.05), (30, 0.004, 0.05)])
    prices, sizes = prices + [prices[-1]] * 45, sizes + [0.05] * 45
    tail, _, more = _ticks([*SPIKE, (10, 0.001, 0.05)], px=prices[-1])
    feed = (prices + tail, None, sizes + more)
    a, b, close, (restart, _) = _ab("trend_filter", {"fast": 2, "slow": 3}, feed,
                                    lambda a: _at(_start_ns(HOUR0)) + timedelta(minutes=35))
    want, got = _fills(a, ("stop_loss",)), _fills(b, ("stop_loss",))
    if not want or any(f["ts"] < restart for f in want):
        raise SetUpError("A's stop isn't after the restart point", [f["ts"] for f in want], restart)
    assert len(got) == len(want) and all(abs(g["qty"] - w["qty"]) < 1e-12 and abs(g["price"] - w["price"]) <=
                                         SLIP * w["price"] for g, w in zip(got, want)), (want, got)
