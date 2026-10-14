"""The candle-step test Strategy on the engine (pe1-plans/candle-step-design.md; Advisor CANDLE-STEP and MUSTs; PM 22:38
UK): its cycle as fills, through the same order path every Strategy uses. Red first: the plan is in
test_candle_step_plan.py; these pin what the engine does with it."""

from decimal import Decimal as D

import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.candle_step import step_at
from test_backtest import _path

PARAMS = {"market": "perp", "allow_short": True, "size_share": 0.6, "stop_loss": 0.001}


def _from_step_0(prices, n, px=100.0):
    """Flat closes from the first daily bar that is step 0 of a cycle, with the bars before it at the same price, so
    nothing reaches the 0.1% stop and the step clock alone trades."""
    first = next(i for i, t in enumerate(prices.index) if step_at(t.to_pydatetime(), 1440)[1] == 0)
    return first, [px] * (first + n)


def _fills(res):
    fills = res.fills.sort_values("ts_last")
    return [(row["side"], D(str(row["filled_qty"])), res.decisions[o]["intent"]) for o, row in fills.iterrows()]


def test_one_cycle_buys_halves_flips_in_two_orders_halves_and_reverses(prices, instrument):
    first, closes = _from_step_0(prices, 7)
    res = run_backtest("candle_step", _path(prices, closes), instrument, PARAMS, half_spread=0)
    fills = _fills(res)
    q = fills[0][1]
    assert q == D(60)  # 0.6 of the 10,000 at 100, the step's own limit
    assert fills == [("BUY", q, "entry"), ("SELL", q / 2, "exit"), ("SELL", q / 2, "exit"), ("SELL", q, "entry"),
                     ("BUY", q / 2, "exit"), ("BUY", q / 2, "exit"), ("BUY", q, "entry")]


def test_a_flip_never_holds_more_than_one_side_at_once(prices, instrument):
    first, closes = _from_step_0(prices, 7)
    res = run_backtest("candle_step", _path(prices, closes), instrument, PARAMS, half_spread=0)
    # Never Q long and Q short at once (a double-size order reads ~1.2): the close fills first. Q is fixed at step 0, so
    # the fees paid since then lift it a hair above 0.6 of the smaller equity.
    assert res.exposure.abs().max() <= 0.6 * 1.01


def test_a_stop_inside_the_cycle_makes_the_rest_of_it_a_no_op(prices, instrument):
    first, closes = _from_step_0(prices, 12)
    closes[first + 1:first + 6] = [99.7] * 5  # through the 0.1% stop on the bar after the buy, then flat
    closes[first + 6:] = [99.7] * (len(closes) - first - 6)
    res = run_backtest("candle_step", _path(prices, closes), instrument, PARAMS, half_spread=0)
    intents = [i for _, _, i in _fills(res)]
    assert intents[:2] == ["entry", "stop_loss"]
    assert intents[2] == "entry"  # nothing until the next cycle's step 0, which buys again
    buys = res.fills.sort_values("ts_last")
    assert step_at(buys["ts_last"].iloc[2].to_pydatetime(), 1440)[1] == 0


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_size_share_outside_its_range_is_refused(bad, prices, instrument):
    with pytest.raises(ValueError, match="size_share"):
        run_backtest("candle_step", _path(prices, [100.0] * 8), instrument, {**PARAMS, "size_share": bad})


def test_a_stop_after_a_reduce_sells_only_what_is_left(prices, instrument):
    first, closes = _from_step_0(prices, 6)  # one cycle: steps 0-5
    closes[first + 2:] = [99.7] * (len(closes) - first - 2)  # through the stop after the half sold, before the flip
    res = run_backtest("candle_step", _path(prices, closes), instrument, PARAMS, half_spread=0)
    fills = _fills(res)
    q = fills[0][1]
    # The resting stop shrank to Q/2 with the reduce (the venue keeps a reduce-only order within the position; paper's
    # _sync_watched_stop does the same): it never sells more than is held, so it never opens a short.
    assert fills[:3] == [("BUY", q, "entry"), ("SELL", q / 2, "exit"), ("SELL", q / 2, "stop_loss")]
    assert len(fills) == 3  # steps 2-5 are no-ops once the stop has ended the cycle
    stop = res.fills[[res.decisions[o]["intent"] == "stop_loss" for o in res.fills.index]]
    assert D(str(stop["quantity"].iloc[0])) == q / 2  # the order itself, not only its reduce-only fill


def test_past_the_run_window_it_closes_and_opens_nothing(prices, instrument):
    first, closes = _from_step_0(prices, 12)
    hours = (first + 3) * 24  # from the first bar it sees: the window ends on step 3, short Q
    res = run_backtest("candle_step", _path(prices, closes), instrument, {**PARAMS, "run_for_hours": hours},
                       half_spread=0)
    fills = _fills(res)
    q = fills[0][1]
    assert fills == [("BUY", q, "entry"), ("SELL", q / 2, "exit"), ("SELL", q / 2, "exit"), ("SELL", q, "entry"),
                     ("BUY", q, "exit")]
