"""The candle-step test Strategy's plan (pe1-plans/candle-step-design.md; Advisor CANDLE-STEP; PM 22:38 UK): the pure
part every engine path reads, so a backtest, paper and a restart decide the same thing on the same bar.

A cycle is 6 bars, keyed by the bar's close time, never by when the Strategy started: buy (+Q), sell half (+Q/2),
sell the rest and short (-Q), cover half (-Q/2), close and reverse (+Q), wait (+Q). Q is fixed at step 0. A stop or
target inside a cycle makes the rest of it a no-op. A reduce step never adds, and a flip is a close then an open."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest

from sleeve_fund.strategies.candle_step import STEPS, Plan, Step, step_at

T0 = datetime(2026, 10, 15, 6, 0, tzinfo=UTC)  # a 6-minute boundary: 06:00 is step 0 on 1-minute bars
MIN = timedelta(minutes=1)


def test_the_step_clock_is_the_bars_close_time_never_the_start():
    assert [step_at(T0 + i * MIN, 1) for i in range(8)] == [(0, s) for s in range(6)] + [(1, 0), (1, 1)]
    assert step_at(T0, 1)[0] + 1 == step_at(T0 + 6 * MIN, 1)[0]
    assert step_at(T0 + 3 * MIN, 1) == step_at(T0 + 3 * MIN + timedelta(seconds=59), 1)  # one bar, one step
    assert step_at(T0 + 15 * MIN, 5) == (step_at(T0 + 15 * MIN, 5)[0], 3)  # 5-minute bars count bars, not minutes


def test_the_targets_are_the_specified_steps():
    assert [s.target for s in STEPS] == [D(1), D("0.5"), D(-1), D("-0.5"), D(1), D(1)]
    assert [s.opens for s in STEPS] == [True, False, True, False, True, False]


def _held(plan, step, held):
    return plan.orders(step, held)


def test_q_is_fixed_at_step_0_and_every_step_works_from_it():
    plan = Plan(cycle=7, q=D("0.8"))
    assert _held(plan, 0, D(0)) == [("open", 1, D("0.8"))]
    assert _held(plan, 1, D("0.8")) == [("reduce", -1, D("0.4"))]
    assert _held(plan, 2, D("0.4")) == [("close", -1, D("0.4")), ("open", -1, D("0.8"))]  # two orders, never one
    assert _held(plan, 3, D("-0.8")) == [("reduce", 1, D("0.4"))]
    assert _held(plan, 4, D("-0.4")) == [("close", 1, D("0.4")), ("open", 1, D("0.8"))]
    assert _held(plan, 5, D("0.8")) == []


def test_a_reduce_step_after_a_trim_halves_at_most_and_never_adds():
    plan = Plan(cycle=7, q=D("0.8"))
    assert _held(plan, 1, D("0.5")) == [("reduce", -1, D("0.1"))]  # the gate trimmed the open to 0.5: down to Q/2
    assert _held(plan, 1, D("0.3")) == []  # already below Q/2: nothing, never a buy
    assert _held(plan, 3, D("-0.3")) == []


def test_a_stop_or_target_inside_the_cycle_makes_the_rest_of_it_a_no_op():
    plan = Plan(cycle=7, q=D("0.8"))
    plan.closed_by_exit()
    for step in range(1, 6):
        assert _held(plan, step, D(0)) == [], step
    assert Plan(cycle=8, q=D("0.8")).orders(0, D(0)) == [("open", 1, D("0.8"))]  # the next cycle starts flat at step 0


def test_a_step_acts_once_however_often_its_bar_is_seen():
    plan = Plan(cycle=7, q=D("0.8"))
    assert plan.act(1, D("0.8")) == [("reduce", -1, D("0.4"))]
    assert plan.act(1, D("0.8")) == []  # the same bar again (a duplicate, or a restart before the fill shows)
    assert plan.done == {1}


def test_started_mid_cycle_it_opens_nothing_until_step_0():
    assert Plan(cycle=7, q=None).orders(3, D(0)) == []
    assert Plan(cycle=7, q=None).orders(2, D(0)) == []


@pytest.mark.parametrize("held, step", [(D("0.8"), 0), (D("0.8"), 5), (D("-0.8"), 2)])
def test_a_step_already_at_its_target_sends_nothing(held, step):
    assert Plan(cycle=7, q=D("0.8")).orders(step, held) == []


def test_the_run_window_ends_in_a_flatten_and_nothing_opens_after_it():
    started = T0
    plan = Plan(cycle=7, q=D("0.8"), started=started, run_for_hours=1)
    assert not plan.expired(started + timedelta(minutes=59))
    assert plan.expired(started + timedelta(hours=1))
    assert plan.after_expiry(D("0.4")) == [("close", -1, D("0.4"))]
    assert plan.after_expiry(D(0)) == []
    assert not Plan(cycle=7, q=D("0.8"), started=started, run_for_hours=None).expired(started + timedelta(days=30))


def test_step_is_a_plain_value():
    assert STEPS[2] == Step(2, D(-1), True)
