"""P1-D13: where a resting exit is booked when only the bar is known (sleeve_fund.instruments.exit_price)."""

import pytest

from sleeve_fund.instruments import STOP_SLIPPAGE_FLOOR, exit_price, stop_slippage

BAR = (60_000.0, 62_000.0, 57_000.0, 59_000.0)  # open, high, low, close


@pytest.mark.parametrize("side, trigger, rested, want", [
    (1, 58_800.0, True, ("stop", 57_000.0)),  # traded through inside the bar: the low
    (1, 61_000.0, True, ("stop", 60_000.0)),  # the open was already through it: the open
    (1, 61_000.0, False, ("stop", 57_000.0)),  # placed inside the bar (its entry filled there): it can't have gapped
    (-1, 61_200.0, True, ("stop", 62_000.0)),  # a short's stop: the high
    (-1, 59_000.0, True, ("stop", 60_000.0)),  # a short's stop gapped through at the open
])
def test_a_stop_is_booked_at_the_open_on_a_gap_else_at_the_bars_worst_price(side, trigger, rested, want):
    assert exit_price("stop", side, trigger, BAR, rested) == want


def test_a_target_alone_is_booked_at_its_trigger_never_better():
    assert exit_price("target", 1, 61_800.0, BAR, True) == ("target", 61_800.0)
    assert exit_price("target", 1, 59_000.0, BAR, True) == ("target", 59_000.0)  # the open had passed it


@pytest.mark.parametrize("side, target, stop, want", [
    (1, 61_800.0, 58_800.0, ("stop", 57_000.0)),  # both inside the bar: adverse first
    (1, 59_500.0, 58_800.0, ("target", 59_500.0)),  # the open was past the target: it fills first
    (-1, 58_200.0, 61_200.0, ("stop", 62_000.0)),
    (-1, 60_500.0, 61_200.0, ("target", 60_500.0)),
    (1, 61_800.0, 56_000.0, ("target", 61_800.0)),  # the stop stayed outside the bar
])
def test_with_the_target_and_the_stop_both_inside_one_bar_the_stop_comes_first(side, target, stop, want):
    assert exit_price("target", side, target, BAR, True, stop=stop) == want


def test_stop_slippage_is_the_larger_of_half_the_spread_and_the_floor():
    assert stop_slippage(0.0) == STOP_SLIPPAGE_FLOOR == 0.0005
    assert stop_slippage(0.0001) == 0.0005
    assert stop_slippage(0.001) == 0.001
