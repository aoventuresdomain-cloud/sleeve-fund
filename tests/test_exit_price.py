"""P1-D13: where a resting stop is booked when only the bar is known (sleeve_fund.instruments.exit_price), and the
taker slippage every backtest stop and target pays (taker_slippage, target_fill_px)."""

from decimal import Decimal

import pytest

from sleeve_fund.instruments import TAKER_SLIPPAGE_FLOOR, exit_price, taker_slippage, target_fill_px

BAR = (60_000.0, 62_000.0, 57_000.0, 59_000.0)  # open, high, low, close


@pytest.mark.parametrize("side, trigger, rested, want", [
    (1, 58_800.0, True, 57_000.0),  # traded through inside the bar: the low
    (1, 61_000.0, True, 60_000.0),  # the open was already through it: the open
    (1, 61_000.0, False, 57_000.0),  # placed inside the bar (its entry filled there): it can't have gapped
    (-1, 61_200.0, True, 62_000.0),  # a short's stop: the high
    (-1, 59_000.0, True, 60_000.0),  # a short's stop gapped through at the open
])
def test_a_stop_is_booked_at_the_open_on_a_gap_else_at_the_bars_worst_price(side, trigger, rested, want):
    assert exit_price(side, trigger, BAR, rested) == want


def test_taker_slippage_is_the_larger_of_half_the_spread_and_the_floor():
    assert taker_slippage(0.0) == TAKER_SLIPPAGE_FLOOR == Decimal("0.0005")
    assert taker_slippage(0.0001) == Decimal("0.0005")
    assert taker_slippage(0.001) == Decimal("0.001")


def test_a_target_books_at_its_level_less_the_taker_slippage():
    assert target_fill_px(61_800, True, 0.001) == Decimal("61738.2")
    assert target_fill_px(58_200, False, 0.0) == Decimal("58229.1")
