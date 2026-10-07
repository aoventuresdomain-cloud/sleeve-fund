"""The pure portfolio rules that P2-1a moves onto main with the sizing core: the monthly rebalance date and the P2-1b
weight-to-entry conversion (Advisor's rulings, 6 Oct 17:09). Ported unchanged from #165's test_central_sizing.py and
test_p2_1b.py; the engine wiring that uses them is P2-1b."""

from datetime import datetime, timezone
from decimal import Decimal as D

import pytest

from sleeve_fund.portfolio.allocation import next_rebalance
from sleeve_fund.portfolio.conversion import band_order, band_target, trades_from_weights


@pytest.mark.parametrize("after, due", [("2026-10-06T16:00", "2026-11-02"), ("2026-05-31T23:59", "2026-06-01"),
                                        ("2026-09-30T00:00", "2026-10-05"), ("2026-12-08T00:00", "2027-01-04"),
                                        ("2026-11-02T00:00", "2026-12-07")])
def test_the_monthly_rebalance_is_the_first_monday_at_midnight_utc(after, due):
    t = datetime.fromisoformat(after).replace(tzinfo=timezone.utc)
    assert next_rebalance(t) == datetime.fromisoformat(due).replace(tzinfo=timezone.utc)


def test_the_rebalance_needs_a_time_zone():
    with pytest.raises(ValueError):
        next_rebalance(datetime(2026, 10, 6))


def test_a_sign_change_in_the_weight_is_an_exit_then_an_entry():
    w = [0.0, 0.5, 0.8, -0.3, -0.6, 0.0, 0.0]
    assert trades_from_weights(w) == [(1, "entry", 1), (3, "exit", 1), (3, "entry", -1), (5, "exit", -1)]


def test_after_a_stop_out_re_entry_waits_for_the_weight_to_go_to_zero_and_back():
    w = [0.0, 0.5, 0.5, 0.6, 0.5, 0.0, 0.4, 0.4]
    assert trades_from_weights(w, stopped={2}) == [(1, "entry", 1), (2, "exit", 1), (6, "entry", 1)]


def test_the_band_target_is_central_sizing_times_weight_over_full_weight():
    assert band_target(D("2.000"), 0.3, 0.6) == D("1.000")
    assert band_target(D("2.000"), 0.6, 0.6) == D("2.000")


@pytest.mark.parametrize("target, order", [(D("1.250"), D(0)), (D("1.251"), D("0.251")),
                                           (D("0.750"), D(0)), (D("0.749"), D("-0.251"))])
def test_the_band_trades_only_past_25pct_of_the_size_held(target, order):
    assert band_order(target, D("1.000"), D("0.25")) == order
