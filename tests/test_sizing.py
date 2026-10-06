"""Central sizing (v2 P2-1, with A8, B1 and B2): each limit binding, the venue minimum rounded up or skipped,
and the overlays, against hand calculations."""

from decimal import Decimal

import pytest

from sleeve_fund.portfolio.sizing import ATR_STOP_MULTIPLE, SizingInputs, loss_at_stop, size_entry

D = Decimal


def _in(**kw) -> SizingInputs:
    base = dict(allocated_equity=10_000.0, price=100.0, side=1, leg_cost=0.001, risk_per_trade=0.01,
                position_cap_pct=0.5, lot=D("0.01"), min_qty=D("0.01"), stop_frac=0.02)
    return SizingInputs(**{**base, **kw})


def test_risk_per_trade_sets_the_size_when_no_cap_binds():
    s = size_entry(_in())
    loss = 0.02 + 0.001 + 0.98 * 0.001  # stop, buy leg, sell leg at the stop price
    assert s.sized_by == "risk per trade" and s.qty == D("45.49")  # 100 / 0.02198 = 4,549.59 of notional
    assert s.risk_budget == pytest.approx(100.0) and s.risk_amount == pytest.approx(45.49 * 100 * loss)


def test_the_margin_cap_binds_on_a_perp_and_notional_is_margin_times_leverage():
    s = size_entry(_in(perp=True, leverage=2.0, position_cap_pct=0.2))
    assert s.sized_by == "margin cap" and s.qty == D("40.00")  # margin 2,000, notional 4,000


def test_the_leverage_cap_binds_on_a_tight_stop():
    s = size_entry(_in(perp=True, leverage=2.0, position_cap_pct=1.0, stop_frac=0.002))
    assert s.sized_by == "2x leverage cap" and s.qty == D("199.80")  # 10,000 x 2 x (1 - 0.001)


def test_the_spot_position_cap_binds():
    s = size_entry(_in(position_cap_pct=0.33))
    assert s.sized_by == "position cap" and s.qty == D("33.00")


def test_the_largest_order_and_volume_caps_bind():
    assert size_entry(_in(max_notional=1_000.0)).qty == D("10.00")
    s = size_entry(_in(max_notional=1_000.0, volume_notional=500.0))
    assert s.sized_by == "share of the bar's volume" and s.qty == D("5.00")


def test_the_venue_minimum_is_rounded_up_while_its_risk_stays_within_one_and_a_half_budgets():
    s = size_entry(_in(allocated_equity=1_000.0, lot=D(1), min_qty=D(1), stop_frac=0.1))
    assert s.rounded_up and s.qty == D(1) and s.sized_by.startswith("venue minimum")
    assert s.risk_amount == pytest.approx(100 * loss_at_stop(0.1, 0.001)) and s.risk_amount <= 1.5 * s.risk_budget


def test_the_venue_minimum_is_skipped_when_its_risk_is_too_far_over_budget():
    s = size_entry(_in(allocated_equity=1_000.0, lot=D(1), min_qty=D(1), stop_frac=0.2))
    assert not s.ok and s.qty == 0 and "over 1.5x the 10.00 budget" in s.skipped


def test_the_venue_minimum_is_skipped_when_it_would_break_a_cap():
    s = size_entry(_in(allocated_equity=1_000.0, lot=D(1), min_qty=D(1), stop_frac=0.1, max_notional=90.0))
    assert not s.ok and "largest order cap" in s.skipped


def test_shorts_can_carry_half_the_risk_of_longs():
    """B1: everything else equal, a short at 0.5% risks half what a long at 1% does."""
    long_ = size_entry(_in(risk_long=0.01, risk_short=0.005))
    short = size_entry(_in(side=-1, risk_long=0.01, risk_short=0.005))
    assert short.sized_by == "risk per trade, short" and long_.sized_by == "risk per trade, long"
    assert short.risk_budget == pytest.approx(long_.risk_budget / 2)
    assert short.risk_amount == pytest.approx(short.risk_budget, rel=1e-3)


def test_a_fraction_of_full_size_scales_the_budget_before_the_caps():
    """A8: 2 of 3 signals gives 2/3 of full size."""
    full, part = size_entry(_in()), size_entry(_in(fraction=2 / 3))
    assert part.risk_budget == pytest.approx(full.risk_budget * 2 / 3)
    assert part.qty == D("30.33")  # 4,549.59 x 2/3 = 3,033.06 of notional


def test_the_regime_weight_scales_the_budget():
    assert size_entry(_in(regime_weight=0.5)).risk_budget == pytest.approx(50.0)
    with pytest.raises(ValueError, match="regime weight"):
        size_entry(_in(regime_weight=0.0))


def test_volatility_targeting_matches_a_hand_calculation():
    """B2: notional = target vol x allocated equity / instrument vol = 0.01 x 10,000 / 0.02 = 5,000."""
    s = size_entry(_in(overlay="vol_target", vol_target=0.01, instrument_vol=0.02, perp=True, leverage=3.0,
                       position_cap_pct=0.2))
    assert s.sized_by == "volatility target" and s.qty == D("50.00")


def test_a_vol_sized_perp_entry_never_exceeds_the_margin_cap():
    """Regression for QA round 13: the vol target is on allocated equity, and the margin cap binds after it."""
    s = size_entry(_in(overlay="vol_target", vol_target=0.05, instrument_vol=0.005, perp=True, leverage=3.0,
                       position_cap_pct=0.2))
    assert s.sized_by == "margin cap" and float(s.qty) * 100 <= 0.2 * 10_000 * 3


def test_no_stop_falls_back_to_atr_and_without_atr_there_is_no_sizing():
    s = size_entry(_in(stop_frac=None, atr=2.0))
    assert s.stop_frac == pytest.approx(ATR_STOP_MULTIPLE * 2.0 / 100) and "ATR(14)" in s.sized_by
    none = size_entry(_in(stop_frac=None))
    assert not none.ok and "no sizing without a stop" in none.skipped


@pytest.mark.parametrize("bad", [dict(side=0), dict(overlay="kelly"), dict(fraction=1.5), dict(price=0.0),
                                 dict(leverage=2.0), dict(leg_cost=float("nan"))])
def test_bad_inputs_are_refused(bad):
    with pytest.raises(ValueError):
        size_entry(_in(**bad))


def test_no_allocated_equity_skips():
    assert "no allocated equity" in size_entry(_in(allocated_equity=0.0)).skipped
