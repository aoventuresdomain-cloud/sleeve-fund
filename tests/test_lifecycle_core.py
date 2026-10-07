"""P2-6 lifecycle pure core (sleeve_fund.lifecycle) and the day-0 money helpers it uses (sleeve_fund.money).
Cells follow pe1-plans/phase2-pe1-plans.md section 4; the wiring cells (L3, L6's stale data, L7, L11, L12) land with
the wiring."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from sleeve_fund import lifecycle as lc
from sleeve_fund.money import money, scale

T0 = datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc)
H = timedelta(hours=1)
D = timedelta(days=1)


# --- money (day-0 note v4) ----------------------------------------------------------------------------------------


def test_float_money_is_refused():
    with pytest.raises(TypeError):
        money(0.1)
    with pytest.raises(TypeError):
        money(True)
    assert money("0.1") == Decimal("0.1") and money(3) == Decimal(3)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_money_is_refused(bad):
    with pytest.raises(ValueError):
        money(bad)


def test_scale_takes_the_ratio_as_written():
    assert scale(Decimal("100"), 0.1) == Decimal("10.0")
    with pytest.raises(ValueError):
        scale(Decimal("1"), float("nan"))
    with pytest.raises(TypeError):
        scale(1.0, 0.5)


# --- L1 / L9 / L10: moves -----------------------------------------------------------------------------------------


def test_l1_only_an_active_strategy_can_open():
    assert lc.can_open(lc.ACTIVE)
    assert not lc.can_open(lc.WINDING_DOWN)
    assert not lc.can_open(lc.RETIRED)


def test_allowed_moves():
    assert lc.move(lc.ACTIVE, lc.WINDING_DOWN, "pm", flat=False) == lc.WINDING_DOWN
    assert lc.move(lc.WINDING_DOWN, lc.RETIRED, "pm", flat=True) == lc.RETIRED
    assert lc.move(lc.ACTIVE, lc.RETIRED, "pm", flat=True) == lc.RETIRED


@pytest.mark.parametrize("current,to", [(lc.WINDING_DOWN, lc.ACTIVE), (lc.RETIRED, lc.ACTIVE),
                                        (lc.RETIRED, lc.WINDING_DOWN), (lc.ACTIVE, lc.ACTIVE)])
def test_refused_moves_say_why(current, to):
    with pytest.raises(lc.LifecycleError) as e:
        lc.move(current, to, "pm", flat=True)
    assert e.value.reason


def test_l8_never_retired_while_holding():
    for current in (lc.ACTIVE, lc.WINDING_DOWN):
        with pytest.raises(lc.LifecycleError, match="close it now"):
            lc.move(current, lc.RETIRED, "pm", flat=False)


@pytest.mark.parametrize("actor", ["supervisor", "system", "PM", "", None])
def test_l9_only_the_pm_moves_a_strategy(actor):
    with pytest.raises(lc.NotPM):
        lc.move(lc.ACTIVE, lc.WINDING_DOWN, actor, flat=True)
    with pytest.raises(lc.NotPM):
        lc.close_now(lc.ACTIVE, actor, T0, flat=False)


def test_l10_close_now_flattens_then_retires():
    for current in (lc.ACTIVE, lc.WINDING_DOWN):
        m = lc.close_now(current, "pm", T0, flat=False)
        assert m == lc.Move(lc.WINDING_DOWN, T0)
        assert lc.due(m.lifecycle, T0, m.deadline, T0, flat=False, noticed=False) == lc.FLATTEN
        assert lc.due(m.lifecycle, T0, m.deadline, T0 + H, flat=True, noticed=False) == lc.RETIRE
    assert lc.close_now(lc.ACTIVE, "pm", T0, flat=True) == lc.Move(lc.RETIRED, None)
    with pytest.raises(lc.LifecycleError):
        lc.close_now(lc.RETIRED, "pm", T0, flat=False)


# --- L4: deadline ---------------------------------------------------------------------------------------------------


def test_l4_deadline_is_twice_the_larger_p95():
    bt = [timedelta(days=i % 5 + 1) for i in range(40)]  # p95 5 days
    paper = [timedelta(days=3)] * 10
    assert lc.deadline(T0, bt, paper) == T0 + 10 * D
    assert lc.deadline(T0, bt, [timedelta(days=7)]) == T0 + 14 * D  # paper's p95 is longer: it wins


def test_l4_p95_is_nearest_rank():
    bt = [H] * 19 + [10 * D]  # 20 trades: the 19th is the p95, so the one long outlier doesn't set it
    assert lc.window(bt, []) == lc.WINDOW_FLOOR
    assert lc.window([H] * 18 + [2 * D, 10 * D], []) == 4 * D


def test_l4_fallback_is_twice_the_longest_hold_seen():
    assert lc.window([2 * D] * 19, [4 * D]) == 8 * D
    assert lc.window([], [5 * D]) == 10 * D


def test_l4_floor_and_cap():
    assert lc.window([], []) == lc.WINDOW_FLOOR == 3 * D
    assert lc.window([H] * 30, []) == 3 * D
    assert lc.window([20 * D] * 30, []) == lc.WINDOW_CAP == 30 * D


def test_deadline_needs_utc():
    with pytest.raises(ValueError):
        lc.deadline(datetime(2026, 10, 7), [], [])
    with pytest.raises(ValueError):
        lc.window([-H], [])


# --- L5 / L6 / L8: due ----------------------------------------------------------------------------------------------


def test_l5_notice_once_at_75_percent():
    end = T0 + 4 * D
    assert lc.due(lc.WINDING_DOWN, T0, end, T0 + 2 * D, flat=False, noticed=False) == lc.NONE
    assert lc.due(lc.WINDING_DOWN, T0, end, T0 + 3 * D - H, flat=False, noticed=False) == lc.NONE
    assert lc.due(lc.WINDING_DOWN, T0, end, T0 + 3 * D, flat=False, noticed=False) == lc.NOTICE
    assert lc.due(lc.WINDING_DOWN, T0, end, T0 + 3 * D, flat=False, noticed=True) == lc.NONE


def test_l6_flatten_from_the_deadline_every_bar_until_flat():
    end = T0 + 4 * D
    assert lc.due(lc.WINDING_DOWN, T0, end, end - H, flat=False, noticed=True) == lc.NONE
    for t in (end, end + H, end + 2 * H):  # each bar close after a failed flatten asks again: the retry
        assert lc.due(lc.WINDING_DOWN, T0, end, t, flat=False, noticed=True) == lc.FLATTEN
    assert lc.due(lc.WINDING_DOWN, T0, end, end, flat=False, noticed=False) == lc.FLATTEN  # no notice first


def test_l8_retire_only_when_flat():
    end = T0 + 4 * D
    assert lc.due(lc.WINDING_DOWN, T0, end, T0 + H, flat=True, noticed=False) == lc.RETIRE
    assert lc.due(lc.WINDING_DOWN, T0, end, end + D, flat=False, noticed=True) != lc.RETIRE


def test_due_is_none_unless_winding_down():
    for s in (lc.ACTIVE, lc.RETIRED):
        assert lc.due(s, None, None, T0, flat=False, noticed=False) == lc.NONE
    with pytest.raises(ValueError):
        lc.due(lc.WINDING_DOWN, None, None, T0, flat=False, noticed=False)
    with pytest.raises(ValueError):
        lc.due(lc.WINDING_DOWN, T0, T0 - H, T0, flat=False, noticed=False)


# --- L2: the wind-down stop -----------------------------------------------------------------------------------------


def test_l2_own_stop_atr_from_the_price_at_wind_down_start():
    s = lc.wind_down_stop(1, Decimal("100"), Decimal("2"), stop_atr=1.5)
    assert (s.level, s.distance, s.clamped) == (Decimal("97.0"), Decimal("3.0"), False)
    assert "its own stop" in s.reason
    s = lc.wind_down_stop(-1, "100", "2", stop_atr=1.5)
    assert s.level == Decimal("103.0")


def test_l2_fallback_is_three_atrs():
    s = lc.wind_down_stop(1, Decimal("100"), Decimal("2"))
    assert (s.level, s.atrs) == (Decimal("94.0"), 3.0)
    assert "fallback" in s.reason


def test_l2_never_beyond_half_way_to_liquidation():
    s = lc.wind_down_stop(1, Decimal("100"), Decimal("2"), liquidation=Decimal("90"))  # 6 asked, 5 allowed
    assert (s.level, s.distance, s.clamped) == (Decimal("95"), Decimal("5"), True)
    assert "half the distance to liquidation" in s.reason
    s = lc.wind_down_stop(-1, Decimal("100"), Decimal("2"), liquidation=Decimal("108"))
    assert (s.level, s.clamped) == (Decimal("104"), True)
    s = lc.wind_down_stop(1, Decimal("100"), Decimal("1"), liquidation=Decimal("90"))  # 3 is inside 5
    assert (s.level, s.clamped) == (Decimal("97.0"), False)


def test_l2_refusals():
    with pytest.raises(TypeError):
        lc.wind_down_stop(1, 100.0, Decimal("2"))  # float money
    with pytest.raises(TypeError):
        lc.wind_down_stop(1, Decimal("100"), Decimal("2"), liquidation=90.0)
    with pytest.raises(ValueError):
        lc.wind_down_stop(1, Decimal("100"), Decimal("2"), liquidation=Decimal("110"))  # wrong side
    with pytest.raises(ValueError):
        lc.wind_down_stop(0, Decimal("100"), Decimal("2"))
    with pytest.raises(ValueError):
        lc.wind_down_stop(1, Decimal("100"), Decimal("2"), stop_atr=0)
    with pytest.raises(ValueError):
        lc.wind_down_stop(1, Decimal("100"), Decimal("40"))  # 120 below 100
