"""Central sizing, the pure core (v2 P2-1a): each limit binding, the venue minimum rounded up or skipped, the overlays,
against hand calculations; Decimal money; and the Independent Quant Advisor's three guarantees (MUST FIX, 7 Oct 17:03).

Ported from #165's tests/test_sizing.py. Inputs are now Decimal money, per the frozen day-0 interface note v4; every
expected value is unchanged."""

import ast
import dataclasses
import inspect
from decimal import ROUND_DOWN, Decimal

import pytest

from sleeve_fund.money import money, scale
from sleeve_fund.portfolio import sizing
from sleeve_fund.portfolio.sizing import (
    ATR_STOP_MULTIPLE,
    DEFAULT_STOP_SLIPPAGE,
    SizingInputs,
    loss_at_stop,
    margin_per_unit,
    risk_per_unit,
    rounds_up_too_often,
    size_entry,
)

D = Decimal


def _in(**kw) -> SizingInputs:
    base = dict(allocated_equity=D(10_000), price=D(100), side=1, leg_cost=0.001, taker_fee=D("0.001"), half_spread=0.0, risk_per_trade=0.01,
                position_cap_pct=0.5, lot=D("0.01"), min_qty=D("0.01"), stop_frac=0.02, stop_slippage=0.0,
                vol_floor=0.0)  # no slippage and no floor unless a test sets them, so each limit is hand-checkable
    return SizingInputs(**{**base, **kw})


# --- each limit, by hand ----------------------------------------------------------------------------------------

def test_risk_per_trade_sets_the_size_when_no_cap_binds():
    s = size_entry(_in())
    loss = 0.02 + 0.001 + 0.98 * 0.001  # stop, buy leg, sell leg at the stop price
    assert s.sized_by == "risk per trade" and s.qty == D("45.49")  # 100 / 0.02198 = 4,549.59 of notional
    assert s.risk_budget == D(100) and float(s.risk_amount) == pytest.approx(45.49 * 100 * loss)


def test_the_margin_cap_binds_on_a_perp_and_notional_is_margin_times_leverage():
    s = size_entry(_in(perp=True, leverage=2.0, position_cap_pct=0.2))
    assert s.sized_by == "margin cap" and s.qty == D("40.00")  # margin 2,000, notional 4,000


def test_the_leverage_cap_binds_on_a_tight_stop():
    # SZ-LEV-FEE (Advisor 21:53 UK): notional = equity / (1/L + 2 x taker), computed here from the formula.
    s = size_entry(_in(perp=True, leverage=2.0, position_cap_pct=1.0, stop_frac=0.002))
    expected = (D(10_000) / (1 / D(2) + 2 * D("0.001")) / D(100)).quantize(D("0.01"), rounding=ROUND_DOWN)
    assert s.sized_by == "2x leverage cap" and s.qty == expected


@pytest.mark.parametrize("lev", [1.0, 1.5, 2.0, 3.0, 5.0, 20.0])
@pytest.mark.parametrize("taker", ["0", "0.0002", "0.0005", "0.001", "0.0026"])
@pytest.mark.parametrize("side", [1, -1])
def test_at_the_leverage_cap_margin_and_both_taker_fees_fit_the_equity(lev, taker, side):
    """The finding: equity x L x (1 - cost) let margin + fees pass the balance at a 100% cap. Now they never do."""
    s = size_entry(_in(side=side, perp=True, leverage=lev, position_cap_pct=1.0, stop_frac=0.001, risk_per_trade=0.9,
                       taker_fee=D(taker)))
    assert s.sized_by in (f"{lev:g}x leverage cap", "margin cap")  # a tie at a zero rate names the margin cap first
    notional = s.qty * D(100)
    assert notional / D(repr(lev)) + 2 * D(taker) * notional <= D(10_000)
    assert s.limits[f"{lev:g}x leverage cap"] == D(10_000) / (1 / D(repr(lev)) + 2 * D(taker))


@pytest.mark.parametrize("taker", [None, 0.001, "0.001", D("NaN"), D("-0.001")])
def test_a_perp_entry_without_a_known_taker_rate_is_refused(taker):
    """Fail closed (Advisor 21:53 UK): the rate comes from the fill model's fee config; missing or not a Decimal refuses."""
    s = size_entry(_in(perp=True, leverage=2.0, taker_fee=taker))
    assert not s.ok and s.qty == 0 and "taker fee rate" in s.skipped


def test_a_spot_entry_needs_no_taker_rate():
    assert size_entry(_in(taker_fee=None)) == size_entry(_in())


def test_the_spot_position_cap_binds():
    s = size_entry(_in(position_cap_pct=0.33))
    assert s.sized_by == "position cap" and s.qty == D("33.00")


def test_the_largest_order_and_volume_caps_bind():
    assert size_entry(_in(max_notional=D(1_000))).qty == D("10.00")
    s = size_entry(_in(max_notional=D(1_000), volume_notional=D(500)))
    assert s.sized_by == "share of the bar's volume" and s.qty == D("5.00")


def test_the_venue_minimum_is_rounded_up_while_its_risk_stays_within_one_and_a_half_budgets():
    s = size_entry(_in(allocated_equity=D(1_000), lot=D(1), min_qty=D(1), stop_frac=0.1))
    assert s.rounded_up and s.qty == D(1) and s.sized_by.startswith("venue minimum")
    assert float(s.risk_amount) == pytest.approx(100 * loss_at_stop(0.1, 0.001)) and s.risk_amount <= 15


def test_the_venue_minimum_is_skipped_when_its_risk_is_too_far_over_budget():
    s = size_entry(_in(allocated_equity=D(1_000), lot=D(1), min_qty=D(1), stop_frac=0.2))
    assert not s.ok and s.qty == 0 and "over 1.5x the 10.00 budget" in s.skipped


def test_the_venue_minimum_is_skipped_when_it_would_break_a_cap():
    s = size_entry(_in(allocated_equity=D(1_000), lot=D(1), min_qty=D(1), stop_frac=0.1, max_notional=D(90)))
    assert not s.ok and "largest order cap" in s.skipped


def test_a_strategy_rounding_up_on_more_than_a_fifth_of_its_entries_is_flagged():
    assert not rounds_up_too_often(10, 2) and rounds_up_too_often(10, 3) and not rounds_up_too_often(0, 0)


def test_shorts_can_carry_half_the_risk_of_longs():
    """B1: everything else equal, a short at 0.5% risks half what a long at 1% does."""
    long_ = size_entry(_in(risk_long=0.01, risk_short=0.005))
    short = size_entry(_in(side=-1, risk_long=0.01, risk_short=0.005))
    assert short.sized_by == "risk per trade, short" and long_.sized_by == "risk per trade, long"
    assert short.risk_budget == long_.risk_budget / 2
    assert float(short.risk_amount) == pytest.approx(float(short.risk_budget), rel=1e-3)


def test_a_fraction_of_full_size_scales_the_budget_before_the_caps():
    """A8: 2 of 3 signals gives 2/3 of full size."""
    full, part = size_entry(_in()), size_entry(_in(fraction=2 / 3))
    assert float(part.risk_budget) == pytest.approx(float(full.risk_budget) * 2 / 3)
    assert part.qty == D("30.33")  # 4,549.59 x 2/3 = 3,033.06 of notional


def test_the_regime_weight_scales_the_budget():
    assert size_entry(_in(regime_weight=0.5)).risk_budget == D(50)
    with pytest.raises(ValueError, match="regime weight"):
        size_entry(_in(regime_weight=0.0))


def test_volatility_targeting_matches_a_hand_calculation():
    """B2: notional = target vol x allocated equity / instrument vol = 0.01 x 10,000 / 0.02 = 5,000."""
    s = size_entry(_in(overlay="vol_target", vol_target=0.01, instrument_vol=0.02, perp=True, leverage=3.0,
                       position_cap_pct=0.2, risk_per_trade=0.5))
    assert s.sized_by == "volatility target" and s.qty == D("50.00")


def test_a_vol_sized_perp_entry_never_exceeds_the_margin_cap():
    """Regression for QA round 13: the vol target is on allocated equity, and the margin cap binds after it."""
    s = size_entry(_in(overlay="vol_target", vol_target=0.05, instrument_vol=0.005, perp=True, leverage=3.0,
                       position_cap_pct=0.2, risk_per_trade=0.5))
    assert s.sized_by == "margin cap" and s.qty * 100 <= D(6_000)


def test_no_stop_falls_back_to_atr_and_without_atr_there_is_no_sizing():
    s = size_entry(_in(stop_frac=None, atr=D(2)))
    assert s.stop_frac == pytest.approx(ATR_STOP_MULTIPLE * 2.0 / 100) and "ATR(14)" in s.sized_by
    none = size_entry(_in(stop_frac=None))
    assert not none.ok and "no sizing without a stop" in none.skipped


@pytest.mark.parametrize("bad", [dict(side=0), dict(overlay="kelly"), dict(fraction=1.5), dict(price=D(0)),
                                 dict(leverage=2.0), dict(lot=D(0)), dict(risk_per_trade=-0.01)])
def test_bad_inputs_are_refused(bad):
    with pytest.raises(ValueError):
        size_entry(_in(**bad))


def test_no_allocated_equity_skips():
    assert "no allocated equity" in size_entry(_in(allocated_equity=D(0))).skipped


@pytest.mark.parametrize("half_spread, slip", [(0.0002, 0.0005), (0.0008, 0.0008)])
def test_a_stop_fills_past_its_price_by_the_larger_of_half_the_spread_and_five_basis_points(half_spread, slip):
    """Advisor, 6 Oct 16:45 (D13): the loss per unit carries a named stop slippage, by default the larger of half the
    spread and 0.05%; a venue's own setting replaces it."""
    plain = size_entry(_in(leg_cost=0.0015, half_spread=half_spread, stop_slippage=None))
    loss = loss_at_stop(0.02, 0.0015) + 0.98 * slip
    assert plain.qty == D(str(int(100 / loss / 100 * 100) / 100))  # 100 of risk over loss per unit, at 100
    assert size_entry(_in(leg_cost=0.0015, half_spread=half_spread, stop_slippage=0.0)).qty > plain.qty


def test_no_volatility_or_no_floor_skips_a_vol_sized_entry():
    """Advisor, 16:45: a missing volatility, or a missing floor, skips the entry rather than sizing on a guess."""
    for kw in ({"instrument_vol": None}, {"instrument_vol": 0.02, "vol_floor": None}):
        s = size_entry(_in(overlay="vol_target", vol_target=0.01, **kw))
        assert not s.ok and "volatility" in s.skipped
    for kw in ({"instrument_vol": float("nan")}, {"instrument_vol": 0.02, "vol_floor": float("nan")}):
        with pytest.raises(ValueError, match="finite"):  # NaN is refused at the boundary (DA rule 4)
            _in(overlay="vol_target", vol_target=0.01, **kw)


@pytest.mark.parametrize("stop", [0.0, -0.01])
def test_a_declared_stop_of_zero_is_refused_and_never_falls_back_to_the_atr(stop):
    """Advisor, 16:45: never fall back silently; the ATR stop is only for a definition that declares none."""
    with pytest.raises(ValueError, match="a declared stop must be above 0"):
        size_entry(_in(stop_frac=stop, atr=D(2)))


def test_with_a_stop_and_a_volatility_target_the_smaller_size_wins():
    """Advisor, 6 Oct 2026: both are worked out, never multiplied."""
    s = size_entry(_in(overlay="vol_target", vol_target=0.01, instrument_vol=0.005))  # 20,000 by volatility
    assert s.sized_by == "risk per trade" and s.qty == D("45.49")
    s = size_entry(_in(overlay="vol_target", vol_target=0.01, instrument_vol=0.04))  # 2,500 by volatility
    assert s.sized_by == "volatility target" and s.qty == D("25.00")


def test_a_quiet_stretch_cannot_size_up_past_the_volatility_floor():
    s = size_entry(_in(overlay="vol_target", vol_target=0.01, instrument_vol=0.001, vol_floor=0.04))
    assert s.sized_by == "volatility target" and s.qty == D("25.00")


def test_a_stop_too_wide_for_the_leverage_is_refused_never_moved_or_shrunk_to_fit():
    """The PM's rule on isolated margin (Q199-1): at 3x the price is ~33% from liquidation whatever the size, so a 20%
    stop is past half of it. Shrinking the notional can't help: the entry is refused, naming liquidation."""
    s = size_entry(_in(perp=True, leverage=3.0, position_cap_pct=1.0, stop_frac=0.2, risk_per_trade=0.9,
                       stop_to_liquidation=0.5, maintenance_margin=0.005))
    assert s.qty == 0 and not s.ok and "liquidation" in s.skipped and s.stop_frac == 0.2

# --- Decimal money (day-0 interface note v4; DA rules) -------------------------------------------------------------

@pytest.mark.parametrize("field", ["allocated_equity", "price", "lot", "min_qty", "atr", "max_notional",
                                   "volume_notional"])
def test_float_money_is_refused_with_a_type_error(field):
    with pytest.raises(TypeError, match="money"):
        _in(**{field: 1.0})


def test_int_and_string_money_are_taken_exactly_and_the_quantity_is_decimal():
    s = size_entry(_in(allocated_equity=10_000, price="100"))
    assert s.qty == D("45.49") and type(s.qty) is Decimal
    assert type(s.risk_budget) is Decimal and type(s.risk_amount) is Decimal
    assert all(type(v) is Decimal for v in s.limits.values())


@pytest.mark.parametrize("bad", [D("NaN"), D("Infinity"), "NaN"])
def test_nan_and_infinite_money_are_refused(bad):
    with pytest.raises(ValueError, match="finite"):
        _in(price=bad)


def test_a_ratio_meets_money_only_through_scale_and_exactly():
    assert scale(D("10000"), 0.01) == D("100.00")
    assert scale(D("3"), 0.1) == D("0.3")  # never 0.3000000000000000166...
    with pytest.raises(TypeError):
        scale(10_000.0, 0.01)
    with pytest.raises(TypeError):
        money(True)


def test_the_result_is_frozen():
    s = size_entry(_in())
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.qty = D(1)
    with pytest.raises(TypeError):
        s.limits["position cap"] = D(0)


def test_the_inputs_keep_the_frozen_interface():
    """Day-0 interface note v4 (HoE, 7 Oct 17:50): a change needs the HoE's OK."""
    assert [f.name for f in dataclasses.fields(SizingInputs)] == [
        "allocated_equity", "price", "side", "lot", "min_qty", "leg_cost", "half_spread", "risk_per_trade",
        "position_cap_pct", "leverage", "perp", "maintenance_margin", "stop_frac", "atr", "stop_slippage",
        "regime_weight", "fraction", "overlay", "vol_target", "instrument_vol", "vol_floor", "max_notional",
        "volume_notional", "stop_to_liquidation", "risk_long", "risk_short", "taker_fee"]
    assert [f.name for f in dataclasses.fields(sizing.Sizing)] == [
        "qty", "sized_by", "stop_frac", "risk_budget", "risk_amount", "limits", "rounded_up", "skipped"]


def test_the_core_reads_no_store_database_runner_or_strategy():
    tree = ast.parse(inspect.getsource(sizing))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    banned = ("store", "db", "runner", "strategies", "paper", "dashboard", "sqlalchemy", "psycopg", "nautilus")
    assert not [m for m in names if any(b in m.split(".") or m.startswith(b) for b in banned)], names


def test_importing_the_core_loads_no_engine_or_store():
    """margin.py carries the engine's liquidation arithmetic with no imports, so the core stays pure (Q199-1)."""
    import subprocess
    import sys

    code = ("import sys, sleeve_fund.portfolio.sizing; "
            "print([m for m in sys.modules if m.startswith(('nautilus', 'sqlalchemy', 'sleeve_fund.store', "
            "'sleeve_fund.strategies', 'sleeve_fund.paper'))])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


# --- the Advisor's three guarantees (MUST FIX, 7 Oct 17:03) --------------------------------------------------------

GRID = [dict(perp=p, leverage=lev, stop_frac=st, atr=atr, overlay=ov, vol_target=0.02, instrument_vol=0.01,
             position_cap_pct=1.0, risk_per_trade=0.05, maintenance_margin=0.005)
        for p, lev in ((False, 1.0), (True, 1.0), (True, 2.0), (True, 5.0), (True, 10.0))
        for st in (None, 0.005, 0.05, 0.3)
        for atr in (None, D(0), D(3))
        for ov in ("stop", "vol_target")]


@pytest.mark.parametrize("kw", GRID)
def test_guarantee_1_no_entry_is_sized_without_a_stop_so_none_exceeds_the_stopless_1x(kw):
    """A strategy with no stop is capped at 1x: central sizing never sizes one at all. With no declared stop and no
    ATR to set the fallback from, the entry is skipped at every leverage; whatever is sized carries a stop."""
    s = size_entry(_in(**kw))
    if s.stop_frac is None:
        assert s.qty == 0 and "no sizing without a stop" in s.skipped
    if s.ok:
        assert s.stop_frac is not None and s.stop_frac > 0


def test_guarantee_1_the_fallback_stop_is_a_real_stop_with_d13_slippage():
    """The fallback (2.5 x Wilder ATR) is returned as the stop to place, and the size carries its stop slippage."""
    s = size_entry(_in(stop_frac=None, atr=D(2), half_spread=0.0002, stop_slippage=None))
    assert s.stop_frac == pytest.approx(0.05)
    loss = loss_at_stop(0.05, 0.001) + 0.95 * DEFAULT_STOP_SLIPPAGE
    assert float(s.risk_amount) == pytest.approx(float(s.qty) * 100 * loss)


@pytest.mark.parametrize("atr_pct, move", [(0.01, 0.10), (0.02, 0.10), (0.05, 0.15), (0.10, 0.30)])
def test_guarantee_2_stopless_open_risk_counts_at_notional_times_max_of_10pct_and_3_daily_atrs(atr_pct, move):
    stopless = sizing.Sizing(D(0), "", None, D(0), D(0))
    i = _in(price=D("250"))
    got = risk_per_unit(stopless, i, atr_pct)
    assert got == scale(D("250"), max(0.10, 3 * atr_pct))  # open_risk's one formula, exactly
    assert float(got) == pytest.approx(250 * move)
    with pytest.raises(ValueError, match="ATR isn't known"):
        risk_per_unit(stopless, i, None)


@pytest.mark.parametrize("side", [1, -1])
def test_guarantee_2_a_stopped_entry_counts_as_open_risk_counts_the_position(side):
    """Note v4 section 2: the gate's per-unit risk is open_risk.position_risk's, the distance to the stop, so an entry
    counts as its position will once held. Sizing's costs and slippage stay in its own risk amount, above it."""
    from sleeve_fund.open_risk import position_risk

    i = _in(side=side, perp=True, leverage=3.0, stop_frac=0.03, half_spread=0.0004, stop_slippage=None)
    s = size_entry(i)
    held = position_risk(side * float(s.qty), 100.0, 100 * (1 - side * 0.03), 0.02)
    assert float(risk_per_unit(s, i) * s.qty) == pytest.approx(held, abs=1e-9)
    assert risk_per_unit(s, i) * s.qty < s.risk_amount
    assert margin_per_unit(i) == D(100) / 3 and margin_per_unit(_in()) == D(100)


@pytest.mark.parametrize("lev", [1.0, 2.0, 3.0, 5.0, 10.0, 20.0])
@pytest.mark.parametrize("stop", [0.005, 0.02, 0.05, 0.1, 0.25, 0.45])
@pytest.mark.parametrize("cap", [None, 0.5, 0.3])
@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("mm", [0.005, 0.05])
def test_guarantee_3_every_perp_stop_is_within_half_the_engines_distance_to_liquidation(lev, stop, cap, side, mm):
    """Measured as the engine liquidates (margin.entry_liquidation, which strategies.base uses: isolated margin at the leverage, the
    venue's maintenance margin, the fee) at the size taken: within the share, or refused naming liquidation."""
    from sleeve_fund.strategies.base import entry_liquidation

    s = size_entry(_in(side=side, perp=True, leverage=lev, stop_frac=stop, position_cap_pct=1.0, risk_per_trade=0.5,
                       maintenance_margin=mm, stop_to_liquidation=cap, leg_cost=0.001))
    if not s.ok:
        assert "liquidation" in s.skipped and s.stop_frac == stop
        _, dist = entry_liquidation(10_000.0, 50.0, 100.0, side, 0.001, mm, lev)
        assert stop > (cap or 0.5) * dist
        return
    _, dist = entry_liquidation(10_000.0, float(s.qty), 100.0, side, 0.001, mm, lev)
    assert stop <= (cap or 0.5) * dist + 1e-12


@pytest.mark.parametrize("lev", [1.0, 1.5, 2.0, 3.0, 5.0, 20.0])
@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("mm", [0.0, 0.005, 0.05])
def test_the_liquidation_distance_is_the_engines_whatever_the_size(lev, side, mm):
    """HoE 20:15 UK: the core calls the engine's own function, not a copy; on isolated margin the size doesn't move it."""
    from sleeve_fund.portfolio import sizing
    from sleeve_fund.strategies import base

    assert sizing.entry_liquidation is base.entry_liquidation
    got = sizing.liquidation_distance(_in(side=side, perp=True, leverage=lev, maintenance_margin=mm, leg_cost=0.001))
    for qty in (0.5, 20.0, 10_000.0 * lev / 100 * 0.95):  # margin + fee within the balance (SZ-LEV-FEE)
        _, dist = base.entry_liquidation(10_000.0, qty, 100.0, side, 0.001, mm, lev)
        assert got == pytest.approx(dist, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("lev, stop", [(3.0, 0.20), (2.0, 0.30), (5.0, 0.10)])
@pytest.mark.parametrize("side", [1, -1])
def test_guarantee_3_pinned_refusals(lev, stop, side):
    """HoE 20:15 UK pins: each stop is past half the way to liquidation at its leverage, so the entry is refused naming
    liquidation, with the stop as declared and nothing sized."""
    s = size_entry(_in(side=side, perp=True, leverage=lev, stop_frac=stop, position_cap_pct=1.0, risk_per_trade=0.9))
    assert not s.ok and s.qty == 0 and s.stop_frac == stop
    assert s.skipped.startswith("entry refused") and "liquidation price" in s.skipped


def test_guarantee_3_the_venues_maintenance_margin_counts():
    """Advisor 20:08 UK (a): the venue's maintenance margin, never a zero approximation. At 3x long the distance is
    (1/3 - mm) / (1 - mm): 33.33% at mm 0, 29.82% at 5%, so a 16% stop fits half of the first and not the second."""
    kw = dict(perp=True, leverage=3.0, position_cap_pct=1.0, stop_frac=0.16, risk_per_trade=0.9)
    assert size_entry(_in(**kw, maintenance_margin=0.0)).ok
    s = size_entry(_in(**kw, maintenance_margin=0.05))
    assert not s.ok and "liquidation" in s.skipped


def test_guarantee_3_never_refuses_a_stopless_entry_the_1x_cap_governs_it():
    """Advisor 20:08 UK (b): the half-way check needs a stop. A stopless entry never reaches it: it is skipped for having
    no stop (guarantee 1), never refused by the liquidation rule."""
    s = size_entry(_in(perp=True, leverage=5.0, stop_frac=None, atr=None))
    assert not s.ok and "no sizing without a stop" in s.skipped and "liquidation" not in s.skipped
    assert not any(k.startswith("liquidation rule") for k in s.limits)


def test_guarantee_3_exactly_half_way_is_allowed_and_past_it_refused():
    # 2x long, no maintenance margin: liquidation is exactly 50% away, so a 25% stop is exactly half way.
    kw = dict(perp=True, leverage=2.0, position_cap_pct=1.0, maintenance_margin=0.0, risk_per_trade=0.9)
    assert size_entry(_in(**kw, stop_frac=0.25)).ok
    assert "liquidation" in size_entry(_in(**kw, stop_frac=0.2501)).skipped
    with pytest.raises(ValueError, match="at most 50% of the way"):
        size_entry(_in(perp=True, leverage=3.0, stop_to_liquidation=0.6))


# --- A8 steps ------------------------------------------------------------------------------------------------------

def test_steps_between_fractions_send_only_the_difference_and_skip_below_the_minimum():
    from sleeve_fund.portfolio.sizing import step_order

    held, sent = D(0), []
    for f in (1 / 3, 2 / 3, 1.0, 0.0):
        o = step_order(D("0.900"), held, f, D("0.001"), D("0.001"))
        sent.append(o.qty)
        held += o.qty
    assert sent == [D("0.300"), D("0.300"), D("0.300"), D("-0.900")]
    small = step_order(D("0.900"), D(0), 1 / 3, D("0.001"), D("0.350"))
    assert small.qty == 0 and "below the venue's smallest order" in small.skipped
    # Closing to 0 always goes in full, however small.
    assert step_order(D("0.900"), D("0.010"), 0.0, D("0.001"), D("0.350")).qty == D("-0.010")
