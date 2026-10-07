"""P2-1a pure sizing core (Quant Developer, phase 2 wave 2.2; PM yes on #165 18:51 UK, build approved 19:54 UK 7 Oct):
strict xfails written by QA BEFORE the build. They are the item's Done-when. The engineer makes each pass and lifts
its mark (marks-only) before handing over. Master: quant-review/v2-p2/test_p2_1a_xfails.py (Head of QA).

Rebuilt from the 6 Oct P2-1 master (quant-review/p2-1-xfails, md5 0c09f93f): its pure-core cells are carried over with
their hand calculations unchanged, moved onto the FROZEN day-0 interface note v4 (qd/day0-interface-note.md, HoE 17:50
UK). Not carried over, on purpose:
- the opt-in cells [OPT]: superseded by the Advisor's 17:03 UK 7 Oct "direct cut-over is OK" (advisor-rulings.md);
  the golden-baseline diff replaces them (see README: the diff must show only expected sizing changes);
- the rebalance, allocations and DA-6 reset cells: they are P2-1b wiring (quant-review/p2-1b-xfails + P2-1b master).

Sources: as the P2-1 master ([S30], [S39], [A8], [B1], [B2], [R-...] Advisor 6 Oct 16:45), plus
- [V4] day-0 interface note v4: money and quantity are Decimal; ratios float; a float passed for a money field is
  refused with TypeError, int/str/Decimal accepted; NaN/Inf refused at every dataclass boundary; quantities round DOWN
  to the lot; money never rounded inside a core; intent_for uses open_risk.position_risk so sizing and gate agree.
- [ISO] the engine's liquidation is on ISOLATED margin: margin = notional / the profile leverage
  (markets.isolated_margin), liquidation via strategies.base.entry_liquidation; main refuses an entry whose stop is past
  the share of that distance (base.py ~2010). HoQA correction 7 Oct 20:40 UK (HoE OK + Advisor confirmed 20:15 UK): the 6 Oct R-LIQ cell
  and the first G3 sweep measured it on the whole allocated equity (cross margin), which is not how the engine liquidates.
- [ADV-2008a] Advisor 20:08 UK: the venue's maintenance margin is used; a stopless entry is not refused by this check
  (the 1x cap governs); exactly 50% allowed, above refused.
- [G1-G3] Advisor MUST FIX, 17:03 UK 7 Oct (advisor-rulings.md:275): deleting _cap_pct keeps (G1) a stopless Model is
  capped at 1x; (G2) open risk counts stopless positions at notional x max(10%, 3 daily ATR); (G3) the stop sits no
  further than half the distance to liquidation. [G4] note v4: the fallback stop is a REAL stop with D13's slippage.

ASSUMED INTERFACES (adapt names, never assertions): note v4 section 1.
- sleeve_fund.portfolio.sizing: SizingInputs, Sizing (qty, sized_by, stop_frac, risk_budget, risk_amount, limits,
  rounded_up, skipped), size_entry, intent_for(s, i, underlying) -> Intent (qty, price, risk_per_unit: Decimal),
  ROUND_UP_FLAG, rounds_up_too_often(entries, rounded_up), step_order(full, held, fraction, lot, min_qty) -> .qty,
  .skipped. `ok` is read as `qty > 0` when Sizing has no `ok`.
- B1's risk_long / risk_short fields as in WIP 269be90 (note v4: "risk_per_trade (risk_long / risk_short override)").
- sleeve_fund.portfolio.holding_for(position, mark, atr_pct) -> Holding with .risk (note v4 section 2); position has
  .qty (signed Decimal) and .stop (Decimal or None).
A module or name not built yet fails as AssertionError ("not built"), so each cell is a strict xfail until it is.
"""

import math
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

ITEM = "P2-1a pure sizing core (Quant Developer)"
SLIP = 0.0005  # R-SLIP: the stop slippage floor, 0.05%
MONEY = ("allocated_equity", "price", "atr", "max_notional", "volume_notional")


def xf(done: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"{ITEM} Done-when: {done}")


def _need(*names, module="sleeve_fund.portfolio.sizing"):
    import importlib

    try:
        mod = importlib.import_module(module)
        return tuple(getattr(mod, n) for n in names)
    except (ImportError, AttributeError) as e:
        raise AssertionError(f"not built: {module} {names}: {e}") from None


def _in(**kw):
    (SizingInputs,) = _need("SizingInputs")
    # No fees or spread: the only cost is the 0.05% stop slippage floor, so every quantity below is by hand.
    base = dict(allocated_equity=D(10_000), price=D(100), side=1, leg_cost=0.0, half_spread=0.0, risk_per_trade=0.01,
                position_cap_pct=0.9, lot=D("0.01"), min_qty=D("0.01"), stop_frac=0.02)
    kw = {k: (D(str(v)) if k in MONEY and isinstance(v, float) else v) for k, v in kw.items()}  # [V4] str, never float
    return SizingInputs(**{**base, **kw})


def _size(**kw):
    (size_entry,) = _need("size_entry")
    return size_entry(_in(**kw))


def _ok(s):
    return getattr(s, "ok", s.qty > 0)


def _skipped(**kw):
    """No order, skipped WITH a reason (not an exception, not a silent quantity)."""
    s = _size(**kw)
    assert s.qty == 0 and not _ok(s) and s.skipped, (s.qty, s.sized_by, s.skipped)
    return True


def _skipped_or_refused(**kw):
    """A NaN input: skipped with a reason, or refused with ValueError at the dataclass boundary [V4 rule 4]."""
    if any(isinstance(v, float) and math.isnan(v) for v in kw.values()):
        try:
            return _skipped(**kw)
        except ValueError:
            return True
    return _skipped(**kw)


def _within_one_lot(s, lot=D("0.01"), price=100.0):
    """Risk-bound: the quantity uses the budget, short of it by less than one more lot's risk."""
    ra, rb = float(s.risk_amount), float(s.risk_budget)
    per_unit = ra / (float(s.qty) * price)  # loss per unit of notional, costs and slippage included
    return ra <= rb + 1e-9 and ra + float(lot) * price * per_unit > rb


# --- rounding and the venue minimum ---------------------------------------------------------------------------

def test_size_rounds_down_to_the_step_and_never_past_the_cap():
    # Margin cap on a 2x perp: margin <= 20% x 10,000 = 2,000, notional 4,000 at 30,000 = 0.1333.. -> 0.133.
    s = _size(price=30_000.0, lot=D("0.001"), min_qty=D("0.001"), perp=True, leverage=2.0, position_cap_pct=0.2,
              risk_per_trade=0.05)
    assert s.qty == D("0.133") and s.qty * D(30_000) <= D(4_000), s.qty
    # Risk-bound: 100 of risk over a 2% stop + 0.05% slippage = 4,878-4,880 of notional = 0.1626.. -> 0.162, not 0.163.
    s = _size(price=30_000.0, lot=D("0.001"), min_qty=D("0.001"))
    assert s.qty == D("0.162"), s.qty
    assert float(s.qty) * 30_000 * (0.02 + SLIP * 0.98) <= 100.0 + 1e-9


def test_a_min_size_entry_over_one_and_a_half_budgets_is_refused_not_bumped():
    # Budget 1% of 1,000 = 10. One unit (the minimum) at 100 with a 20% stop risks 20+ = 2x the budget.
    assert _skipped(allocated_equity=1_000.0, lot=D(1), min_qty=D(1), stop_frac=0.2)


def test_the_venue_minimum_rounds_up_at_one_and_a_half_budgets():
    # Budget 1% of 800 = 8, so 1.5x = 12. One unit at 96: a 12.44% stop + 0.05% slippage risks 11.99-11.99, inside.
    s = _size(allocated_equity=800.0, price=96.0, lot=D(1), min_qty=D(1), stop_frac=0.1244)
    assert s.ok and s.qty == D(1) and s.rounded_up
    assert 11.98 <= float(s.risk_amount) <= 1.5 * float(s.risk_budget) + 1e-9, s.risk_amount
    # A 12.5% stop risked exactly 12 before slippage; with it, 12.04-12.05 > 12: no order.
    assert _skipped(allocated_equity=800.0, price=96.0, lot=D(1), min_qty=D(1), stop_frac=0.125)


@pytest.mark.parametrize("kw", [dict(max_notional=550.0), dict(position_cap_pct=0.5)])
def test_rounding_up_to_the_minimum_never_breaks_a_cap(kw):
    # Budget 1% of 1,000 = 10; 10 / 2.05% = 488 of notional, under one unit at 600. The minimum's risk, 12.3, is
    # within 1.5 x 10, but one unit (600) is over the 550 largest order / the 500 position cap.
    assert _skipped(allocated_equity=1_000.0, price=600.0, lot=D(1), min_qty=D(1), **kw)
    s = _size(allocated_equity=1_000.0, price=600.0, lot=D(1), min_qty=D(1), **kw)
    assert "cap" in s.skipped, s.skipped


def test_a_strategy_rounding_up_on_over_20pct_of_trades_is_flagged():
    ROUND_UP_FLAG, rounds_up_too_often = _need("ROUND_UP_FLAG", "rounds_up_too_often")

    assert ROUND_UP_FLAG == 0.2
    assert not rounds_up_too_often(entries=5, rounded_up=1)  # exactly 20%: not over
    assert not rounds_up_too_often(entries=10, rounded_up=2)
    assert rounds_up_too_often(entries=9, rounded_up=2)  # 22%


# --- the stop: fallback, zero, NaN, slippage --------------------------------------------------------------------

def test_the_fallback_stop_is_two_and_a_half_atrs():
    s = _size(stop_frac=None, atr=2.0)
    assert s.stop_frac == pytest.approx(0.05)  # 2.5 x 2 / 100
    assert s.ok and _within_one_lot(s)


@pytest.mark.parametrize("atr", [0.0, float("nan"), None])
def test_no_stop_and_a_zero_or_nan_atr_skips_the_entry(atr):
    assert _skipped_or_refused(stop_frac=None, atr=atr)


@pytest.mark.parametrize("atr", [None, 2.0])
def test_a_declared_stop_of_zero_is_refused_at_validation(atr):
    with pytest.raises(ValueError):
        _size(stop_frac=0.0, atr=atr)


def test_sizing_uses_the_stop_plus_its_slippage():
    # Long, 2% stop, 100 of budget at 100. No slippage would be 50.00 exactly.
    floor = _size()  # half spread 0: the 0.05% floor -> 100 / 2.049-2.05% = 48.78-48.80
    assert D("48.78") <= floor.qty <= D("48.80"), floor.qty
    spread = _size(half_spread=0.002)  # half spread 0.2% > 0.05% -> 100 / 2.196-2.2% = 45.45-45.53
    assert D("45.45") <= spread.qty <= D("45.53"), spread.qty
    venue = _size(stop_slippage=0.003)  # the venue's own setting, 0.3% -> 100 / 2.294-2.3% = 43.47-43.59
    assert D("43.47") <= venue.qty <= D("43.59"), venue.qty


# --- caps on margin, leverage 1x/2x/3x, liquidation, short symmetry ----------------------------------------------

@pytest.mark.parametrize("lev, qty", [(1.0, D("20.00")), (2.0, D("40.00")), (3.0, D("60.00"))])
def test_the_cap_is_measured_on_margin_at_each_leverage(lev, qty):
    s = _size(perp=True, leverage=lev, position_cap_pct=0.2, risk_per_trade=0.5)  # risk alone would size ~2,440 units
    assert s.qty == qty and "margin" in s.sized_by, (s.qty, s.sized_by)
    assert float(s.qty) * 100 <= 10_000 * lev  # never past the leverage cap either


def test_on_spot_the_cap_is_on_the_notional():
    s = _size(position_cap_pct=0.2, risk_per_trade=0.5)
    assert s.qty == D("20.00")


def test_a_stop_too_wide_for_the_leverage_is_refused_never_moved():
    # 20% stop at 3x isolated: liquidation ~33% away (1/3 - 0.5% maintenance, plus the fee), half of it ~16.4% < 20%.
    s = _size(stop_frac=0.2, perp=True, leverage=3.0, position_cap_pct=1.0, risk_per_trade=0.9,
              maintenance_margin=0.005, stop_to_liquidation=0.5)
    assert s.qty == 0 and s.skipped and "liquidation" in s.skipped, (s.qty, s.sized_by, s.skipped)
    assert s.stop_frac == 0.2  # never moved


@pytest.mark.parametrize("kw", [dict(), dict(perp=True, leverage=2.0, position_cap_pct=0.2, risk_per_trade=0.5)])
def test_a_short_is_sized_as_the_mirror_of_a_long(kw):
    base = {"perp": True, "leverage": 1.0, **kw}
    long_, short = _size(side=1, **base), _size(side=-1, **base)
    assert short.risk_budget == long_.risk_budget
    assert short.sized_by.replace("short", "").replace("long", "") == long_.sized_by.replace("short", "").replace(
        "long", "")
    if kw:  # the cap binds
        assert short.qty == long_.qty == D("40.00")
    else:  # the exit leg's slippage is on a short's higher exit value: at most a few lots apart
        assert _within_one_lot(long_) and _within_one_lot(short)
        assert abs(short.qty - long_.qty) <= D("0.10"), (long_.qty, short.qty)


def test_shorts_at_half_risk_size_at_half():
    long_ = _size(side=1, perp=True, risk_long=0.01, risk_short=0.005)
    short = _size(side=-1, perp=True, risk_long=0.01, risk_short=0.005)
    assert long_.risk_budget == D(100) and short.risk_budget == D(50)
    assert D("48.78") <= long_.qty <= D("48.80") and D("24.37") <= short.qty <= D("24.40"), (long_.qty, short.qty)
    assert _within_one_lot(long_) and _within_one_lot(short)
    assert "short" in short.sized_by and "long" in long_.sized_by


# --- B2 volatility targeting ----------------------------------------------------------------------------------

def test_volatility_targeting_matches_a_hand_calculation():
    # 0.01 x 10,000 / 0.02 = 5,000 of notional = 50 at 100; the floor (1%) and the 3x margin cap (6,000) don't bind.
    s = _size(overlay="vol_target", vol_target=0.01, instrument_vol=0.02, vol_floor=0.01, perp=True, leverage=3.0,
              position_cap_pct=0.2, risk_per_trade=0.5)
    assert s.qty == D("50.00") and "volatil" in s.sized_by


@pytest.mark.parametrize("vol, qty", [(0.005, (D("48.78"), D("48.80"))),  # vol size 20,000 -> capped 6,000; stop 4,880
                                      (0.04, (D("25.00"), D("25.00")))])  # vol size 2,500 < stop size 4,880
def test_the_smaller_of_the_stop_size_and_the_vol_size_is_taken(vol, qty):
    s = _size(overlay="vol_target", vol_target=0.01, instrument_vol=vol, vol_floor=0.001, perp=True, leverage=3.0,
              position_cap_pct=0.2)
    assert qty[0] <= s.qty <= qty[1], (s.qty, s.sized_by)
    assert ("volatil" in s.sized_by) == (vol == 0.04), s.sized_by


def test_volatility_below_the_floor_is_sized_at_the_floor():
    # 0.01 x 10,000 / max(0.004, 0.008) = 12,500 = 125 units; at the raw 0.4% it would be 250.
    s = _size(overlay="vol_target", vol_target=0.01, instrument_vol=0.004, vol_floor=0.008, perp=True, leverage=3.0,
              position_cap_pct=1.0, risk_per_trade=0.9)
    assert s.qty == D("125.00"), (s.qty, s.sized_by)


@pytest.mark.parametrize("vol, floor", [(float("nan"), 0.01), (None, 0.01), (0.02, None), (0.02, float("nan"))])
def test_nan_vol_or_a_missing_floor_skips_the_entry(vol, floor):
    assert _skipped_or_refused(overlay="vol_target", vol_target=0.01, instrument_vol=vol, vol_floor=floor, perp=True,
                    leverage=2.0, position_cap_pct=0.2, risk_per_trade=0.5)


@pytest.mark.parametrize("lev", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("vol", [0.005, 0.0])
def test_a_vol_sized_perp_entry_never_exceeds_the_margin_cap(lev, vol):
    s = _size(overlay="vol_target", vol_target=0.05, instrument_vol=vol, vol_floor=0.01, perp=True, leverage=lev,
              position_cap_pct=0.2, risk_per_trade=0.5)
    assert s.qty == D(str(20 * int(lev))) + D("0.00") and "margin" in s.sized_by, (s.qty, s.sized_by)


# --- A8 fractional steps --------------------------------------------------------------------------------------

def test_the_fraction_is_applied_before_the_caps():
    # Risk alone: 100 / (1% + 0.05%) = ~9,524 of notional; margin cap on 2x at 20% = 4,000.
    kw = dict(stop_frac=0.01, perp=True, leverage=2.0, position_cap_pct=0.2)
    assert _size(fraction=2 / 3, **kw).qty == D("40.00")  # min(~6,349, 4,000): the cap, not 2/3 of it
    third = _size(fraction=1 / 3, **kw).qty  # ~3,175 is under the cap: 31.74-31.76
    assert D("31.74") <= third <= D("31.76"), third


def test_fraction_steps_send_only_the_difference():
    (step_order,) = _need("step_order")

    full, held, orders = D("0.900"), D(0), []
    for f in (1 / 3, 2 / 3, 1.0, 0.0):
        o = step_order(full, held, f, D("0.001"), D("0.001"))
        orders.append(o.qty)
        held += o.qty
    assert orders == [D("0.300"), D("0.300"), D("0.300"), D("-0.900")] and held == 0


def test_a_step_below_the_minimum_is_skipped_without_drifting_the_target():
    (step_order,) = _need("step_order")

    full, min_qty = D("0.900"), D("0.350")
    first = step_order(full, D(0), 1 / 3, D("0.001"), min_qty)  # 0.300 < 0.350
    assert first.qty == 0 and first.skipped
    second = step_order(full, D(0), 2 / 3, D("0.001"), min_qty)  # still from 0 held: the target is 2/3 of full
    assert second.qty == D("0.600")


# --- [V4] Decimal boundary ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["allocated_equity", "price"])
def test_float_money_is_refused_and_int_str_decimal_agree(field):
    (SizingInputs, size_entry) = _need("SizingInputs", "size_entry")
    base = dict(allocated_equity=D(10_000), price=D(100), side=1, leg_cost=0.0, half_spread=0.0, risk_per_trade=0.01,
                position_cap_pct=0.9, lot=D("0.01"), min_qty=D("0.01"), stop_frac=0.02)
    with pytest.raises(TypeError):
        SizingInputs(**{**base, field: float(base[field])})
    sizes = {size_entry(SizingInputs(**{**base, field: v})).qty for v in (int(base[field]), str(base[field]),
                                                                          base[field])}
    assert len(sizes) == 1, sizes


@pytest.mark.parametrize("bad", [D("NaN"), D("Infinity"), D("-Infinity")])
def test_nan_or_inf_money_is_refused(bad):
    with pytest.raises(ValueError):
        _size(allocated_equity=bad)


@pytest.mark.parametrize("lot", [D("0.01"), D("0.001"), D("0.00000001")])
def test_quantity_is_an_exact_decimal_multiple_of_the_lot(lot):
    s = _size(price=D("30123.45"), lot=lot, min_qty=lot)
    assert isinstance(s.qty, D) and isinstance(s.risk_amount, D) and isinstance(s.risk_budget, D)
    assert s.qty > 0 and s.qty % lot == 0, (s.qty, lot)
    assert (s.qty + lot) * D("30123.45") * D("0.0205") > s.risk_budget  # one more lot would pass the budget


# --- [G1-G4] the guarantees that replace _cap_pct (Advisor MUST FIX 17:03 UK) -------------------------------------

STOPLESS = [dict(overlay="vol_target", vol_target=0.05, instrument_vol=0.01, vol_floor=0.005, stop_frac=None, atr=None,
                 perp=True, leverage=lev, position_cap_pct=1.0, risk_per_trade=0.5) for lev in (1.0, 2.0, 3.0, 5.0)]


@pytest.mark.parametrize("kw", STOPLESS, ids=lambda kw: f"lev{kw['leverage']}")
def test_g1_a_stopless_size_is_capped_at_one_times(kw):
    s = _size(**kw)
    if s.qty == 0:
        assert s.skipped
        return
    if s.stop_frac is None:  # no stop placed: the 1x cap binds
        assert s.qty * D(100) <= D(10_000), (s.qty, s.sized_by)
    else:  # a fallback stop was placed instead: it must be a real one (G4)
        assert s.stop_frac > 0


def test_g1_sweep_never_sizes_a_stopless_position_past_one_times():
    seen = 0
    for lev in (1.0, 2.0, 3.0, 5.0):
        for vol in (0.001, 0.01, 0.05):
            for overlay in ("stop", "vol_target"):
                try:
                    s = _size(overlay=overlay, vol_target=0.05, instrument_vol=vol, vol_floor=0.001, stop_frac=None,
                              atr=None, perp=True, leverage=lev, position_cap_pct=1.0, risk_per_trade=0.5)
                except ValueError:
                    continue
                if s.qty > 0 and s.stop_frac is None:
                    seen += 1
                    assert s.qty * D(100) <= D(10_000), (lev, vol, overlay, s.qty, s.sized_by)
    assert seen >= 0  # vacuous if the core never sizes without a stop, which G4 then pins


def test_g2_open_risk_formula_on_main_is_the_stopless_rule():
    """Guard (green on main 1709cd9): the ONE formula is notional x max(10%, 3 daily ATR) [G2]."""
    from sleeve_fund.open_risk import position_risk

    assert position_risk(2.0, 100.0, None, 0.02) == pytest.approx(2 * 100 * 0.10)  # 3 x 2% = 6% < 10%
    assert position_risk(-2.0, 100.0, None, 0.05) == pytest.approx(2 * 100 * 0.15)  # 3 x 5% = 15%
    assert position_risk(2.0, 100.0, 95.0, 0.05) == pytest.approx(10.0)  # with a stop: to the stop


@pytest.mark.parametrize("qty, atr_pct, risk", [(D(2), 0.02, D(20)), (D(-2), 0.05, D(30)), (D("0.5"), 0.0333, D(5))])
def test_g2_holding_for_counts_a_stopless_position_at_the_stopless_move(qty, atr_pct, risk):
    (holding_for,) = _need("holding_for", module="sleeve_fund.portfolio")
    h = holding_for(SimpleNamespace(qty=qty, stop=None, strategy="s", underlying="BTC"), D(100), atr_pct)
    assert isinstance(h.risk, D)
    assert abs(h.risk - risk) <= D("0.01"), (h.risk, risk)


def test_g3_every_perp_stop_fits_within_half_the_engines_distance_to_liquidation():
    from sleeve_fund.strategies.base import entry_liquidation

    for lev in (1.0, 2.0, 3.0, 5.0):
        for stop in (0.02, 0.05, 0.1, 0.2, 0.3, 0.45):
            for side in (1, -1):
                for mm in (0.005, 0.05):
                    s = _size(stop_frac=stop, side=side, perp=True, leverage=lev, position_cap_pct=1.0,
                              risk_per_trade=0.9, maintenance_margin=mm, stop_to_liquidation=0.5)
                    if s.qty == 0:
                        assert s.skipped, (lev, stop, side, mm)
                        continue
                    assert s.stop_frac == stop, (lev, stop, s.stop_frac)  # never moved
                    _, dist = entry_liquidation(10_000.0, float(s.qty), 100.0, side, 0.0, mm, lev)
                    assert stop <= 0.5 * dist + 1e-12, (lev, stop, side, mm, s.qty, dist)


@xf("G4: no declared stop -> the 2.5 x ATR fallback is a REAL stop: Intent's risk per unit is the stop distance, not "
    "the stopless 10% measure, and the size carries D13's stop slippage [G4, R-FB, R-SLIP]")
def test_g4_the_fallback_stop_is_a_real_stop_in_the_intent():
    (intent_for,) = _need("intent_for")
    i = _in(stop_frac=None, atr=D(2))  # 2.5 x 2 / 100 = 5% stop
    (size_entry,) = _need("size_entry")
    s = size_entry(i)
    assert s.stop_frac == pytest.approx(0.05) and s.qty > 0
    intent = intent_for(s, i, "BTC")
    assert intent.qty == s.qty and intent.price == i.price
    assert abs(intent.risk_per_unit - D(5)) <= D("0.000001"), intent.risk_per_unit  # 5% of 100, not 10% stopless
    # The size used stop + slippage: 100 / (5% + 0.05% x 0.95..1) = 1,980.2-1,980.4 of notional -> 19.80-19.81
    assert D("19.80") <= s.qty <= D("19.81"), s.qty


@xf("intent_for gives the gate the same per-unit risk sizing used (open_risk.position_risk), long and short [V4 s.2]")
@pytest.mark.parametrize("side", [1, -1])
def test_intent_risk_matches_position_risk(side):
    from sleeve_fund.open_risk import position_risk

    (intent_for, size_entry) = _need("intent_for", "size_entry")
    i = _in(side=side, perp=True, stop_frac=0.03)
    s = size_entry(i)
    intent = intent_for(s, i, "BTC")
    stop_px = 100 * (1 - side * 0.03)
    expected = position_risk(side * float(s.qty), 100.0, stop_px, 0.02)
    assert abs(float(intent.risk_per_unit * s.qty) - expected) <= 0.01, (intent.risk_per_unit, s.qty, expected)
