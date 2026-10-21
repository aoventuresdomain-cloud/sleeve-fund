"""VS-1 (A12; Advisor ruling 15 Oct 06:58 UK): a perp liquidates at its venue's maintenance tiers, not one 0.5% constant.
The venue keeps notional x rate - amount for the tier the position's notional falls in, and its notional moves with the
price, so the liquidation price is solved per tier and kept where its own notional lands in that tier's band (an exact
tie on a band's floor takes the higher tier). The tables here are made up for the arithmetic: continuous at every
boundary, as the venue's maintenance amounts make them."""

import math

import pytest

from sleeve_fund import margin
from sleeve_fund.margin import Tier

# A test table, continuous at each ceiling: 0.004 x 50k - 0 = 0.005 x 50k - 50; 0.005 x 250k - 50 = 0.01 x 250k - 1,300
TIERS = (Tier(50_000, 0.004, 0), Tier(250_000, 0.005, 50), Tier(1_000_000, 0.01, 1_300), Tier(math.inf, 0.025, 16_300))


def _formula_a(cash, qty, entry, lev, tiers=TIERS):
    """Formula A alone (the maintenance on the position's value at the price), on the isolated margin at lev. The
    price used is the nearer of A and B (VS-1-TABLE: test_vs1_nearer_liquidation); these cells pin A's solve."""
    return margin.liquidation_price(margin.isolated_margin(qty, entry, lev, cash + qty * entry) - qty * entry, qty,
                                    tiers)


def _venue(cash, qty, p, tiers=TIERS):
    """The venue's own test at price p: equity cash + qty x p against the tier of the notional |qty| x p."""
    t = margin.tier_at(tiers, abs(qty) * p)
    return cash + qty * p - (abs(qty) * p * t.rate - t.amount)


def test_one_rate_is_the_formula_as_before():
    for cash, qty in ((-30_000.0, 1.0), (80_000.0, -1.0), (-1_000.0, 0.05)):
        old = cash / (0.005 * abs(qty) - qty)
        assert margin.liquidation_price(cash, qty, 0.005) == pytest.approx(old, rel=1e-15)
        assert margin.liquidation_price(cash, qty, margin.flat(0.005)) == pytest.approx(old, rel=1e-15)


def test_a_long_in_the_first_tier_liquidates_where_the_venue_does():
    p = _formula_a(40_000.0, 1.0, 60_000.0, 2.0)  # 1 BTC long at 60,000, 2x, on 100,000
    assert p == pytest.approx(30_000 / 0.996, rel=1e-12)  # tier 1: (-30,000 + 0) / (0.004 - 1)
    assert margin.tier_at(TIERS, p) is TIERS[0]
    assert _venue(-30_000.0, 1.0, p) == pytest.approx(0, abs=1e-6)


def test_a_3x_short_near_the_first_ceiling_is_solved_in_the_next_tier():
    """Advisor (b)(1): a short's notional rises towards liquidation. Tier 1 alone would put it at 79,681.27, whose
    notional is past tier 1's 50,000 ceiling: the venue liquidates it at tier 2's price."""
    p = margin.isolated_liquidation(160_000.0, -1.0, 60_000.0, 3.0, TIERS)  # 1 BTC short at 60,000, 3x, on 100,000
    assert p == pytest.approx((80_000 + 50) / 1.005, rel=1e-12)
    assert margin.tier_at(TIERS, p) is TIERS[1]
    assert _venue(80_000.0, -1.0, p) == pytest.approx(0, abs=1e-6)
    assert 80_000 / 1.004 > 50_000  # the tier-1 answer is not in its own band


def test_an_exact_tie_on_a_band_floor_takes_the_higher_tier():
    cash = 50_000 * (0.004 - 1)  # liquidates at exactly 50,000 notional for 1 unit long
    p = margin.liquidation_price(cash, 1.0, TIERS)
    assert p == pytest.approx(50_000, rel=1e-12)
    assert margin.tier_at(TIERS, 50_000) is TIERS[1]


@pytest.mark.parametrize("qty", [1.0, -1.0, 4.0, -4.0, 20.0, -20.0])
@pytest.mark.parametrize("lev", [1.0, 2.0, 3.0])
def test_every_answer_is_the_venues_own_and_in_its_own_band(qty, lev):
    cash = -qty * 60_000.0 + 60_000.0 * abs(qty)  # a spot-style book that put up the whole notional
    p = _formula_a(cash, qty, 60_000.0, lev)
    if p is None:
        assert qty > 0 and lev <= 1  # a fully paid long has no liquidation
        return
    margin_up = abs(qty) * 60_000.0 / lev
    assert _venue(margin_up - qty * 60_000.0, qty, p) == pytest.approx(0, abs=1e-6)


def test_a_funding_move_is_a_fresh_solve_never_a_shift_at_the_entry_s_tier():
    """Advisor (b)(2): recomputed on every move, never cached from entry. A payment that carries the price across a
    band gives the venue's price in the new tier."""
    cash, qty = 80_000.0, -1.0
    p0 = margin.liquidation_price(cash, qty, TIERS)
    for funded in (-10.0, -500.0, -31_000.0, 400.0):
        moved = margin.shifted_liquidation(p0, qty, funded, TIERS)
        assert moved == pytest.approx(margin.liquidation_price(cash + funded, qty, TIERS), rel=1e-12)
    assert margin.tier_at(TIERS, margin.shifted_liquidation(p0, qty, -31_000.0, TIERS)) is TIERS[0]


def test_flat_has_no_liquidation_and_fully_paid_liquidates_at_maintenance():
    """TEC-BH102-1X (HoE, Advisor 21 Oct): flat has none; a 1x long, fully paid for, liquidates at the maintenance on
    its entry value, here the 60,000 tier's 0.5% less its 50 amount (until 21 Oct it had none: BH-102)."""
    assert margin.liquidation_price(0.0, 0.0, TIERS) is None
    assert margin.isolated_liquidation(0.0, 1.0, 60_000.0, 1.0, TIERS) == pytest.approx(60_000.0 * 0.005 - 50)


def test_a_table_must_be_ordered_and_end_unbounded():
    with pytest.raises(ValueError):
        margin.tiers((Tier(250_000, 0.005, 50), Tier(50_000, 0.004, 0)))
    with pytest.raises(ValueError):
        margin.tiers((Tier(50_000, 0.004, 0),))
    with pytest.raises(ValueError):
        margin.tiers((Tier(50_000, 0.004, 0), Tier(math.inf, 0.005, 0)))  # not continuous at 50,000
    assert margin.tiers(TIERS) == TIERS


# --- the venue's tables, and none -------------------------------------------------------------------------------------

@pytest.mark.no_maintenance_tables
def test_a_perp_on_a_pair_with_no_table_is_refused_by_name_never_at_an_assumed_rate(monkeypatch):
    from sleeve_fund import markets
    from sleeve_fund.venues import venue

    monkeypatch.setattr(venue("BINANCE"), "maintenance_tiers", {})
    with pytest.raises(markets.NoMaintenanceTable, match="^BTC/USDT has no maintenance table"):
        markets.terms({"market": "perp"}, "BINANCE", "BTC/USDT")
    with pytest.raises(markets.NoMaintenanceTable, match="BTCUSDT-PERP"):
        markets.terms({"market": "perp"}, "BINANCE", "BTCUSDT-PERP")
    # what reads no liquidation still works: fees, funding, the label
    t = markets.terms({"market": "perp"}, "BINANCE")
    assert t.funding_venue == "BINANCE" and t.maintenance_margin is None


@pytest.mark.no_maintenance_tables
def test_the_venue_s_table_is_found_by_pair_or_by_its_own_symbol(monkeypatch):
    from sleeve_fund import markets
    from sleeve_fund.venues import venue

    monkeypatch.setattr(venue("BINANCE"), "maintenance_tiers", {"ETH/USDT": TIERS})
    assert markets.terms({"market": "perp"}, "BINANCE", "ETH/USDT").maintenance_margin == TIERS
    assert markets.terms({"market": "perp"}, "BINANCE", "ETHUSDT-PERP").maintenance_margin == TIERS
    with pytest.raises(markets.NoMaintenanceTable):
        markets.terms({"market": "perp"}, "BINANCE", "BTC/USDT")


def test_the_dashboard_s_maintenance_is_the_tier_s_at_the_mark():
    from sleeve_fund import markets

    # 4 units at 60,000: notional 240,000, tier 2 of TIERS: 240,000 x 0.005 - 50 = 1,150
    t = markets.tier_at(TIERS, 240_000)
    assert (240_000 * t.rate - t.amount, t.rate) == (1_150.0, 0.005)


@pytest.mark.no_maintenance_tables
def test_the_start_check_names_a_perp_strategy_whose_pair_has_no_table(tmp_path, monkeypatch):
    """VS-1 Done when 1: a perp Strategy already in the book on a pair whose table isn't kept is named by the start
    check (FAIL), never run at an assumed rate; a spot one, or a perp with its table, passes."""
    from sleeve_fund import start_check
    from sleeve_fund.store import Store
    from sleeve_fund.venues import venue

    url = f"sqlite:///{tmp_path}/j.db"
    store = Store(url)
    store.create_sleeve(name="p1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="ETH/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE")
    store.engine.dispose()
    monkeypatch.setattr(venue("BINANCE"), "maintenance_tiers", {"BTC/USDT": TIERS})
    _, page = start_check.run(url, environ={})
    assert "Maintenance tables: PASS" in page, page
    monkeypatch.setattr(venue("BINANCE"), "maintenance_tiers", {})
    ok, page = start_check.run(url, environ={})
    assert not ok and "Maintenance tables: FAIL" in page, page
    assert "  FAIL  p1: BTC/USDT has no maintenance table kept" in page and "FAIL  s1" not in page, page


def test_a_stop_held_at_the_cap_is_pulled_in_when_the_position_reaches_a_higher_rate():
    """VS-1 Done when 3 (HoQA re-clamp): the half-liquidation cap is measured to the CURRENT liquidation price, which
    is solved at the tier the position's notional lands in. A 2x long from 100 that grows from a 0.4% tier to a 1%
    tier liquidates nearer, so a stop held at its old cap is pulled in to half the way to the new price, journaled."""
    import dataclasses

    from test_timing import _late_strategy

    from sleeve_fund import risk

    s, _, events, _, _ = _late_strategy()
    s._cfg.perp = dataclasses.replace(s._cfg.perp, maintenance_margin=TIERS)
    s.runtime.profile, s.runtime.now = risk.profile("balanced"), lambda: None
    s._entry_px, s._entry_side, s._last_close, s._safety_stop = 100.0, 1, 100.0, None

    def held(qty):
        cash = -qty * 50  # all in at 2x: half the notional is the margin
        s._mark = lambda: (qty * 50, cash, qty, 100.0)
        liq = s._liq_now(cash, qty)
        # the nearer of the two formulas, each solved in the tier its own notional lands in (VS-1-TABLE)
        assert liq == pytest.approx(margin.nearer_liquidation(cash, qty, 100.0, TIERS), rel=1e-12)
        return liq

    small = held(100.0)  # notional at liquidation about 5,000: the 0.4% tier
    assert s._hold_stop_capped(0.30) and s._stop_frac == pytest.approx(1 - (100 - 0.5 * (100 - small)) / 100)
    before = s._stop_frac
    big = held(6_000.0)  # about 301,700: the 1% tier, so liquidation is nearer
    assert big > small
    assert s._hold_stop_capped(0.30) is True and s._stop_frac < before
    assert s._stop_frac == pytest.approx(1 - (100 - 0.5 * (100 - big)) / 100, abs=1e-9)
    said = [e for e in events if e[2] in ("stop_clamped_liquidation", "stop_clamped")]
    assert len(said) == 2 and f"{big:,.6g}" in said[-1][3], said
