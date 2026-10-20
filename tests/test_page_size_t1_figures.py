"""PAGE-SIZE-T1 (HoE 21:19 UK, Advisor confirmed 21:20 UK): every decision T1 judged carries what T1 compared, exactly
as the check had it, so the page's size lines read the row and never re-derive it:

  t1_used_before  the loss at the stops already in the asset, fund currency (the book before, reservations included)
  t1_cap          T1's cap, fund currency (4% of book equity when seeded)
  risk_per_unit   the order's loss at the stop per unit, as T1 used it (a stopless 1x entry's stopless measure)
  size_share      approved qty x risk_per_unit / (t1_cap - t1_used_before): the share of T1's room used; None with none

T1 off: no figures. Figures here are worked by hand."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D

from sleeve_fund.portfolio.compliance import SEEDED
from sleeve_fund.portfolio.limits import Book, Holding, Intent, decide, decide_pair

E = D(100_000)
STEP = D("0.001")
# T4 and T3 opened wide so T1 is the test that binds
LOOSE = replace(SEEDED, gross=10.0, net_instrument=10.0, margin=1.0, open_risk=1.0)


def _btc(qty, risk="500", underlying="BTC", side=1) -> Intent:
    """At 50,000 on spot; `risk` lost at the stop per unit."""
    return Intent(underlying, side, D(qty), D(50_000), D(50_000), D(risk), STEP, STEP, D(0), None)


def _book(btc_risk="1000"):
    return Book(E, (Holding("BTC", D(10_000), D(10_000), D(btc_risk)),))


def test_an_approved_order_journals_what_t1_compared():
    """Cap 4,000, 1,000 already at the stops, room 3,000; 0.2 x 500 = 100 of it: a thirtieth."""
    d = decide(_book(), _btc("0.2"), LOOSE)
    assert d.outcome == "approved"
    assert d.t1 == {"t1_used_before": D(1000), "t1_cap": D(4000), "risk_per_unit": D(500),
                    "size_share": D(100) / D(3000)}


def test_t1_trims_to_its_room_and_the_share_is_all_of_it():
    """10 asked at 2,000 a unit at the stop; room 3,000 / 2,000 = 1.5 approved (margin would allow 1.8), using all
    of T1's room."""
    d = decide(_book(), _btc("10", risk="2000"), LOOSE)
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("1.5"), "asset_open_risk")
    assert d.t1["size_share"] == 1 and d.t1["t1_used_before"] == D(1000) and d.t1["t1_cap"] == D(4000)


def test_no_room_left_keeps_the_figures_and_no_share():
    d = decide(_book("4000"), _btc("0.2"), LOOSE)
    assert d.outcome == "rejected" and d.t1 == {"t1_used_before": D(4000), "t1_cap": D(4000),
                                                "risk_per_unit": D(500), "size_share": None}


def test_t1_off_journals_nothing_and_another_asset_starts_from_zero():
    assert decide(_book(), _btc("0.2"), replace(LOOSE, asset_open_risk=None)).t1 == {}
    eth = decide(_book(), _btc("0.2", underlying="ETH"), LOOSE).t1
    assert eth["t1_used_before"] == 0 and eth["size_share"] == D(100) / D(4000)


def test_a_refusal_before_t1_is_judged_has_no_figures():
    assert decide(Book(D(0)), _btc("0.2"), LOOSE).t1 == {}


def test_each_leg_of_a_pair_journals_its_own_asset():
    """BTC 0.2 (room 3,000, risk 100) and ETH 1 at 200 a unit (room 4,000, risk 200)."""
    a, b = decide_pair(_book(), _btc("0.2"), _btc("1", risk="200", underlying="ETH", side=-1), LOOSE)
    assert a.outcome == b.outcome == "approved"
    assert a.t1 == {"t1_used_before": D(1000), "t1_cap": D(4000), "risk_per_unit": D(500),
                    "size_share": D(100) / D(3000)}
    assert b.t1 == {"t1_used_before": D(0), "t1_cap": D(4000), "risk_per_unit": D(200), "size_share": D("0.05")}
