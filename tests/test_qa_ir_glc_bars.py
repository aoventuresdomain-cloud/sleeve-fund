"""Integration round on main 1709cd9: two more cells re-pinned to the rulings now on main.

Ported from quant-review/v2-p1/integration-1709cd9-scripts/test_ir_glc_bars.py (round integration-1709cd9.md). The
source loaded gap-liq-cap-189-scripts/test_qa_glc189_probes.py from QA's folder; it is carried verbatim as
tests/qa_ir_glc189_lib.py.

ir4: GAP-LIQ-CAP outage replay, 3x/2x short (gap-liq-cap-189 p2). The booked loss is exactly X (unchanged, 1e-6).
     The equity marks after the liquidation sit 0.0525 cents below the journal's 10,000 - X on 1709cd9: the
     P1-GLC-3 class (FOLLOW-UP, marks vs journal up to half a cent). Pinned at that bound, not at 1e-6.
ir5: stop-safety guard R-S9 (ext-ss). D13 (2) (Advisor 6 Oct 18:16): a bars-only stop the bar traded through
     fills at the worse of its trigger and the bar's adverse extreme, then the stop slippage floor. Pinned as: never
     better than the trigger (or the open on a gap), never worse than the low less 0.05 % (+ rounding).

Only ir4 is ported: it goes red before GAP-LIQ-CAP (#189, be5cefe) on b7368b5. ir5 passes on every older head tried
(1918f8b, 018bb23, a83de76, 1582c3a, 9ae1f8a, 00f4b5a), so it pins no regression.
"""
from decimal import Decimal as D

import pytest

import tests.qa_ir_glc189_lib as g

HALF_CENT = D("0.005")


@pytest.mark.parametrize("side,profile,lev,gap,back", [(-1, "aggressive", 3.0, 0.40, 0.10),
                                                       (-1, "balanced", 2.0, 0.60, 0.10),
                                                       (-1, "aggressive", 3.0, 0.3317, 0.10)],
                         ids=["3x-short-0.4", "2x-short-0.6", "3x-short-0.3317"])
def test_ir4_outage_replay_books_exactly_x_and_marks_stay_within_half_a_cent(side, profile, lev, gap, back):
    """Source: quant-review/v2-p1/integration-1709cd9-scripts/test_ir_glc_bars.py::
    test_ir4_outage_replay_books_exactly_x_and_marks_stay_within_half_a_cent. Finding: ir4 / IR-1 (P1-GLC-3 class)."""
    from sleeve_fund.strategies import REGISTRY

    hh = g.m._hub_harness()
    saved = REGISTRY.get("probe")
    REGISTRY["probe"] = hh._probe_classes()
    try:
        p = hh.flat_prices(30)
        p = hh.shape(hh.shape(p, 7.0, 8.0, hh.adverse(side, gap)), 8.0, 30, hh.adverse(side, back))
        r = hh.restart(p, 6, 12, side=side, perp=True, profile=profile, stop=0.01, qty=0.25)
    finally:
        if saved is None:
            REGISTRY.pop("probe", None)
        else:
            REGISTRY["probe"] = saved
    dec = g.dec
    held = [f for f in r.fills if f["order_id"] == "O-held"]
    closed = [f for f in r.fills if f["order_id"] != "O-held"]
    q = sum(dec(f["qty"]) for f in held)
    entry_px = sum(dec(f["qty"]) * dec(f["price"]) for f in held) / q
    x = q * entry_px / dec(lev) + sum(dec(f["fee"]) for f in held) + sum(dec(f["fee"]) for f in closed)
    loss = D(10_000) - dec(r.store.journal_book("q146", 10_000)["cash"]) + dec(r.store.funding_total("q146"))
    marks = [dec(mk["equity"]) for mk in r.store.equity_series("q146", limit=100_000)]
    assert sorted({o["intent"] for o in r.orders if o["order_id"] != "O-held"}) == ["liquidation"]
    assert abs(loss - x) <= g.TOL, (loss, x)
    assert not list(r.store.insurance("q146"))
    assert marks and min(marks) >= D(10_000) - x - HALF_CENT, (min(marks), D(10_000) - x)
