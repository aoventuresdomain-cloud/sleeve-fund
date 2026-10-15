"""P1-GLC-3 (HoQA #189 round; Advisor 7 Oct: fixed before NIGHTLY-RECON goes live): after a liquidation booked at the
bankruptcy price, the equity marks the journal keeps equal its own cash to well below the cent. The simulated venue
realised the market-price fill's P&L to the cent, which the engine's float adjustment to the bankruptcy price didn't
see, so marks drifted from the journal by up to half a cent per fill slice: the nightly check's one-cent alert would
have tripped on a liquidated Strategy with a few slices. Red on main 76d7053 in paper (-0.0034, +0.0046, ...).

The GAP-LIQ-CAP master's harness (tests/test_gap_liq_cap_xfails.py): one probe Strategy entered, then a gap through its
liquidation price, short of bankruptcy and past it, long and short, 1x-3x, in paper and backtest. The backtest's own
fills report and curve keep the venue's cent fees (research/runner._liquidations_booked); only the journal is pinned.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

import test_gap_liq_cap_xfails as m
from sleeve_fund import markets

TOL = D("0.000001")
MM = 0.005  # LOW_FEE_PERP maintenance margin (KRAKEN simulated perp)
GRID = {"1x-short": (-1, "conservative", 1.0), "2x-short": (-1, "balanced", 2.0), "3x-short": (-1, "aggressive", 3.0),
        "2x-long": (1, "balanced", 2.0), "3x-long": (1, "aggressive", 3.0)}


def _gap(case: str, where: str) -> float:
    side, _, lev = GRID[case]
    if where == "past_bankruptcy":
        return round(1 - side * (1 / lev + 0.20), 6)
    return round((markets.liquidation_price(1 / lev - side, side, MM) + 1 - side / lev) / 2, 6)


CELLS = [(mode, case, where) for mode in ("paper", "backtest") for case in GRID
         for where in ("short_of_bankruptcy", "past_bankruptcy")]


@pytest.mark.parametrize("mode,case,where", CELLS, ids=["-".join(c) for c in CELLS])
def test_glc3_marks_after_a_liquidation_equal_the_journals_cash_below_the_cent(monkeypatch, mode, case, where):
    for k, v in GRID.items():
        monkeypatch.setitem(m.CASES, k, v)
    d = (m._backtest if mode == "backtest" else m._paper)(case, _gap(case, where))
    assert not d.get("handler_errors"), d.get("handler_errors")
    assert set(d["closing_intents"]) == {"liquidation"}, d["closing_intents"]
    cash = D(repr(float(d["cash_after"])))
    marks = [D(repr(float(v))) for v in d["marks"]]
    assert abs(marks[-1] - cash) <= TOL, (marks[-1] - cash)
    assert min(marks) >= cash - TOL, (min(marks) - cash)
