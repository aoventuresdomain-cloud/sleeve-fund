"""Analysis pack v3 (HoE 21 Oct, DA handover regime-pack-handover.md): (b) the best trade's and the best 3 trades'
share of the total profit beside the best 5's. Expected values are worked out by hand here, never read from the code."""
from __future__ import annotations

import pytest

from sleeve_fund.research import analysis_pack as ap


def _trips(pnls):
    return [{"pnl": p, "ret": p / 50.0} for p in pnls]


# (b)
def test_the_best_trade_and_the_best_three_share_of_the_total_profit():
    tf = ap.trade_figures(_trips([5, 30, -11, 10, 1, 20, 5]), 100.0)  # total 60, best 30, best three 30 + 20 + 10
    assert tf["best1_share"] == pytest.approx(30 / 60)
    assert tf["best3_share"] == pytest.approx(60 / 60)
    assert tf["best5_share"] == pytest.approx(70 / 60)  # unchanged beside them
    tf = ap.trade_figures(_trips([10, -4]), 100.0)  # fewer than three trades: the best three are all of them
    assert tf["best1_share"] == pytest.approx(10 / 6) and tf["best3_share"] == pytest.approx(6 / 6)


def test_no_total_profit_reads_n_a_for_every_share():
    for pnls in ([-1, 0], [5, -5], []):
        tf = ap.trade_figures(_trips(pnls), 100.0)
        assert tf["best1_share"] is None and tf["best3_share"] is None and tf["best5_share"] is None
