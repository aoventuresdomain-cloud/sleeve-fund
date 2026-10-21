"""Analysis pack v3 (HoE 21 Oct, DA handover regime-pack-handover.md): (b) the best trade's and the best 3 trades'
share of the total profit beside the best 5's; (a) the Strategy's return while in the market beside the price's in and
out of it; (c) a plain price-only hold over the window. Expected values are worked out by hand here, never read from the code."""
from __future__ import annotations

import pandas as pd
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


# (a)
def _daily(values, start="2026-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="1D", tz="UTC"), dtype=float)


def test_the_strategy_in_the_market_beside_the_price_in_and_out_of_it():
    closes = _daily([100, 110, 121, 100, 90, 99, 108.9, 120, 120, 132])
    equity = _daily([1000, 1000, 1100, 1210, 1210, 1210, 1210, 1300, 1300, 1300])
    # held at the closes of days 1, 2 and 6 (0-based), so the moves into days 2, 3 and 7 are in the market
    exposure = _daily([0, 1, 1, 0, 0, 0, -0.5, 0, 0, 0])
    got = ap.in_and_out(equity, exposure, closes)
    assert got["days_in"] == 3 and got["days_out"] == 6
    assert got["strategy_in"] == pytest.approx(1300 / 1000 - 1)  # every gain came while in
    assert got["strategy_out"] == pytest.approx(0.0)
    assert got["price_in"] == pytest.approx((121 / 110) * (100 / 121) * (120 / 108.9) - 1)
    assert got["price_out"] == pytest.approx(
        (110 / 100) * (90 / 100) * (99 / 90) * (108.9 / 99) * (120 / 120) * (132 / 120) - 1)


def test_a_day_is_in_the_market_when_any_bar_of_it_held_a_position():
    idx = pd.date_range("2026-01-01", periods=72, freq="1h", tz="UTC")  # three days of hourly bars
    closes = pd.Series(100.0, index=idx)
    closes.iloc[25:49] = [100.0 + k for k in range(1, 25)]  # the move into 3 Jan's close climbs 100 to 124
    closes.iloc[49:] = 124.0
    exposure = pd.Series(0.0, index=idx)
    exposure.iloc[30:33] = 1.0  # held for three hours on the second day only
    equity = pd.Series(1000.0, index=idx)
    equity.iloc[30:] = 1050.0
    got = ap.in_and_out(equity, exposure, closes)
    # daily closes at 1, 2 and 3 Jan midnight (the part day after 3 Jan's is left out); held at 06:00-08:00 on 2 Jan,
    # inside the move into 3 Jan's close, so that day is in and the move into 2 Jan is out
    assert got["days_in"] == 1 and got["days_out"] == 1
    assert got["strategy_in"] == pytest.approx(0.05) and got["price_in"] == pytest.approx(124 / 100 - 1)
    assert got["price_out"] == pytest.approx(0.0)


def test_never_in_the_market_reads_n_a_for_the_in_figures():
    closes = _daily([100, 110, 121])
    got = ap.in_and_out(_daily([1000] * 3), _daily([0] * 3), closes)
    assert got["days_in"] == 0 and got["strategy_in"] is None and got["price_in"] is None
    assert got["price_out"] == pytest.approx(0.21)


# (c)
def test_the_price_only_hold_over_the_window():
    assert ap.price_only_hold(_daily([100, 80, 150])) == pytest.approx(0.5)
    idx = pd.date_range("2026-01-01", periods=50, freq="1h", tz="UTC")  # to 3 Jan's close, and one bar after it
    closes = pd.Series([100.0] * 24 + [120.0] * 25 + [999.0], index=idx)
    assert ap.price_only_hold(closes) == pytest.approx(0.2)  # first to last daily close, as the window's total


# The sheet
@pytest.fixture(scope="module")
def sheet(tmp_path_factory):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research import tearsheet
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import venue

    ledger = IdeaLedger(tmp_path_factory.mktemp("pack_v3") / "l.jsonl")
    prices = synthetic_ohlcv(days=900, seed=11)
    r = run_study(SPEC, prices, venue("KRAKEN").instrument("BTC", "USD"), dataset="syn", ledger=ledger,
                  synthetic=True, holdout_days=0, train_days=300, test_days=200)
    return r, prices, tearsheet.render(r, ledger)


def test_the_sheet_carries_the_new_lines(sheet):
    r, prices, text = sheet
    assert "| Best trade's share of total profit |" in text and "| Best 3 trades' share of total profit |" in text
    close = prices["close"].astype(float)
    assert f"Price-only hold over the window (price only, no fees): {close.iloc[-1] / close.iloc[0] - 1:+.2%}" in text
    assert "| In the market |" in text and "| Out of the market |" in text
