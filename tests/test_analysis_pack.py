"""R-17/R-18/R-19 (HoE 21 Oct; spec research/quant-researcher/fund-grade-analysis-spec.md sections 3-6): every figure of
the analysis pack on a small fixture whose answer is worked out by hand here, never read from the code."""
from __future__ import annotations

import math
import statistics
from types import SimpleNamespace

import pandas as pd
import pytest

from sleeve_fund.research import analysis_pack as ap

UTC = "UTC"


def _daily(values, start="2026-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="1D", tz=UTC), dtype=float)


# R-17
def test_rolling_12_months_starts_with_the_first_full_window_and_matches_a_hand_sum():
    rets = [0.01 if k % 2 else -0.005 for k in range(366)]
    r = _daily(rets)
    roll = ap.rolling_12m(r)
    assert len(roll) == 2  # windows ending on day 365 and day 366 only
    for end, (got_ret, got_sharpe) in zip((365, 366), roll[["return", "sharpe"]].itertuples(index=False)):
        w = rets[end - 365:end]
        assert got_ret == pytest.approx(math.prod(1 + x for x in w) - 1)
        assert got_sharpe == pytest.approx(statistics.mean(w) / statistics.stdev(w) * math.sqrt(365))
    assert ap.rolling_12m(_daily([0.01] * 364)).empty  # no full window: nothing, not a part-year figure


def test_best_and_worst_day_week_and_month_count_only_whole_calendar_periods():
    # Days 1 Jan to 3 Feb 2026 (a return stamped at the next midnight is that day's move): +10% on Sat 10 Jan,
    # -5% on Tue 20 Jan, flat otherwise. Whole weeks: 5-11, 12-18, 19-25 and 26 Jan - 1 Feb; whole months: January.
    days = pd.date_range("2026-01-01", "2026-02-03", freq="1D", tz=UTC)
    r = pd.Series(0.0, index=days + pd.Timedelta("1D"))
    r[pd.Timestamp("2026-01-11", tz=UTC)] = 0.10
    r[pd.Timestamp("2026-01-21", tz=UTC)] = -0.05
    bw = ap.best_worst(r)
    assert bw["day"][0] == pytest.approx(0.10) and bw["day"][1] == pd.Timestamp("2026-01-10")
    assert bw["day"][2] == pytest.approx(-0.05) and bw["day"][3] == pd.Timestamp("2026-01-20")
    assert bw["week"][0] == pytest.approx(0.10) and bw["week"][1] == pd.Timestamp("2026-01-05")
    assert bw["week"][2] == pytest.approx(-0.05) and bw["week"][3] == pd.Timestamp("2026-01-19")
    assert bw["month"][0] == bw["month"][2] == pytest.approx(1.10 * 0.95 - 1)  # January only; February is a part
    assert bw["month"][1] == pd.Timestamp("2026-01-01")
    # a +50% day in the part week before 5 Jan counts as a day, never as a week
    r[pd.Timestamp("2026-01-03", tz=UTC)] = 0.50
    bw = ap.best_worst(r)
    assert bw["day"][0] == pytest.approx(0.50) and bw["week"][0] == pytest.approx(0.10)


def test_drawdowns_are_episodes_from_peak_to_recovery_and_an_open_one_is_not_recovered():
    eq = _daily([100, 110, 99, 121, 110, 88, 95])
    d = ap.drawdowns(eq)
    assert len(d) == 2
    assert d[0].depth == pytest.approx(99 / 110 - 1) and d[0].start == eq.index[1] and d[0].trough == eq.index[2]
    assert d[0].recovered == eq.index[3] and d[0].days(eq.index[-1]) == 2
    assert d[1].depth == pytest.approx(88 / 121 - 1) and d[1].start == eq.index[3] and d[1].trough == eq.index[5]
    assert d[1].recovered is None and d[1].days(eq.index[-1]) == 3
    assert [x.depth for x in ap.worst_drawdowns(eq)] == [pytest.approx(88 / 121 - 1), pytest.approx(99 / 110 - 1)]
    assert ap.longest_under_water(eq).start == eq.index[3]  # the open one, 3 days, over the 2-day one
    assert ap.drawdowns(_daily([1, 2, 3])) == []
    back = ap.drawdowns(_daily([100, 110, 99, 110, 105]))  # back exactly at the peak is recovered
    assert [(d.depth, d.recovered is None) for d in back] == [(pytest.approx(-0.1), False),
                                                              (pytest.approx(105 / 110 - 1), True)]


def test_var_and_expected_shortfall_take_the_k_worst_days_with_k_the_ceiling_of_5_percent():
    base = [0.01] * 38 + [-0.05, -0.04]  # 40 days: k = 2
    assert ap.var_es(_daily(base)) == (pytest.approx(0.04), pytest.approx(0.045))
    # 39 days: k = ceil(1.95) = 2, so the 2 worst (-5%, -3%): VaR 3%, ES 4%
    assert ap.var_es(_daily([0.01] * 37 + [-0.05, -0.03])) == (pytest.approx(0.03), pytest.approx(0.04))
    twenty = [0.01] * 19 + [-0.03]  # 20 days: k = 1
    assert ap.var_es(_daily(twenty)) == (pytest.approx(0.03), pytest.approx(0.03))


def test_correlation_and_beta_to_buy_and_hold():
    b = _daily([0.01, -0.02, 0.03, 0.0])
    corr, beta = ap.correlation_beta(2 * b, b)
    assert corr == pytest.approx(1.0) and beta == pytest.approx(2.0)
    corr, beta = ap.correlation_beta(-0.5 * b, b)
    assert corr == pytest.approx(-1.0) and beta == pytest.approx(-0.5)


# R-18
def _trips(pnls, start=100.0):
    out, eq = [], start
    for p in pnls:
        out.append({"pnl": p, "ret": p / 50.0})  # ret: on a 50 notional
        eq += p
    return out


def test_payoff_largest_streaks_with_their_falls_and_the_best_trades_share():
    tf = ap.trade_figures(_trips([10, -5, -5, 20, -10]), 100.0)
    assert tf["payoff_ratio"] == pytest.approx(15 / (20 / 3))  # average win 15, average loss -20/3
    assert tf["largest_win"] == (20, pytest.approx(0.4)) and tf["largest_loss"] == (-10, pytest.approx(-0.2))
    assert tf["longest_win_streak"] == 1 and tf["longest_loss_streak"] == 2
    # equity 100, 110, 105, 100, 120, 110: two losses from the 110 peak to 100, then one from 120 to 110
    assert tf["loss_streaks"] == [(2, pytest.approx(100 / 110 - 1)), (1, pytest.approx(110 / 120 - 1))]
    assert tf["best5_share"] == pytest.approx(1.0)  # all five trades are the top five: 10 of 10
    tf = ap.trade_figures(_trips([30, 20, 10, 5, 5, 1, -11]), 100.0)
    assert tf["best5_share"] == pytest.approx(70 / 60)
    assert ap.trade_figures(_trips([-1, 0]), 100.0)["best5_share"] is None  # no total profit: n/a
    # the fall runs from the highest equity before the streak, not the equity just before it: 100, 120, 105, 110, 105
    assert ap.trade_figures(_trips([20, -15, 5, -5]), 100.0)["loss_streaks"] == [
        (1, pytest.approx(105 / 120 - 1)), (1, pytest.approx(105 / 120 - 1))]


def test_trade_returns_compound_on_the_equity_before_each_trade():
    assert ap.trade_returns(_trips([10, -11, 20]), 100.0) == [pytest.approx(0.1), pytest.approx(-0.1),
                                                                 pytest.approx(20 / 99)]


def test_reshuffling_equal_trades_gives_identical_percentiles():
    out = ap.reshuffle([0.02] * 12, 1000.0)
    assert not out["exact"] and out["orders"] == ap.DRAWS
    for k in ("end_value", "max_drawdown", "longest_loss_streak"):
        assert out[k][0] == out[k][1] == out[k][2]
    assert out["end_value"][0] == pytest.approx(1000 * 1.02 ** 12) and out["max_drawdown"][0] == 0.0


def test_reshuffling_three_trades_enumerates_all_six_orders_exactly():
    # +10%, -10%, -20% from 100. Every order ends at 79.2. Max drawdown -28% in the 4 orders where both losses follow
    # the peak, -20.8% in the 2 that put the win between them; longest losing run 2 in 4 orders, 1 in 2.
    out = ap.reshuffle([0.1, -0.1, -0.2], 100.0)
    assert out["exact"] and out["orders"] == 6
    assert out["end_value"] == (pytest.approx(79.2), pytest.approx(79.2), pytest.approx(79.2))
    # sorted: -0.28 x4, -0.208 x2; linear positions 0.25, 2.5, 4.75
    assert out["max_drawdown"] == (pytest.approx(-0.28), pytest.approx(-0.28), pytest.approx(-0.208))
    # sorted: 1, 1, 2, 2, 2, 2
    assert out["longest_loss_streak"] == (pytest.approx(1.0), pytest.approx(2.0), pytest.approx(2.0))


def test_a_flat_trade_counts_as_a_loss_in_the_luck_test_as_in_the_trade_figures():
    assert ap.reshuffle([0.0, 0.0, 0.0], 100.0)["longest_loss_streak"] == (3.0, 3.0, 3.0)
    assert ap.trade_figures(_trips([0, 0, 0]), 100.0)["longest_loss_streak"] == 3


def test_the_draws_are_fixed_by_the_seed():
    rets = [0.05, -0.03, 0.02, -0.04, 0.01, 0.06, -0.02, 0.03]
    assert ap.reshuffle(rets, 100.0) == ap.reshuffle(rets, 100.0)
    assert ap.reshuffle(rets, 100.0, seed=1) != ap.reshuffle(rets, 100.0)


# R-19
def test_costs_reconcile_gross_to_net_to_the_cent_with_funding_both_ways_and_insurance():
    # One unit bought at a mid of 100 and sold at a mid of 120: 20.00 gross. Fees 0.50, spread and slippage 0.30,
    # funding paid 0.40 and received 0.10, the insurance fund took 0.20 of a shortfall: net 19.10.
    run = SimpleNamespace(fees_paid=0.5, spread_paid=0.3, starting_capital=1000.0,
                          funding=[{"ts": 1, "amount": -0.4}, {"ts": 2, "amount": 0.1}],
                          insurance=[{"ts": 3, "amount": 0.2}], equity=_daily([1000.0, 1019.1]))
    c = ap.costs(run)
    assert round(c["gross"], 2) == 20.00 and round(c["net"], 2) == 19.10
    assert round(c["funding"], 2) == -0.30 and c["insurance"] == pytest.approx(0.2)
    assert round(c["gross"] - c["fees"] - c["spread_slippage"] + c["funding"] + c["insurance"], 2) == round(c["net"], 2)


def test_a_cost_as_a_share_of_average_equity_a_year():
    eq = _daily([1000.0] * 366)  # 365 days at 1,000 on average
    assert ap.per_year(10.0, eq) == pytest.approx(0.01)


# The sheet
def test_a_studys_sheet_carries_the_section_with_its_caveats_and_the_modelled_spread_line(tmp_path, instrument):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research import tearsheet
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1900, seed=3), instrument, dataset="syn", ledger=ledger, synthetic=True,
                  holdout_days=200, train_days=730, test_days=365)
    sheet = tearsheet.render(r, ledger)
    assert "## Risk, trades and costs" in sheet and "Over the full research period" in sheet
    assert "not measured" in sheet and "Survivorship" in sheet and "R-01" in sheet
    assert "Trade-order luck test" in sheet and "Five worst drawdowns" in sheet
    # the cost line's sign convention, beside it (Advisor, 21 Oct), and buy and hold against itself said so (CR F2)
    assert "| Funding (+ received, - paid) |" in sheet and "funding is + when received and - when paid" in sheet
    assert "the insurance fund's share is +" in sheet
    assert "1.00 / 1.00 (itself)" in sheet
    # the sheet's own costs reconcile: gross less fees and spread, plus funding and insurance, is the equity's change
    c = ap.costs(r.full_period)
    assert round(c["gross"] - c["fees"] - c["spread_slippage"] + c["funding"] + c["insurance"], 2) == round(
        float(r.full_period.equity.iloc[-1] - r.full_period.starting_capital), 2)
