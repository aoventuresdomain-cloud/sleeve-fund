"""A multi-asset Strategy's backtest (multi-asset spec §2, §6 and §8 cell 6; Advisor MA-SPEC): a portfolio run of its legs
sized off the fund, per-asset results that sum exactly to the Strategy's, the fund counting each leg once, and the
fixed-list warning on every report. One asset off the others' bar clock refuses the whole Strategy, naming it."""

from __future__ import annotations

import pytest
from nautilus_trader.model import Venue

from sleeve_fund.instruments import spot_pair
from sleeve_fund.research.multi_asset import AssetData, run_multi_asset
from test_portfolio_run import FEES, LOOSE, _daily, _hourly

PARAMS = {}


def _asset(base: str, seed: int, days: int = 140, warm: int = 20) -> AssetData:
    h = _hourly(days, seed)
    d = _daily(h)
    w, window = d.iloc[:warm], d.iloc[warm:]
    return AssetData(spot_pair(base, "USD", FEES, Venue("KRAKEN")), window, h[h.index > w.index[-1]], w)


@pytest.fixture(scope="module")
def ran():
    return run_multi_asset("trend-x", "rsi_cross", [_asset("BTC", 2), _asset("ETH", 5), _asset("SOL", 9)], PARAMS,
                           "2026-10-15", capital=30_000.0, profile=LOOSE)


def test_the_legs_are_named_as_the_store_names_them(ran):
    assert [r["leg"] for r in ran.per_asset] == ["trend-x", "trend-x~ETH", "trend-x~SOL"]
    assert [r["asset"] for r in ran.per_asset] == ["BTC", "ETH", "SOL"]


def test_per_asset_results_sum_exactly_to_the_strategys(ran):
    assert sum(r["pnl"] for r in ran.per_asset) == ran.pooled["pnl"]
    assert sum(r["fees"] for r in ran.per_asset) == ran.pooled["fees"]
    assert sum(r["trades"] for r in ran.per_asset) == ran.pooled["trades"] > 0
    assert sum(r["share"] for r in ran.per_asset) == pytest.approx(1.0)


def test_the_fund_counts_each_leg_once(ran):
    ends = sum(float(r.equity.iloc[-1]) for r in ran.run.legs.values())
    assert float(ran.run.book.iloc[-1]) == pytest.approx(ends, rel=1e-3)  # the last hourly mark against each leg's last


def test_every_report_carries_the_fixed_list_warning(ran):
    assert ran.warning == "Fixed list chosen on 2026-10-15; history may favour assets that survived."


def test_one_asset_off_the_clock_refuses_the_strategy_naming_it():
    short = _asset("ETH", 5)
    short = AssetData(short.instrument, short.prices.iloc[1:], short.exec_prices, short.warmup_prices)
    with pytest.raises(ValueError, match="ETH"):
        run_multi_asset("trend-x", "rsi_cross", [_asset("BTC", 2), short], PARAMS, "2026-10-15", profile=LOOSE)
