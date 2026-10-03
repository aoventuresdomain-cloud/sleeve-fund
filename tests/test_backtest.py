import pytest

from sleeve_fund.research.metrics import returns_from_equity
from sleeve_fund.research.runner import run_backtest


def test_buy_and_hold_pays_taker_fee(prices, instrument):
    res = run_backtest("buy_and_hold", prices, instrument, starting_capital=10_000)
    assert len(res.fills) == 1
    # One full-size buy at 0.8% taker, sized with a 1% buffer.
    assert res.fees_paid == pytest.approx(10_000 * 0.008 * (1 - 0.01 - 0.008), rel=0.01)
    assert res.equity.iloc[0] < 10_000


def test_trend_filter_waits_for_warmup(prices, instrument):
    res = run_backtest("trend_filter", prices, instrument, {"fast": 20, "slow": 100})
    first_fill = res.fills["ts_last"].min()
    assert first_fill >= prices.index[99]
    assert (res.equity.loc[: prices.index[98]] == 10_000).all()


def test_no_look_ahead(prices, instrument):
    # Changing the future must not change the past.
    cut = 600
    altered = prices.copy()
    altered.iloc[cut:, :4] = altered.iloc[cut:, :4] * 3
    a = run_backtest("trend_filter", prices, instrument, {"fast": 20, "slow": 100})
    b = run_backtest("trend_filter", altered, instrument, {"fast": 20, "slow": 100})
    assert a.equity.iloc[:cut].equals(b.equity.iloc[:cut])


def test_fast_must_be_shorter_than_slow(prices, instrument):
    with pytest.raises(ValueError):
        run_backtest("trend_filter", prices, instrument, {"fast": 200, "slow": 50})


def test_flat_strategy_has_flat_equity_when_out(prices, instrument):
    res = run_backtest("trend_filter", prices, instrument, {"fast": 20, "slow": 100})
    out = res.exposure < 0.01
    flat_rets = returns_from_equity(res.equity)[out.iloc[1:].values & out.shift(1, fill_value=True).iloc[1:].values]
    assert (flat_rets.abs() < 1e-12).all()
