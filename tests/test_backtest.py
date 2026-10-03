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


def _path(prices, closes):
    import numpy as np

    df = prices.iloc[: len(closes)].copy()
    c = np.asarray(closes, dtype=float)
    df["close"], df["open"] = c, np.concatenate([[c[0]], c[:-1]])
    df["high"], df["low"] = np.maximum(df["open"], c), np.minimum(df["open"], c)
    return df


def test_stop_loss_exits_and_waits_for_signal_reset(prices, instrument):
    from sleeve_fund.research.metrics import round_trips

    # Flat, then down 1% a day. Buy-and-hold's signal never resets, so one stop-out and no re-entry.
    closes = [10_000.0] * 10 + [10_000.0 * 0.99**i for i in range(1, 60)]
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument, {"stop_loss": 0.05})
    assert len(res.fills) == 2
    (trip,) = round_trips(res.fills)
    assert -0.08 < trip < -0.05  # about -5% plus a day's slippage past the level, plus fees


def test_take_profit_exits(prices, instrument):
    closes = [10_000.0] * 10 + [10_000.0 * 1.01**i for i in range(1, 60)]
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument, {"take_profit": 0.10})
    assert len(res.fills) == 2


def test_risk_per_trade_sizes_from_the_stop(prices, instrument):
    closes = [10_000.0] * 30
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument,
                       {"stop_loss": 0.05, "risk_per_trade": 0.01}, starting_capital=10_000)
    notional = float(res.fills["filled_qty"].iloc[0]) * float(res.fills["avg_px"].iloc[0])
    assert notional == pytest.approx(10_000 * 0.01 / 0.05, rel=0.03)  # ~2,000, not the whole sleeve


@pytest.mark.parametrize("bad", [{"stop_loss": 0}, {"stop_loss": 0.9}, {"risk_per_trade": 0.01}, {"take_profit": -1}])
def test_bad_exit_settings_rejected(prices, instrument, bad):
    with pytest.raises(ValueError):
        run_backtest("buy_and_hold", prices.iloc[:20], instrument, bad)
