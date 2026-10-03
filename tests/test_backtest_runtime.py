"""Backtests with a risk profile run the paper runtime, so the guard acts as it would in paper."""

import pytest

from sleeve_fund.research.runner import run_backtest
from test_backtest import _path


def _notional(fill):
    return float(fill["filled_qty"]) * float(fill["avg_px"])


def test_profile_sets_the_position_cap(prices, instrument):
    res = run_backtest("buy_and_hold", _path(prices, [100.0] * 20), instrument, starting_capital=10_000,
                       risk_profile="balanced")
    assert _notional(res.fills.iloc[0]) == pytest.approx(3_300, rel=0.02)
    assert res.risk_events == []


def test_drawdown_halt_flattens_and_stays_flat(prices, instrument):
    # 2% down a day: never a 3% daily equity loss at a 20% cap, but past the 10% drawdown halt.
    closes = [10_000.0] * 5 + [10_000.0 * 0.98**i for i in range(1, 60)]
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument, risk_profile="conservative")
    kinds = [e["kind"] for e in res.risk_events]
    assert kinds == ["risk_halt"]
    halted_at = res.risk_events[0]["ts"]
    assert "drawdown" in res.risk_events[0]["message"]
    sides = list(res.fills["side"])
    assert sides == ["BUY", "SELL"]
    # Halted: no re-entry, and equity is flat cash from the sell onwards.
    after = res.equity.loc[res.fills["ts_last"].max():]
    assert after.nunique() == 1
    assert 0.88 < after.iloc[-1] / 10_000 < 0.91
    assert halted_at.date() == res.fills["ts_last"].max().date()


def test_daily_loss_pauses_for_a_day_then_trades_again(prices, instrument):
    # A 30% one-day drop at a 20% cap is a 6% daily loss: past the 3% limit, short of the 10% halt.
    closes = [10_000.0] * 5 + [7_000.0] * 20
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument, risk_profile="conservative")
    assert [e["kind"] for e in res.risk_events] == ["risk_pause", "resume"]
    assert list(res.fills["side"]) == ["BUY", "SELL", "BUY"]
    pause, resume = (e["ts"] for e in res.risk_events)
    assert (resume - pause).total_seconds() == pytest.approx(86_400)


def test_without_a_profile_nothing_guards(prices, instrument):
    closes = [10_000.0] * 5 + [10_000.0 * 0.98**i for i in range(1, 60)]
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument)
    assert res.risk_events == []
    assert list(res.fills["side"]) == ["BUY"]


def test_runtime_and_profile_are_exclusive(prices, instrument):
    from sleeve_fund.paper.runtime import SleeveRuntime

    rt = SleeveRuntime.for_backtest(strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                    starting_balance=10_000, risk_profile="balanced")
    with pytest.raises(ValueError, match="not both"):
        run_backtest("buy_and_hold", prices, instrument, runtime=rt, risk_profile="balanced")


def test_stop_rests_at_the_venue_and_is_journaled(prices, instrument):
    closes = [10_000.0] * 10 + [10_000.0 * 0.99**i for i in range(1, 30)]
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument, {"stop_loss": 0.05},
                       risk_profile="aggressive")
    assert list(res.fills["side"]) == ["BUY", "SELL"]
    stop = res.fills.iloc[1]
    assert float(stop["avg_px"]) == pytest.approx(10_000 * 0.95, rel=0.002)  # at the level, not a bar close
    assert res.decisions[res.fills.index[1]]["intent"] == "stop_loss"


def test_a_runtime_passed_in_rests_its_stop_like_every_backtest(instrument):
    """Same bars, same 10% stop: the fill is the same with or without a runtime (review R2-B4)."""
    import pandas as pd

    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store

    idx = pd.date_range("2024-01-02", periods=6, freq="1D", tz="UTC")
    c, o = [100, 110, 120, 80, 85, 90], [100, 105, 118, 119, 82, 86]
    df = pd.DataFrame({"open": o, "high": [max(a, b) + 1 for a, b in zip(o, c)],
                       "low": [min(a, b) - 1 for a, b in zip(o, c)], "close": c, "volume": 1e6}, index=idx)
    plain = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.10}, half_spread=0)
    store = Store.in_memory()
    store.create_sleeve(name="x", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, params={"stop_loss": 0.10}, risk_profile="aggressive")
    rt = SleeveRuntime(store, "x", tick_seconds=86_400)
    guarded = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.10}, runtime=rt, half_spread=0)
    assert float(plain.fills["avg_px"].iloc[-1]) == pytest.approx(90.0)
    assert float(guarded.fills["avg_px"].iloc[-1]) == pytest.approx(90.0)
    assert [(o["intent"], o["side"]) for o in reversed(store.orders("x"))] == [("entry", "BUY"), ("stop_loss", "SELL")]


def test_trend_filter_runs_minute_bars_quickly(instrument):
    """30 days of 1-minute bars with volatility targeting (43,200-bar window) in seconds (review R2-B3)."""
    import time

    from sleeve_fund.data import synthetic_ohlcv

    df = synthetic_ohlcv(days=30 * 1440, seed=2, vol=0.0008)
    df.index = df.index[0] + (df.index - df.index[0]) / 1440  # daily rows re-stamped a minute apart
    t = time.perf_counter()
    run_backtest("trend_filter", df, instrument, {"fast": 50, "slow": 200, "vol_target": 0.4}, bar_minutes=1)
    assert time.perf_counter() - t < 20
