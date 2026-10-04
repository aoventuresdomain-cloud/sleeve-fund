"""Backtests with a risk profile run the paper runtime, so the guard acts as it would in paper."""

import pandas as pd
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


def test_a_halt_and_an_exit_on_the_same_bar_sell_once(instrument):
    """A risk halt's flatten and the strategy's own exit fell on one bar (16 Nov 2020 in this series).
    The backtest's market sell had not reached the venue yet, so both sold the whole position and the
    book went short. Only the first sell goes now."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.paper.runtime import SleeveRuntime

    rt = SleeveRuntime.for_backtest(strategy="trend_filter", instrument="ETH/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                    starting_balance=10_000, risk_profile="aggressive")
    run_backtest("trend_filter", synthetic_ohlcv(days=1200, seed=3, vol=0.03), instrument,
                 params={"fast": 5, "slow": 20}, runtime=rt)
    orders = rt.store.orders(limit=10_000)[::-1]
    halt = [o for o in orders if o["intent"] == "risk_halt"]
    assert halt, "the series should still contain the halt bar"
    same_bar = [o for o in orders if o["side"] == "SELL" and o["ts"] == halt[0]["ts"]]
    assert [o["intent"] for o in same_bar] == ["risk_halt"]
    assert rt.store.journal_book("backtest", 10_000)["qty"] >= 0


def _minutes_to_daily(m):
    g = m.resample("1D", closed="right", label="right")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum()}).dropna()


def test_the_guard_checks_every_execution_bar_like_paper(instrument):
    """Review round 3, R3-M1: paper values the book every 30 s, so a daily-loss pause fires near its
    limit. A backtest fed minute bars now does the same instead of waiting for the day's close."""
    import numpy as np

    days = 12
    idx = pd.date_range("2024-01-01 00:01", periods=days * 1440, freq="1min", tz="UTC")
    close = np.full(len(idx), 100.0)
    crash = slice(10 * 1440, 11 * 1440)  # day 11 falls 40% minute by minute, then stays there
    close[crash] = np.linspace(100.0, 60.0, 1440)
    close[11 * 1440:] = 60.0
    m = pd.DataFrame({"open": np.concatenate([[100.0], close[:-1]]), "close": close, "volume": 5.0}, index=idx)
    m["high"], m["low"] = m[["open", "close"]].max(axis=1), m[["open", "close"]].min(axis=1)
    daily = _minutes_to_daily(m)

    def pause_sell(**kw):
        res = run_backtest("buy_and_hold", daily, instrument, starting_capital=10_000, risk_profile="aggressive",
                           half_spread=0.0, **kw)
        assert [e["kind"] for e in res.risk_events][:1] == ["risk_pause"]
        sells = res.fills[res.fills["side"] == "SELL"]
        return float(sells.iloc[0]["avg_px"])

    # Once a day: the 8% daily-loss limit is noticed at the close, after a 40% fall at a 50% position.
    assert pause_sell() == pytest.approx(60.0, abs=0.5)
    # Every minute: noticed as the loss passes 8%, about a 16% fall at a 50% position.
    px = pause_sell(exec_prices=m, exec_minutes=1)
    assert 82.0 < px < 85.0


def test_a_buy_takes_at_most_a_quarter_of_what_trades(prices, instrument):
    """Review round 3, R3-M6: a backtest filled 98 units on a bar that traded 1. Buys are now capped at
    a quarter of an average bar's volume over the last day, in every mode, and the journal says so."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    thin = prices.iloc[:30].copy()
    thin["volume"] = 1.0
    rt = SleeveRuntime.for_backtest(strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                    starting_balance=1_000_000, risk_profile="aggressive")
    run_backtest("buy_and_hold", thin, instrument, runtime=rt)
    (buy,) = [o for o in rt.store.orders(limit=100) if o["side"] == "BUY"]
    assert buy["qty"] == pytest.approx(0.25, abs=1e-8)
    assert buy["signal"]["sized_by"] == "share of the bar's volume"

    deep = prices.iloc[:30].copy()
    deep["volume"] = 1e9
    res = run_backtest("buy_and_hold", deep, instrument, starting_capital=1_000_000, risk_profile="aggressive")
    assert _notional(res.fills.iloc[0]) == pytest.approx(500_000, rel=0.02)  # the profile cap, not volume


def test_the_participation_cap_can_be_set_or_turned_off(instrument):
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies.trend_filter import TrendFilterConfig

    bt = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
    assert TrendFilterConfig(instrument_id=instrument.id, bar_type=bt, assumed_taker_fee=0.008).max_participation == 0.25
    assert TrendFilterConfig(instrument_id=instrument.id, bar_type=bt, assumed_taker_fee=0.008,
                             max_participation=None).max_participation is None
    with pytest.raises(ValueError, match="max_participation"):
        TrendFilterConfig(instrument_id=instrument.id, bar_type=bt, assumed_taker_fee=0.008, max_participation=2)


def _volume_run(prices, instrument, volume, params=None):
    from sleeve_fund.paper.runtime import SleeveRuntime

    bars = prices.iloc[:30].copy()
    bars["volume"] = volume
    rt = SleeveRuntime.for_backtest(strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                    starting_balance=1_000_000, risk_profile="aggressive")
    run_backtest("buy_and_hold", bars, instrument, params=params, runtime=rt)
    buys = [o for o in rt.store.orders(limit=100) if o["side"] == "BUY"]
    return buys, rt.store.events(rt.name, limit=100)


def test_bars_with_no_volume_do_not_silence_the_strategy(prices, instrument):
    """Review round 5, R5-M3: on zero-volume bars the cap allowed nothing, so no entries and no word why.
    With nothing traded there is nothing to take a share of: the cap stands aside, once, and says so.
    The engine makes no market on a bar where nothing traded, and its rejections are journaled too."""
    buys, events = _volume_run(prices, instrument, 0.0)
    assert buys and all(b["signal"]["sized_by"] == "aggressive risk profile cap" for b in buys)
    (note,) = [e for e in events if e["kind"] == "no_volume"]
    assert "volume cap is off" in note["message"]
    assert any(e["kind"] == "order_rejected" and "No market" in e["message"] for e in events)


def test_a_buy_the_volume_cap_blocks_says_so(prices, instrument):
    # A tenth of it rounds to nothing (the venue is shown at least the smallest size where anything traded).
    buys, events = _volume_run(prices, instrument, 1e-8, {"max_participation": 0.1})
    assert buys == []
    (note,) = [e for e in events if e["kind"] == "buy_skipped"]  # said once, not every bar
    assert "share of the bar's volume" in note["message"] and "smallest order" in note["message"]


@pytest.mark.usefixtures("maker_on")
def test_a_halt_on_the_bar_an_entry_fills_cancels_its_stop_and_target(instrument):
    """A maker entry filled and the drawdown halt fired on the same minute, while the entry's stop and
    target were not yet at the venue. The halt sold the position, and the target, still resting, sold it
    again hours later: a short in a cash account and a phantom exit (review round 6, N6-1). This path is
    the reviewer's seed 5 that found it."""
    import numpy as np

    from sleeve_fund.venues import venue

    eth = venue("kraken").instrument("ETH", "USD")
    n = 87 * 1440
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.06 / np.sqrt(1440), 90 * 1440)
    jumps = rng.random(90 * 1440) < 2 / 1440
    r[jumps] += rng.normal(0, 0.06, jumps.sum())
    c = 1000 * np.exp(np.cumsum(r))[:n]
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2024-01-01 00:01", periods=n, freq="1min", tz="UTC")
    m = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995, "close": c,
                      "volume": 50.0}, index=idx)
    daily = m.resample("1D", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    # No spread: the post-only entry rests a tick under the last trade, where the reviewer's run had it.
    res = run_backtest("trend_filter", daily, eth, starting_capital=10_000, exec_prices=m, exec_minutes=1,
                       risk_profile="conservative", bar_minutes=1440, half_spread=0.0,
                       params={"fast": 2, "slow": 3, "stop_loss": 0.04, "take_profit": 0.05, "maker_wait_minutes": 60})
    j = res.journal
    assert [e["kind"] for e in res.risk_events] == ["risk_halt"]
    halt = pd.Timestamp(res.risk_events[0]["ts"])
    orders = {o["intent"]: o for o in j.orders_.values() if pd.Timestamp(o["ts"]) == halt}
    assert orders["risk_halt"]["status"] == "filled"
    # The entry's stop and target (resting since its first slice filled) were cancelled, not left to sell.
    exits = [o for o in j.orders_.values() if o["intent"] in ("stop_loss", "take_profit")
             and pd.Timestamp(o["ts"]) <= halt and o["status"] not in ("filled", "rejected")]
    assert exits and all(o["status"] == "canceled" for o in exits)
    pos = 0.0
    for f in j.fills_:
        pos += f["qty"] if f["side"] == "BUY" else -f["qty"]
        assert pos > -1e-9, f"sold more than was held at {f['ts']}"
    assert abs(pos) < 1e-9 and res.equity.index[-1] > halt


@pytest.mark.strategy_errors
def test_a_failing_risk_check_or_bar_reaches_the_result_and_raises_one_alert(prices, instrument, monkeypatch):
    """Review round 8, M8-3: a failing tick or bar showed no banner on the backtest page; with the tick
    failing, a -50% crash ran with no halt and read as a result. Every failure is counted into the
    result, the page refuses to stand behind it, and the journal says so once per handler."""
    from sleeve_fund.dashboard.preview import _errors
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.strategies.buy_and_hold import BuyAndHold

    def down(self, **kw):
        raise RuntimeError("journal down")

    monkeypatch.setattr(SleeveRuntime, "tick", down)
    res = run_backtest("buy_and_hold", prices.iloc[:60], instrument, risk_profile="balanced")
    assert res.handler_error_count >= 59 and res.handler_errors[0] == ("_on_tick", "RuntimeError('journal down')")
    assert [e["kind"] for e in res.journal.events_ if e["level"] == "error"] == ["tick_failed"]
    banner = _errors(res.handler_errors, res.handler_error_count)
    assert f"hit {res.handler_error_count} errors" in banner and "the first handling the risk check: journal down" in banner
    monkeypatch.undo()

    def broken(self, bar):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(BuyAndHold, "want_long", broken)
    res = run_backtest("buy_and_hold", prices.iloc[:30], instrument)
    assert res.handler_error_count == 30 and res.handler_errors[0][0] == "on_bar" and res.fills.empty


def test_the_alert_send_bound_holds_whatever_the_far_end_does():
    """Review round 8, m8-T: the whole-send bound could be lifted and every test still passed (the
    trickle test sets its own)."""
    from sleeve_fund import alerts

    assert 0 < alerts.TOTAL <= 3 * alerts.TIMEOUT
