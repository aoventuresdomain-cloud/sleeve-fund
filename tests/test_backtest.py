import pytest
import pandas as pd

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
    # Deep enough that the orders here fill whole within the share of each bar the venue shows
    # (runner.BOOK_SHARE): these tests are about prices and levels, not liquidity.
    df["volume"] = 1e6
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
    cost = float(instrument.taker_fee) + res.half_spread  # each leg: the taker fee and half the spread
    loss_at_stop = 0.05 + cost + 0.95 * cost
    assert notional == pytest.approx(10_000 * 0.01 / loss_at_stop, rel=0.03)  # ~1,500, not the whole sleeve


def test_a_stopped_trade_loses_its_risk_budget_with_costs_and_reads_minus_one_r(prices, instrument):
    """Round 4, R4-M7: a 6% stop sized to lose 1% lost 1.3%, because sizing left out fees and spread."""
    from sleeve_fund.dashboard import trading

    closes = [10_000.0] * 10 + [10_000.0 * 0.99**i for i in range(1, 30)]
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument,
                       {"stop_loss": 0.06, "risk_per_trade": 0.01, "take_profit": 0.18}, starting_capital=10_000,
                       risk_profile="aggressive")
    assert len(res.fills) == 2
    assert res.equity.iloc[-1] == pytest.approx(10_000 * 0.99, abs=10_000 * 0.0006)  # 1% lost, give or take
    j = res.journal
    (trip,) = trading.trips(j.fills(limit=10), j.events(limit=100), {o["order_id"]: o for o in j.orders()})
    assert trip["r"] == pytest.approx(-1.0, abs=0.06)
    assert 2.0 < trip["planned_r"] < 3.0  # an 18% target is under 3R once costs come off both ends


@pytest.mark.parametrize("bad", [{"stop_loss": 0}, {"stop_loss": 0.9}, {"risk_per_trade": 0.01}, {"take_profit": -1}])
def test_bad_exit_settings_rejected(prices, instrument, bad):
    with pytest.raises(ValueError):
        run_backtest("buy_and_hold", prices.iloc[:20], instrument, bad)


def test_multi_indicator_example_enters_on_all_conditions_and_trails_out(prices, instrument):
    from sleeve_fund.research.metrics import round_trips

    # Long uptrend, a sharp dip whose last day has a volume spike, a rebound, then a slide.
    closes = [10_000.0 * 1.005**i for i in range(260)]
    for _ in range(6):
        closes.append(closes[-1] * 0.96)
    closes += [closes[-1] * 1.01**i for i in range(1, 15)] + [closes[-1] * 1.01**14 * 0.97**i for i in range(1, 15)]
    df = _path(prices, closes)
    df["volume"] = 1_000.0
    df.iloc[265, df.columns.get_loc("volume")] = 5_000.0
    res = run_backtest("rsi_pullback", df, instrument, {"rsi_entry": 35, "atr_mult": 2.0})
    assert len(res.fills) == 2
    first = res.fills.sort_values("ts_last")["ts_last"].iloc[0]
    assert first >= df.index[265]  # not before the volume spike
    assert len(round_trips(res.fills)) == 1
    df["volume"] = 1_000.0  # same prices, no volume spike: the entry must not fire
    assert run_backtest("rsi_pullback", df, instrument, {"rsi_entry": 35, "atr_mult": 2.0}).fills.empty


def test_stop_fills_at_its_level_inside_the_bar(prices, instrument):
    # The fall from 100 to 90 happens inside one daily bar; a close-only check would sell at 90.
    closes = [100.0] * 10 + [90.0] * 5
    res = run_backtest("buy_and_hold", _path(prices, closes), instrument, {"stop_loss": 0.05}, half_spread=0)
    sells = res.fills[res.fills["side"] == "SELL"]
    assert len(sells) == 1
    assert float(sells["avg_px"].iloc[0]) == pytest.approx(95.0)
    assert res.decisions[sells.index[0]]["intent"] == "stop_loss"


def test_stop_fills_at_the_open_when_price_gaps_through(prices, instrument):
    df = _path(prices, [100.0] * 10 + [80.0] * 5)
    df.iloc[10, df.columns.get_loc("open")] = 85.0  # opened below the 95 stop
    df.iloc[10, df.columns.get_loc("high")] = 85.0
    res = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.05}, half_spread=0)
    sells = res.fills[res.fills["side"] == "SELL"]
    assert float(sells["avg_px"].iloc[0]) == pytest.approx(85.0)


def test_stop_wins_when_one_bar_touches_both(prices, instrument):
    df = _path(prices, [100.0] * 10 + [100.0] * 5)
    df.iloc[10, df.columns.get_loc("high")] = 115.0
    df.iloc[10, df.columns.get_loc("low")] = 90.0
    res = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.05, "take_profit": 0.10}, half_spread=0)
    sells = res.fills[res.fills["side"] == "SELL"]
    assert len(sells) == 1 and float(sells["avg_px"].iloc[0]) == pytest.approx(95.0)


def test_take_profit_fills_at_its_level_not_the_close(prices, instrument):
    """Review R5-M1: the target used to trigger on the high and sell at the close, anywhere from -0.56R
    to +2.80R against a planned +0.62R. It now rests as a limit and fills at its level."""
    df = _path(prices, [100.0] * 10 + [104.0] * 5)
    df.iloc[10, df.columns.get_loc("high")] = 112.0
    res = run_backtest("buy_and_hold", df, instrument, {"take_profit": 0.10}, half_spread=0)
    sells = res.fills[res.fills["side"] == "SELL"]
    assert len(sells) == 1 and float(sells["avg_px"].iloc[0]) == pytest.approx(110.0)
    d = res.decisions[sells.index[0]]
    assert d["intent"] == "take_profit" and "resting sell at 110" in d["reason"]


def test_take_profit_is_never_credited_more_than_its_level(prices, instrument):
    # Opens above the target: a resting limit would get the open, but bars can't show the queue, so
    # the backtest gives only the level, and the linked stop is cancelled.
    df = _path(prices, [100.0] * 10 + [120.0] * 5)
    df.iloc[10, df.columns.get_loc("open")] = 118.0
    df.iloc[10, df.columns.get_loc("low")] = 118.0
    res = run_backtest("buy_and_hold", df, instrument, {"take_profit": 0.10, "stop_loss": 0.05}, half_spread=0)
    sells = res.fills[res.fills["side"] == "SELL"]
    assert len(sells) == 1 and float(sells["avg_px"].iloc[0]) == pytest.approx(110.0)


def test_a_target_that_cannot_cover_its_costs_is_refused(prices, instrument):
    with pytest.raises(ValueError, match="round trip"):
        run_backtest("buy_and_hold", prices.iloc[:20], instrument, {"take_profit": 0.012, "stop_loss": 0.004})


def test_signal_exit_cancels_the_resting_stop_first(prices, instrument):
    res = run_backtest("trend_filter", prices, instrument, {"fast": 20, "slow": 100, "stop_loss": 0.30})
    buys, sells = (res.fills[res.fills["side"] == s] for s in ("BUY", "SELL"))
    assert len(buys) >= 2 and len(sells) >= len(buys) - 1  # every exit went through despite the stop


def test_position_cap_sizes_like_the_risk_profile(prices, instrument):
    res = run_backtest("buy_and_hold", _path(prices, [100.0] * 20), instrument, {"position_cap_pct": 0.33},
                       starting_capital=10_000)
    notional = float(res.fills["filled_qty"].iloc[0]) * float(res.fills["avg_px"].iloc[0])
    assert notional == pytest.approx(3_300, rel=0.02)
    assert res.decisions[res.fills.index[0]]["signal"]["sized_by"] == "risk profile cap"


def test_vol_target_rebalances_part_of_the_position_with_reasons(prices, instrument):
    res = run_backtest("trend_filter", prices, instrument, {"fast": 10, "slow": 30, "ema": 1, "vol_target": 0.3},
                       starting_capital=100_000)
    intents = [res.decisions[i]["intent"] for i in res.fills.index]
    assert "rebalance" in intents and intents[0] == "entry"
    reb = next(res.decisions[i] for i in res.fills.index if res.decisions[i]["intent"] == "rebalance")
    assert {"from_weight", "to_weight", "volatility"} <= set(reb["signal"]) and "hold" in reb["reason"]
    assert 0 < res.exposure.max() <= 1.0 and res.exposure[res.exposure > 0].mean() < 0.9  # not all in


def test_rebalancing_cannot_be_combined_with_stops(prices, instrument):
    with pytest.raises(ValueError, match="all-or-nothing"):
        run_backtest("trend_filter", prices.iloc[:50], instrument, {"vol_target": 0.3, "stop_loss": 0.05})


def test_any_bar_length_backtests(instrument):
    from sleeve_fund.data import synthetic_ohlcv

    four_hourly = synthetic_ohlcv(days=600, seed=5, vol=0.03)
    four_hourly.index = pd.date_range("2024-01-01 04:00", periods=len(four_hourly), freq="4h", tz="UTC")
    res = run_backtest("trend_filter", four_hourly, instrument, {"fast": 6, "slow": 30}, bar_minutes=240)
    assert len(res.fills) > 2 and res.equity.index.equals(four_hourly.index)


def _ranged(prices, closes, half_range):
    """A path whose every bar spans close +/- half_range, so its average true range is 2 x half_range."""
    df = _path(prices, closes)
    df["open"] = df["close"]
    df["high"], df["low"] = df["close"] + half_range, df["close"] - half_range
    return df


def test_an_atr_stop_is_set_from_the_market_at_entry(prices, instrument):
    """Review round 7: the PM asked for a volatility stop. 2 ATRs on bars spanning 100 +/- 1 is 4 below
    the entry; the entry waits until the ATR has its 14 bars. A 2R target pays 2R after costs (round 8)."""
    df = _ranged(prices, [100.0] * 30 + [93.0] * 5, 1.0)
    df.iloc[30, df.columns.get_loc("open")] = df.iloc[30, df.columns.get_loc("high")] = 100.0  # falls inside bar 30
    res = run_backtest("buy_and_hold", df, instrument, {"stop_atr": 2.0, "take_profit_r": 2.0}, half_spread=0)
    buys, sells = res.fills[res.fills["side"] == "BUY"], res.fills[res.fills["side"] == "SELL"]
    assert len(buys) == 1 and buys["ts_last"].iloc[0] >= df.index[13]  # not before 14 bars of range
    assert len(sells) == 1 and float(sells["avg_px"].iloc[0]) == pytest.approx(96.0)
    d = res.decisions[sells.index[0]]
    assert d["intent"] == "stop_loss" and "2 x the 14-bar simple average true range" in d["reason"]
    entry = res.decisions[buys.index[0]]["signal"]
    # 1R = 4% + 0.8% in + 0.8% of 96% out = 5.568%; 2R after costs needs (2 x 5.568% + 1.6%) / 0.992 = 12.84%.
    assert entry["stop_frac"] == pytest.approx(0.04) and entry["tp_frac"] == pytest.approx(0.128387, abs=1e-5)
    assert entry["planned_r"] == pytest.approx(2.0)


def test_a_target_in_r_pays_that_r_after_costs(prices, instrument):
    """Review round 8 (R8-M1): "2R" realised about +0.7R because it was 2 stop distances before costs.
    Now the target sits where a hit nets twice what a stop-out loses, fees included."""
    df = _path(prices, [100.0] * 10 + [104.0] * 5)
    df.iloc[10, df.columns.get_loc("high")] = 116.0
    res = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.05, "take_profit_r": 2.0}, half_spread=0)
    buys, sells = res.fills[res.fills["side"] == "BUY"], res.fills[res.fills["side"] == "SELL"]
    assert len(sells) == 1 and float(sells["avg_px"].iloc[0]) == pytest.approx(114.839, abs=0.01)
    assert "2R after costs" in res.decisions[sells.index[0]]["reason"]
    entry, out = float(buys["avg_px"].iloc[0]), float(sells["avg_px"].iloc[0])
    net = out * (1 - 0.008) - entry * (1 + 0.008)
    loss = entry * (1 + 0.008) - entry * 0.95 * (1 - 0.008)
    assert net / loss == pytest.approx(2.0, abs=0.01)


def test_a_swing_low_counts_the_bar_the_entry_decides_on(prices, instrument):
    """Review round 8, m8-T: the bar the entry decides on has closed, so its low is one of the last 10
    (no look-ahead in using it); leaving it out set the stop under an older, higher low."""
    df = _ranged(prices, [100.0] * 20 + [90.0] * 5, 1.0)
    df.iloc[5, df.columns.get_loc("low")] = 97.0
    df.iloc[9, df.columns.get_loc("low")] = 96.0  # the tenth bar: the entry decides on its close
    df.iloc[20, df.columns.get_loc("open")] = df.iloc[20, df.columns.get_loc("high")] = 100.0
    res = run_backtest("buy_and_hold", df, instrument, {"stop_swing_bars": 10}, half_spread=0)
    buys, sells = res.fills[res.fills["side"] == "BUY"], res.fills[res.fills["side"] == "SELL"]
    assert res.decisions[buys.index[0]]["signal"]["stop_basis"].startswith("at the lowest low of the last 10 bars (96)")
    assert len(sells) == 1 and float(sells["avg_px"].iloc[0]) == pytest.approx(96.0)


def test_a_swing_low_stop_sits_under_the_recent_low(prices, instrument):
    df = _ranged(prices, [100.0] * 20 + [90.0] * 5, 1.0)
    df.iloc[5, df.columns.get_loc("low")] = 97.0  # the lowest low of the first 10 bars
    df.iloc[20, df.columns.get_loc("open")] = df.iloc[20, df.columns.get_loc("high")] = 100.0
    res = run_backtest("buy_and_hold", df, instrument, {"stop_swing_bars": 10}, half_spread=0)
    sells = res.fills[res.fills["side"] == "SELL"]
    assert len(sells) == 1 and float(sells["avg_px"].iloc[0]) == pytest.approx(97.0)
    assert "lowest low of the last 10 bars (97)" in res.decisions[sells.index[0]]["reason"]


@pytest.mark.parametrize("bad, why", [
    ({"stop_loss": 0.05, "stop_atr": 2.0}, "one kind of stop"),
    ({"take_profit_r": 2.0}, "needs a stop-loss"),
    ({"stop_loss": 0.05, "take_profit": 0.1, "take_profit_r": 2.0}, "not both"),
    ({"stop_swing_bars": 1}, "whole number of bars"),
    ({"stop_loss": 0.05, "take_profit": 0.01}, "round trip"),  # a 1% target can't cover the costs
])
def test_bad_stop_and_target_combinations_are_refused(prices, instrument, bad, why):
    with pytest.raises(ValueError, match=why):
        run_backtest("buy_and_hold", prices.iloc[:20], instrument, bad)


def test_a_target_in_r_clears_costs_even_on_a_tight_stop(prices, instrument):
    """A 0.5% stop (a quiet market's ATR) at 2R before costs would have been a 1% target that loses on
    every hit (review round 8, R8-M2). After costs, 2R sits far enough out to pay 2R."""
    df = _ranged(prices, [100.0] * 30, 0.125)  # ATR 0.25, so 2 ATRs is 0.5%
    res = run_backtest("buy_and_hold", df, instrument, {"stop_atr": 2.0, "take_profit_r": 2.0}, half_spread=0)
    buys = res.fills[res.fills["side"] == "BUY"]
    entry = res.decisions[buys.index[0]]["signal"]
    assert entry["stop_frac"] == pytest.approx(0.005) and entry["tp_frac"] > 0.05
    assert entry["planned_r"] == pytest.approx(2.0)
    resting = [d for d in res.decisions.values() if d["intent"] in ("stop_loss", "take_profit")]
    assert [d["intent"] for d in resting] == ["stop_loss"]  # the target is judged on each bar (Advisor NA-2)


def test_sub_cent_prices_get_a_fine_enough_price_step(prices):
    """At SHIB-like prices the old 2/4/6 rule gave a step of 5% of the price: 31 distinct closes a year and
    stops filling from -0.36R to -1.61R (review round 10, B10-2). Now the step stays near 0.01% of the price."""
    from sleeve_fund.instruments import history_price_decimals, price_decimals
    from sleeve_fund.venues import venue

    assert [price_decimals(p) for p in (60_000, 2.0, 0.48, 0.0765, 0.005, 0.00002)] == [2, 4, 6, 6, 7, 9]
    shib = prices.copy()
    scale = 2e-5 / prices["close"].median()
    shib[["open", "high", "low", "close"]] = shib[["open", "high", "low", "close"]] * scale
    shib["volume"] = shib["volume"] / scale  # the same dollar volume
    d = history_price_decimals(shib["close"])
    inst = venue("KRAKEN").instrument("SHIB", "USD", price_precision=d)
    res = run_backtest("trend_filter", shib, inst, {"fast": 10, "slow": 30, "stop_loss": 0.03}, half_spread=0)
    rounded = shib["close"].round(d)
    assert rounded.nunique() > 0.9 * shib["close"].nunique()
    assert not res.fills.empty
    with pytest.raises(ValueError, match="can't be tested"):
        history_price_decimals(shib["close"] / 1000)

