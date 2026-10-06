"""The strategy sprint's two new models (5 Oct 2026): the 4-hour dip-buy (M4) and the Donchian ensemble (A1)."""

import pytest
from nautilus_trader.model import BarType, InstrumentId

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies import REGISTRY
from test_backtest import _path


def _dip(**params):
    from sleeve_fund.strategies.dip_buy import DipBuy, DipBuyConfig

    cfg = DipBuyConfig(instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"),
                       bar_type=BarType.from_str("BTC/USD.KRAKEN-4-HOUR-LAST-EXTERNAL"), assumed_taker_fee=0.008,
                       **params)
    return DipBuy(cfg)


def _donchian(**params):
    from sleeve_fund.strategies.donchian import Donchian, DonchianConfig

    cfg = DonchianConfig(instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"),
                         bar_type=BarType.from_str("BTC/USD.KRAKEN-1-DAY-LAST-EXTERNAL"), assumed_taker_fee=0.008,
                         **params)
    return Donchian(cfg)


def test_both_models_are_in_the_library():
    assert {"dip_buy", "donchian"} <= set(REGISTRY)


def test_dip_buy_buys_a_deep_dip_only_in_a_daily_up_trend_and_sells_the_bounce_to_the_short_average():
    s = _dip()
    # close, rsi, atr, 24h high, 24h low, 6-bar average, regime
    assert s.target_side(95.0, 4.0, 2.0, 100.0, 94.0, 98.0, 1) == 1  # RSI 4 <= 10 and 5 below the high >= 1 ATR
    assert "long until the 6-bar average" in s._why[0]
    assert s.target_side(96.0, 20.0, 2.0, 100.0, 94.0, 97.5, 1) == 1  # still under the average: held
    assert s.target_side(98.0, 60.0, 2.0, 100.0, 94.0, 97.0, 1) == 0  # reached the average: out
    assert "reached the 6-bar average" in s._why[0]
    assert s.target_side(95.0, 4.0, 2.0, 100.0, 94.0, 98.0, 0) == 0  # no daily trend: no dip-buy
    assert s.target_side(95.0, 4.0, 2.0, 100.0, 94.0, 98.0, -1) == 0  # a down-trend never buys the dip
    assert s.target_side(99.5, 4.0, 2.0, 100.0, 94.0, 98.0, 1) == 0  # only 0.25 ATR below the high: not deep enough
    assert s.target_side(95.0, 4.0, 2.0, 100.0, 94.0, 98.0, None) == 0
    assert "still filling" in s._why[0]


def test_dip_buy_shorts_a_rally_in_a_daily_down_trend_and_ends_a_leg_on_its_time_stop():
    s = _dip(time_stop_bars=3)
    assert s.target_side(105.0, 96.0, 2.0, 106.0, 100.0, 102.0, -1) == -1  # RSI 96 >= 90, 5 above the low
    assert [s.target_side(104.0, 80.0, 2.0, 106.0, 100.0, 102.0, -1) for _ in range(3)] == [-1, -1, 0]
    assert "time stop" in s._why[0]


def test_dip_buy_reads_the_daily_regime_from_closed_days_only():
    s = _dip(trend_sma_days=2, trend_ema_days=2)
    s.day_sma.update_raw(100.0)
    s.day_ema.update_raw(100.0)
    s._ema_prev = s.day_ema.value
    s.day_sma.update_raw(110.0)
    s.day_ema.update_raw(110.0)
    s._day_close = 110.0
    assert s.regime() == 1  # above its 2-day average, and the average rose
    s._ema_prev, s._day_close = s.day_ema.value, 90.0
    s.day_sma.update_raw(90.0)
    s.day_ema.update_raw(90.0)
    assert s.regime() == -1


def test_dip_buy_warm_up_fills_the_daily_trend_and_settles_the_daily_ema():
    from sleeve_fund.strategies.dip_buy import DipBuy

    # The 20-day EMA settles over 200 days (ten lengths), past the 100-day average: 201 days of 4-hour bars.
    assert DipBuy.warmup_needed({"trend_sma_days": 100, "trend_ema_days": 20}, 240) == 201 * 6


def test_dip_buy_refuses_settings_that_make_no_sense():
    with pytest.raises(ValueError, match="rsi_entry"):
        _dip(rsi_entry=60.0)
    with pytest.raises(ValueError, match="time_stop_bars"):
        _dip(time_stop_bars=-1)


def test_donchian_thirds_enter_on_their_breakouts_and_leave_at_their_half_length_lows():
    s = _donchian(lookbacks="2,4", vol_lookback_days=10)

    class _Bar:
        def __init__(self, c):
            from nautilus_trader.model import Price

            self.close = Price(c, 2)

    for c in [100, 101, 102, 101, 100, 99, 100, 101, 102, 101, 100]:
        s.update_indicators(_Bar(float(c)))
    assert s._on == {2: False, 4: False}  # the last close, 100, broke both half-length lows
    s.update_indicators(_Bar(103.0))  # above the last 2 and the last 4 closes: both on
    assert s._on == {2: True, 4: True}
    s.update_indicators(_Bar(102.5))  # 2-day sub-model's half length is 1 day: below 103, out; 4-day holds
    assert s._on == {2: False, 4: True}
    s.update_indicators(_Bar(102.0))  # below the last 2 closes (103, 102.5): the 4-day third is out too
    assert s._on == {2: False, 4: False}


def test_donchian_warm_up_reaches_back_to_a_breakout_that_is_still_on():
    """Review round 13, E13-5: a third stays on until its half-length low breaks, which can be long after its
    breakout. Warmed up over one lookback (101 days), a restart found the 100-day third off while the run over all
    history held it on. The warm-up now covers four lookbacks, so it sees the breakout."""
    from nautilus_trader.model import Price

    from sleeve_fund.strategies.donchian import Donchian

    class _Bar:
        def __init__(self, c):
            self.close = Price(c, 2)

    # A rise to 250 by day 150 (every third breaks out), then 250 days between 240 and 250: no new 100-day high,
    # and never below the 100-day third's 50-day low, so that third stays on.
    closes = [100 + i for i in range(151)] + [250 - 10 * (i % 2) for i in range(250)]
    full = _donchian()
    for c in closes:
        full.update_indicators(_Bar(float(c)))
    assert full._on[100]
    need = Donchian.warmup_needed({}, 1440)
    assert need == 401
    for n in (101, need):  # the old warm-up, then the new one, both ending on the last day
        warm = _donchian()
        for c in closes[-n:]:
            warm.update_indicators(_Bar(float(c)))
        assert (warm._on == full._on) is (n == need)


def test_donchian_sizes_the_share_of_thirds_long_to_the_volatility_target():
    s = _donchian(lookbacks="2,4", vol_target=0.25, vol_lookback_days=10)
    s._closes.extend([100.0] * 5)
    s._on = {2: True, 4: False}
    s._vol = 0.5
    assert s.target_weight(None) == pytest.approx(0.5 * 0.25 / 0.5)  # half the ensemble, half the size
    assert "Long on the 2-day breakout" in s._why[0]
    s._vol = 0.1
    assert s.target_weight(None) == pytest.approx(0.5)  # never more than all the capital on spot


def test_donchian_is_long_only_and_daily():
    with pytest.raises(ValueError, match="long only"):
        _donchian(market="perp", allow_short=True)
    from sleeve_fund.strategies.donchian import DonchianConfig

    with pytest.raises(ValueError, match="daily"):
        DonchianConfig(instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"),
                       bar_type=BarType.from_str("BTC/USD.KRAKEN-4-HOUR-LAST-EXTERNAL"), assumed_taker_fee=0.008)


def test_donchian_trades_a_trend_in_a_backtest(prices, instrument):
    closes = [100.0] * 120 + [100 + i for i in range(1, 80)] + [179 - 2 * i for i in range(1, 60)]
    res = run_backtest("donchian", _path(prices, closes), instrument, {"vol_lookback_days": 20}, half_spread=0)
    sides = list(res.fills.sort_values("ts_last")["side"])
    assert sides and sides[0] == "BUY" and sides[-1] == "SELL" and not res.handler_errors


def test_dip_buy_runs_in_a_backtest_without_errors(prices, instrument):
    res = run_backtest("dip_buy", prices.iloc[:400], instrument, {"trend_sma_days": 20, "trend_ema_days": 5},
                       half_spread=0)
    assert not res.handler_errors
