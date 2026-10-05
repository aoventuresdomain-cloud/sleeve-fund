"""The PM's two test strategies (4 Oct 2026): their cycles on bar closes, in a backtest."""

import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.rsi_bands import RsiBands, RsiBandsConfig
from test_backtest import _path


def test_ping_pong_buys_first_sells_after_the_rise_and_rebuys_after_the_dip(prices, instrument):
    # Bought on the first close (100); +1% at 101.5 sells; the 0.5% dip from 101.5 is 100.99, met at 100.9;
    # then +1% from 100.9 is 101.909, met at 102.
    closes = [100.0, 100.5, 100.8, 101.5, 101.3, 101.2, 100.9, 101.0, 101.5, 102.0, 102.0]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, half_spread=0)
    fills = res.fills.sort_values("ts_last")
    sides = list(fills["side"])
    assert sides == ["BUY", "SELL", "BUY", "SELL"]
    # Each trade is on the bar whose close decided it.
    idx = list(_path(prices, closes).index)
    assert [idx.index(t) for t in fills["ts_last"]] == [0, 3, 6, 9]
    assert res.fees_paid > 0


def test_ping_pong_holds_through_a_fall_without_a_stop(prices, instrument):
    closes = [100.0, 99.0, 97.0, 95.0, 96.0, 98.0, 99.5]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, half_spread=0)
    assert list(res.fills["side"]) == ["BUY"]


def _bands():
    from nautilus_trader.model import BarType, InstrumentId

    cfg = RsiBandsConfig(instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"),
                         bar_type=BarType.from_str("BTC/USD.KRAKEN-1-MINUTE-LAST-EXTERNAL"), assumed_taker_fee=0.008)
    return RsiBands(cfg)


@pytest.mark.parametrize("path, sides", [
    ([45, 30, 40, 54, 55, 60], [0, 1, 1, 1, 0, 0]),  # long at 30, out at 55
    ([45, 70, 60, 51, 50, 45], [0, -1, -1, -1, 0, 0]),  # short at 70, out at 50
    ([45, 29, 72, 65, 49], [0, 1, -1, -1, 0]),  # a long ending above 70 turns short on the same bar
    ([72, 28, 40], [-1, 1, 1]),  # a short ending below 30 turns long on the same bar
])
def test_rsi_bands_legs(path, sides):
    s = _bands()
    assert [s.target_side(float(r)) for r in path] == sides


def test_rsi_bands_trades_long_only_in_a_backtest(prices, instrument):
    # A slide pushes RSI under 30 (buy), a rebound past 55 sells; a rally past 70 is a short, held flat.
    closes = [100.0] * 20 + [100 - i for i in range(1, 12)] + [89 + 1.5 * i for i in range(1, 15)] + [110.0] * 5
    res = run_backtest("rsi_bands", _path(prices, closes), instrument, {"rsi_period": 5}, half_spread=0)
    assert list(res.fills.sort_values("ts_last")["side"]) == ["BUY", "SELL"]


@pytest.mark.parametrize("bad", [{"long_entry": 60}, {"short_exit": 75}, {"rsi_period": 1}])
def test_rsi_bands_rejects_crossed_bands(prices, instrument, bad):
    with pytest.raises(ValueError):
        run_backtest("rsi_bands", prices.iloc[:20], instrument, bad)


def test_ping_pong_in_the_paper_runtime_trades_the_same_cycle(tmp_path):
    """The paper path (paper runtime, recorded ticks, replayed): buy on the first bar, sell after the
    1% rise, buy again after the 0.5% dip, each with the reason the journal shows."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.paper.recorder import Recorder
    from sleeve_fund.research.replay import replay
    from sleeve_fund.venues import venue
    from test_replay import START

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    path = tmp_path / "pp.jsonl.gz"
    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "ping-pong-test", "strategy": "ping_pong", "instrument": "BTC/USD",
                           "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000,
                           "risk_profile": "balanced", "params": {"rise": 0.01, "dip": 0.005},
                           "maker_fee": "0.004", "taker_fee": "0.008", "tick_seconds": 30}}
    rec.start(inst)
    # Flat for 5 minutes, up 1.5% over 20, down 1.2% over 20, then up 1.5% again over 20.
    legs = [(5, 0.0), (20, 0.015), (20, -0.012), (20, 0.015)]
    px, s = 60_000.0, 0
    for minutes, move in legs:
        step = (1 + move) ** (1 / (minutes * 60))
        for _ in range(minutes * 60):
            px *= step
            t = START + s * 1_000_000_000
            rec.quote(QuoteTick(inst.id, Price(px - 0.5, 1), Price(px + 0.5, 1), Quantity(1, 8), Quantity(1, 8),
                                t, t + 1000))
            rec.trade(TradeTick(inst.id, Price(px, 1), Quantity(0.05, 8),
                                AggressorSide.BUY if s % 2 else AggressorSide.SELL, TradeId(str(s)), t + 2000, t + 3000))
            s += 1
    rec.close()
    orders = replay(path)
    assert [o["side"] for o in orders][:3] == ["BUY", "SELL", "BUY"]
    assert orders[0]["reason"].startswith("Start of the cycle")
    assert "past the 1.0% rise" in orders[1]["reason"]
    assert "past the 0.5% dip" in orders[2]["reason"]
    assert all(o["filled_qty"] > 0 for o in orders[:3])


def _cross(**params):
    from nautilus_trader.model import BarType, InstrumentId

    from sleeve_fund.strategies.rsi_cross import RsiCross, RsiCrossConfig

    cfg = RsiCrossConfig(instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"),
                         bar_type=BarType.from_str("BTC/USD.KRAKEN-15-MINUTE-LAST-INTERNAL"), assumed_taker_fee=0.008,
                         **params)
    return RsiCross(cfg)


def _walk(s, path):
    out, prev = [], None
    for r in path:
        out.append(s.target_side(float(r), prev))
        prev = float(r)
    return out


@pytest.mark.parametrize("path, sides", [
    ([45, 25, 28, 31, 50, 55, 60], [0, 0, 0, 1, 1, 0, 0]),  # long on the cross back above 30, out at 55
    ([45, 30, 40], [0, 0, 0]),  # touching 30 from above is not a cross back
    ([60, 75, 71, 69, 50, 45], [0, 0, 0, -1, -1, 0]),  # short on the cross back below 70, out at 45
    ([45, 25, 31, 29, 31], [0, 0, 1, 1, 1]),  # a dip during the long doesn't end it
])
def test_rsi_cross_enters_on_the_cross_back_and_exits_at_the_band(path, sides):
    assert _walk(_cross(), path) == sides


def test_rsi_cross_time_stop_ends_a_leg_that_never_recovers_and_waits_for_a_fresh_cross():
    s = _cross(time_stop_bars=3)
    assert _walk(s, [25, 31, 40, 40, 40, 40, 31]) == [0, 1, 1, 1, 0, 0, 0]
    assert "time stop" in s._why[0] or "no cross" in s._why[0]
    assert _walk(s, [25, 31]) == [0, 1]  # a new cross starts a new leg


def test_rsi_cross_warm_up_settles_the_rsi_and_fills_the_trend_average():
    from sleeve_fund.strategies.rsi_cross import RsiCross

    assert RsiCross.warmup_needed({"rsi_period": 14}, 15) == 141
    # A 4-hour SMA(50) on 15-minute bars needs 50 x 16 = 800 bars.
    assert RsiCross.warmup_needed({"rsi_period": 14, "trend_sma": 50, "trend_minutes": 240}, 15) == 800


def test_rsi_cross_trend_filter_takes_only_legs_with_the_larger_trend():
    s = _cross(trend_sma=2, trend_minutes=240)
    s.trend.update_raw(100.0)
    s.trend.update_raw(110.0)  # average 105
    s._trend_close = 110.0  # above: longs only
    assert _walk(s, [25, 31]) == [0, 1]
    s = _cross(trend_sma=2, trend_minutes=240)
    s.trend.update_raw(110.0)
    s.trend.update_raw(100.0)
    s._trend_close = 100.0  # below its average: no long
    assert _walk(s, [25, 31]) == [0, 0] and "average: no position" in s._why[0]


def test_rsi_cross_trades_in_a_backtest_with_the_sprint_stop(prices, instrument):
    closes = [100.0] * 30 + [100 - i for i in range(1, 12)] + [89 + 1.5 * i for i in range(1, 15)] + [110.0] * 5
    res = run_backtest("rsi_cross", _path(prices, closes), instrument,
                       {"rsi_period": 5, "stop_atr": 2.0, "atr_bars": 14}, half_spread=0)
    sides = list(res.fills.sort_values("ts_last")["side"])
    assert sides[:2] == ["BUY", "SELL"] and not res.handler_errors
