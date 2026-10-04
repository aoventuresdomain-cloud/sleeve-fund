"""Review round 5, R5-M5: the paper runtime fed ticks and the bar backtest fed the same trades as
one-minute bars make the same trades: stops, take-profits, risk-per-trade sizing and the risk guard.

The tick path is the paper sleeve replayed (tests/test_replay.py proves it sends what paper sent);
the bar path is what the backtest page and research run. Paper watches every trade, so its exits sell
at market on the trade that crosses the level; a backtest only sees whole bars, so its exits rest at
the venue and fill at the level. Each exit lands in the same minute, at a price a few basis points
apart, and sizes follow equity, so they drift by as much."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.paper.recorder import Recorder
from sleeve_fund.research.replay import replay
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.venues import venue

START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC, in nanoseconds
SPREAD = 1.0  # a dollar wide, on either side of each trade


def _both(tmp_path, prices, strategy, params, profile):
    """Record a paper session trading one price a second, replay it tick by tick, and backtest the
    same trades as one-minute bars. Returns each path's fills as (side, intent, minute, qty, price)."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    path = tmp_path / "s.jsonl.gz"
    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "parity", "strategy": strategy, "instrument": "BTC/USD",
                           "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000, "risk_profile": profile,
                           "params": params, "max_notional": None, "maker_fee": "0.004", "taker_fee": "0.008",
                           "tick_seconds": 30}}
    rec.start(inst)
    stamps = []
    for s, px in enumerate(np.round(prices, 1)):
        t = START + s * 1_000_000_000
        rec.quote(QuoteTick(inst.id, Price(px - SPREAD / 2, 1), Price(px + SPREAD / 2, 1), Quantity(1, 8),
                            Quantity(1, 8), t, t + 1000))
        rec.trade(TradeTick(inst.id, Price(px, 1), Quantity(0.01, 8), AggressorSide.BUY if s % 2 else AggressorSide.SELL,
                            TradeId(str(s)), t + 2000, t + 3000))
        stamps.append(t + 2000)
    rec.close()

    orders, fills = replay(path, with_fills=True)
    ticks = _by_order(fills, {o["order_id"]: o["intent"] for o in orders})

    trades = pd.Series(np.round(prices, 1), index=pd.to_datetime(stamps, utc=True))
    bars = trades.resample("1min", closed="left", label="right").ohlc()  # stamped at the close, as the engine's
    bars["volume"] = 0.6
    res = run_backtest(strategy, bars, inst, params=params, starting_capital=10_000, risk_profile=profile,
                       bar_minutes=1, half_spread=SPREAD / 2 / float(np.mean(prices)))
    j = res.journal
    return ticks, _by_order(j.fills_, {k: o["intent"] for k, o in j.orders_.items()})


def _by_order(fills, intents):
    """One row per order, in fill order: an order can fill in parts, against trades one at a time."""
    out = {}
    for f in fills:
        side, qty, notional, _ = out.get(f["order_id"], (f["side"], 0.0, 0.0, None))
        out[f["order_id"]] = (side, qty + f["qty"], notional + f["qty"] * f["price"], f["ts"])
    return [(side, intents[k], _minute(ts), qty, notional / qty) for k, (side, qty, notional, ts) in out.items()]


def _minute(ts: datetime) -> datetime:
    """The close of the one-minute bar a fill happened in: a bar fill is stamped at its bar's close."""
    t = pd.Timestamp(ts).tz_convert(timezone.utc)
    return (t if t == t.floor("1min") else t.ceil("1min")).to_pydatetime()


def _same_trades(ticks, bar, price_tol):
    assert [f[:3] for f in ticks] == [f[:3] for f in bar]
    for t, b in zip(ticks, bar):
        assert b[3] == pytest.approx(t[3], rel=2e-3), (t, b)  # sizes follow equity, a few bp apart
        assert b[4] == pytest.approx(t[4], rel=price_tol), (t, b)


def _seconds(minutes):
    return np.arange(minutes * 60)


def test_take_profits_and_risk_sizing_trade_the_same_on_ticks_and_bars(tmp_path):
    s = _seconds(240)  # 4% waves over about 40 minutes: each rally reaches a 3% target
    prices = 60_000 * (1 + 0.04 * np.sin(s / 380) + 0.002 * np.sin(s / 9))
    ticks, bar = _both(tmp_path, prices, "trend_filter",
                       {"fast": 3, "slow": 8, "stop_loss": 0.01, "take_profit": 0.03, "risk_per_trade": 0.005},
                       "aggressive")
    assert sum(f[1] == "take_profit" for f in ticks) >= 5
    assert all(f[1] in ("entry", "exit", "take_profit") for f in ticks)
    _same_trades(ticks, bar, price_tol=1e-3)


def test_stops_trade_the_same_on_ticks_and_bars(tmp_path):
    s = _seconds(240)  # 40-minute cycles: up 2% for 39 minutes, then down 3.2% within one
    saw = (s % 2400) / 2400
    prices = 60_000 * (1 + np.where(saw < 0.975, 0.02 * saw / 0.975, 0.02 - 1.6 * (saw - 0.975)))
    prices *= 1 + 0.001 * np.sin(s / 9)
    ticks, bar = _both(tmp_path, prices, "trend_filter",
                       {"fast": 3, "slow": 8, "stop_loss": 0.01, "take_profit": 0.03, "risk_per_trade": 0.005},
                       "aggressive")
    assert sum(f[1] == "stop_loss" for f in ticks) >= 5
    # Paper sells on the trade through the stop, moving 0.07% a second; the backtest at the stop.
    _same_trades(ticks, bar, price_tol=1e-3)


def test_the_risk_guard_acts_in_the_same_minute_on_ticks_and_bars(tmp_path):
    s = _seconds(120)  # flat for 30 minutes, then down 1% a minute
    prices = 60_000 * np.clip(1 - np.maximum(s - 1800, 0) / 3600 * 0.6, 0.4, 1)
    ticks, bar = _both(tmp_path, prices, "buy_and_hold", {}, "conservative")
    assert [f[1] for f in ticks] == ["entry", "risk_pause"]
    # Paper values the book every 30 s and the backtest at each minute's close: a minute's move apart.
    _same_trades(ticks, bar, price_tol=1e-2)
