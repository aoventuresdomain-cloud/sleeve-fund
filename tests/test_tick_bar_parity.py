"""Review round 5, R5-M5: the paper runtime fed ticks and the bar backtest fed the same trades as
one-minute bars make the same trades: stops, take-profits, risk-per-trade sizing and the risk guard.

The tick path is the paper sleeve replayed (tests/test_replay.py proves it sends what paper sent);
the bar path is what the backtest page and research run. Paper watches every trade, so its exits sell
at market on the trade that crosses the level; a backtest only sees whole bars, so its exits rest at
the venue and fill at the level. Each exit lands in the same minute, at a price a few basis points
apart, and sizes follow equity, so they drift by as much."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.paper.recorder import Recorder
from sleeve_fund.research.replay import replay
from sleeve_fund.instruments import BOOK_SHARE
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store
from sleeve_fund.venues import venue

START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC, in nanoseconds
SPREAD = 12.0  # $12 wide around each trade: a 1 bp half spread, big enough that dropping it shows


INST = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)


def _record(path, prices, strategy, params, profile, minutes=1, size=0.01):
    """A paper session trading one price a second, `size` each, with the quote following each trade."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "parity", "strategy": strategy, "instrument": "BTC/USD",
                           "bar_spec": f"{minutes}-MINUTE-LAST-INTERNAL", "starting_balance": 10_000,
                           "risk_profile": profile, "params": params, "max_notional": None, "maker_fee": "0.004",
                           "taker_fee": "0.008", "tick_seconds": 30}}
    rec.start(INST)
    stamps = []
    for s, px in enumerate(np.round(prices, 1)):
        t = START + s * 1_000_000_000
        rec.trade(TradeTick(INST.id, Price(px, 1), Quantity(size, 8), AggressorSide.BUY if s % 2 else AggressorSide.SELL,
                            TradeId(str(s)), t, t + 1000))
        # The venue's quote follows each trade, as a live feed's does within milliseconds. (The simulated
        # book sits at the last trade's price until the next quote, so a market order then pays no spread.)
        rec.quote(QuoteTick(INST.id, Price(px - SPREAD / 2, 1), Price(px + SPREAD / 2, 1), Quantity(1, 8),
                            Quantity(1, 8), t + 2000, t + 3000))
        stamps.append(t)
    rec.close()
    return pd.Series(np.round(prices, 1), index=pd.to_datetime(stamps, utc=True))


def _both(tmp_path, prices, strategy, params, profile):
    """Record a paper session trading one price a second, replay it tick by tick, and backtest the
    same trades as one-minute bars. Returns each path's fills as (side, intent, minute, qty, price)."""
    inst = INST
    path = tmp_path / "s.jsonl.gz"
    trades = _record(path, prices, strategy, params, profile)
    orders, fills = replay(path, with_fills=True)
    ticks = _by_order(fills, {o["order_id"]: o["intent"] for o in orders})

    bars = trades.resample("1min", closed="left", label="right").ohlc()  # stamped at the close, as the engine's
    # The backtest shows the venue a share of each bar's volume (BOOK_SHARE); paper's simulated
    # venue fills against each whole trade. Scaled so both see the same trades as liquidity.
    bars["volume"] = 0.6 / BOOK_SHARE
    res = run_backtest(strategy, bars, inst, params=params, starting_capital=10_000, risk_profile=profile,
                       bar_minutes=1, half_spread=SPREAD / 2 / float(prices[0]))
    j = res.journal
    return ticks, _by_order(j.fills_, {k: o["intent"] for k, o in j.orders_.items()})


def _by_order(fills, intents):
    """One row per order, in fill order: an order can fill in parts, against trades one at a time.
    The price is what each unit really cost or brought in, fee included: paper pays the spread in its
    fill price (the ask or the bid) and the backtest with the fee, so only the total compares."""
    out = {}
    for f in fills:
        side, qty, notional, fee, _ = out.get(f["order_id"], (f["side"], 0.0, 0.0, 0.0, None))
        out[f["order_id"]] = (side, qty + f["qty"], notional + f["qty"] * f["price"], fee + f["fee"], f["ts"])
    return [(side, intents[k], _minute(ts), qty, (notional + fee if side == "BUY" else notional - fee) / qty)
            for k, (side, qty, notional, fee, ts) in out.items()]


def _minute(ts: datetime) -> datetime:
    """The close of the one-minute bar a fill happened in: a bar fill is stamped at its bar's close."""
    t = pd.Timestamp(ts).tz_convert(timezone.utc)
    return (t if t == t.floor("1min") else t.ceil("1min")).to_pydatetime()


# How far the backtest's all-in price (fee and spread included) may sit from paper's, by intent, in bp
# (low, high). Signal orders are the same market order on both paths, so only rounding separates them:
# a lost 1 bp spread fails. Paper's exits sell at the bid on the trade through the level and the
# backtest's at the level, so the backtest's stops come out better and its targets worse: measured
# -1.7 to +5.1 bp on stops and -4.7 to -2.1 bp on targets. A stop 5% further away (0.05R, 5 bp here)
# or a target 3% further fails.
TOL_BP = {"entry": (-0.3, 0.3), "exit": (-0.3, 0.3), "stop_loss": (-2.5, 7.0), "take_profit": (-6.0, 1.0)}


def _same_trades(ticks, bar, tol_bp=None):
    assert [f[:3] for f in ticks] == [f[:3] for f in bar]
    tol = {**TOL_BP, **(tol_bp or {})}
    for t, b in zip(ticks, bar):
        assert b[3] == pytest.approx(t[3], rel=2e-3), (t, b)  # sizes follow equity, a few bp apart
        lo, hi = tol[t[1]]
        assert lo <= (b[4] / t[4] - 1) * 1e4 <= hi, (t, b)


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
    _same_trades(ticks, bar)


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
    _same_trades(ticks, bar)


def test_the_risk_guard_acts_in_the_same_minute_on_ticks_and_bars(tmp_path):
    s = _seconds(120)  # flat for 30 minutes, then down 1% a minute
    prices = 60_000 * np.clip(1 - np.maximum(s - 1800, 0) / 3600 * 0.6, 0.4, 1)
    ticks, bar = _both(tmp_path, prices, "buy_and_hold", {}, "conservative")
    assert [f[1] for f in ticks] == ["entry", "risk_pause"]
    # Paper values the book every 30 s and the backtest at each minute's close: a minute's move apart.
    _same_trades(ticks, bar, {"risk_pause": (-100.0, 100.0)})


def _maker_both(tmp_path, size):
    """A maker strategy (post-only entries and exits, a 5-minute wait, 15-minute decisions) on both
    paths: paper replayed tick by tick, and the backtest on the same trades as minute bars with their
    true volume. Returns (paper orders, paper fills, paper events, backtest journal)."""
    s = _seconds(16 * 60)
    prices = 60_000 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.0004, len(s))))
    params = {"fast": 3, "slow": 8, "maker_wait_minutes": 5}
    path = tmp_path / "m.jsonl.gz"
    trades = _record(path, prices, "trend_filter", params, "aggressive", minutes=15, size=size)
    store = Store.in_memory()
    orders, fills = replay(path, with_fills=True, store=store)
    bars = trades.resample("1min", closed="left", label="right").ohlc()
    bars["volume"] = 60 * size
    decisions = bars.resample("15min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    res = run_backtest("trend_filter", decisions, INST, params=params, starting_capital=10_000, risk_profile="aggressive",
                       bar_minutes=15, exec_prices=bars, exec_minutes=1, half_spread=SPREAD / 2 / float(prices[0]))
    return orders, fills, store.events("parity", limit=1000, min_level="warning"), res.journal


def test_maker_orders_trade_the_same_on_ticks_and_bars_where_plenty_trades(tmp_path):
    """Deep trades (a whole unit a second against orders of 0.08): both paths fill every post-only
    order whole, at the maker fee, in the same minute or the next. Paper joins the best bid or ask,
    and the backtest (which has no quotes) one tick inside the last trade, so the backtest's price is
    up to the half spread (1 bp) worse, never better."""
    orders, fills, events, j = _maker_both(tmp_path, size=1.0)
    ticks = _by_order(fills, {o["order_id"]: o["intent"] for o in orders})
    bar = _by_order(j.fills_, {k: o["intent"] for k, o in j.orders_.items()})
    kinds = {o["order_id"]: o["order_type"] for o in orders} | {k: o["order_type"] for k, o in j.orders_.items()}
    assert len(ticks) >= 6 and {kinds[f["order_id"]] for f in [*fills, *j.fills_]} == {"POST-ONLY LIMIT"}
    for f in [*fills, *j.fills_]:  # the maker fee, on both paths
        assert f["fee"] / (f["qty"] * f["price"]) == pytest.approx(0.004, abs=1e-5), f
    assert [f[:2] for f in ticks] == [f[:2] for f in bar]
    for t, b in zip(ticks, bar):
        assert timedelta(0) <= b[2] - t[2] <= timedelta(minutes=1), (t, b)
        assert b[3] == pytest.approx(t[3], rel=2e-3), (t, b)
        worse = (b[4] / t[4] - 1) * 1e4 * (1 if t[0] == "BUY" else -1)  # bp the backtest's price is worse
        assert 0 <= worse <= 1.5, (t, b)


def test_where_little_trades_paper_fills_more_maker_orders_and_says_so(tmp_path):
    """Thin trades (0.0005 a second): the backtest gives each post-only order at most BOOK_SHARE of
    what trades through its price, so most of each entry goes at market after the wait. Paper's
    simulated venue fills them whole once the price is through, which is the known gap: each such
    fill is reported in paper's events with what the backtest would have allowed."""
    orders, fills, events, j = _maker_both(tmp_path, size=0.0005)
    paper_maker = {o["order_id"] for o in orders if o["order_type"] == "POST-ONLY LIMIT"}
    bt_entries = [f for f in j.fills_ if j.orders_[f["order_id"]]["intent"] == "entry"]
    bt_maker = sum(f["qty"] for f in bt_entries if j.orders_[f["order_id"]]["order_type"] == "POST-ONLY LIMIT")
    assert bt_maker < 0.25 * sum(f["qty"] for f in bt_entries)
    assert all(f["order_id"] in paper_maker for f in fills)  # paper: every fill a maker fill
    flagged = [e for e in events if e["kind"] == "maker_fill_above_tape"]
    assert len(flagged) >= len(paper_maker) - 1
    assert "a backtest would fill at most" in flagged[0]["message"] and f"({BOOK_SHARE:.0%})" in flagged[0]["message"]
