"""Maker-first orders: post-only limits that pay the maker fee when filled, finished at market."""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest

MAKER, TAKER = 0.004, 0.008


def _minutes(closes, start="2024-01-01", wiggle=0.0, volume=1_000.0):
    """1-minute bars stamped at their close, as the history store returns them. Deep by default, so an
    order fills whole inside the share of a bar the venue shows (runner.BOOK_SHARE)."""
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range(start, periods=len(c), freq="1min", tz="UTC") + pd.Timedelta("1min")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + wiggle, "low": np.minimum(o, c) - wiggle,
                         "close": c, "volume": volume}, index=pd.DatetimeIndex(idx, name="timestamp"))


def _daily(m):
    g = m.resample("1D", closed="right", label="right")
    return g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def _run(m, params, **kw):
    kw.setdefault("half_spread", 0)  # these test the order mechanics; the spread has its own tests
    return run_backtest("buy_and_hold", _daily(m), kw.pop("instrument"), params, exec_prices=m, **kw)


def _fee(fills):
    return float(str(fills["commissions"].iloc[0][0]).split()[0])


def test_a_post_only_entry_fills_at_its_limit_and_pays_the_maker_fee(instrument):
    m = _minutes([10_000.0] * (3 * 1440), wiggle=1.0)  # trades a dollar either side every minute
    res = _run(m, {"maker_wait_minutes": 15}, instrument=instrument)
    assert len(res.fills) == 1
    fill = res.fills.iloc[0]
    assert fill["type"] == "LIMIT" and fill["liquidity_side"] == "MAKER"
    assert float(fill["avg_px"]) == pytest.approx(9_999.99)  # one tick (a cent) under the last price
    assert _fee(res.fills) == pytest.approx(float(fill["filled_qty"]) * 9_999.99 * MAKER, abs=0.01)
    assert fill["ts_last"] < pd.Timestamp("2024-01-02 00:15", tz="UTC")  # well inside the wait


def test_a_post_only_order_takes_only_a_share_of_what_trades(instrument):
    """Review R2-M3: a post-only order joins the back of the queue, so it can't take all of a thin bar's
    volume just because the price traded through it. Each minute fills at most BOOK_SHARE / 4 of the
    minute's volume per print through the limit; what is left after the wait goes at market."""
    from sleeve_fund.research.runner import BOOK_SHARE

    m = _minutes([10_000.0] * (3 * 1440), wiggle=1.0, volume=1.0)  # 1 a minute; the order is about 1
    res = _run(m, {"maker_wait_minutes": 15}, instrument=instrument)
    maker, market = res.fills.iloc[0], res.fills.iloc[1]
    assert maker["type"] == "LIMIT" and market["type"] == "MARKET"
    per_print = BOOK_SHARE / 4
    # 15 minutes of at most two prints through the limit each; the whole order was nearly a minute's volume.
    assert 0 < float(maker["filled_qty"]) <= 15 * 2 * per_print + 1e-9
    assert float(market["filled_qty"]) > 0.2 * float(maker["filled_qty"])
    assert "not filled within 15 minutes" in list(res.decisions.values())[1]["reason"]


def test_an_unfilled_post_only_order_goes_at_market_after_the_wait(instrument):
    day = [10_000.0] * 1440
    rising = [10_000.0 + i for i in range(1, 2 * 1440 + 1)]  # never trades back down to the limit
    res = _run(_minutes(day + rising), {"maker_wait_minutes": 15}, instrument=instrument)
    assert len(res.fills) == 1
    fill = res.fills.iloc[0]
    assert fill["type"] == "MARKET" and fill["liquidity_side"] == "TAKER"
    assert fill["ts_last"] == pd.Timestamp("2024-01-02 00:15", tz="UTC")
    assert _fee(res.fills) == pytest.approx(float(fill["filled_qty"]) * float(fill["avg_px"]) * TAKER, abs=0.01)
    first, second = res.decisions.values()
    assert first["signal"]["order_type"] == "maker" and second["signal"]["order_type"] == "market"
    assert "not filled within 15 minutes" in second["reason"]


def test_without_minute_data_every_maker_order_is_charged_the_taker_fee(prices, instrument):
    # Nothing trades between daily bars, so the order can't be shown to fill: the honest answer.
    a = run_backtest("buy_and_hold", prices, instrument)
    b = run_backtest("buy_and_hold", prices, instrument, {"maker_wait_minutes": 15})
    assert b.fees_paid == pytest.approx(a.fees_paid, rel=1e-3)
    assert list(b.fills["liquidity_side"]) == ["TAKER"]


def test_a_cancel_that_is_not_for_time_does_not_trade(instrument):
    # The run ends five minutes after the decision: shutting down cancels the order, and a cancel
    # this strategy did not make for time (a halt, a stop, a shutdown) must not send a market order.
    day = [10_000.0] * 1440
    res = _run(_minutes(day + [10_000.0 + i for i in range(1, 6)]), {"maker_wait_minutes": 15}, instrument=instrument)
    assert len(res.decisions) == 1 and next(iter(res.decisions.values()))["signal"]["order_type"] == "maker"
    assert res.fills.empty or (res.fills["filled_qty"].astype(float) == 0).all()


def test_stop_loss_exits_are_never_maker_orders(instrument):
    day = [10_000.0] * 1440
    falling = [10_000.0 * (1 - 0.0001 * i) for i in range(1, 2 * 1440)]
    res = _run(_minutes(day + falling, wiggle=1.0), {"maker_wait_minutes": 15, "stop_loss": 0.05},
               instrument=instrument)
    sells = res.fills[res.fills["side"] == "SELL"]
    assert len(sells) == 1 and sells.iloc[0]["liquidity_side"] == "TAKER"
    buys = res.fills[res.fills["side"] == "BUY"]
    assert list(buys["liquidity_side"]) == ["MAKER"]


@pytest.mark.parametrize("wait,match", [(0, "at least 1"), (1440, "shorter than one bar"), (2.5, "whole number")])
def test_the_wait_must_fit_inside_a_bar(prices, instrument, wait, match):
    with pytest.raises(ValueError, match=match):
        run_backtest("buy_and_hold", prices, instrument, {"maker_wait_minutes": wait})


def test_a_stop_the_price_is_already_through_sells_at_market(instrument):
    """A post-only buy fills at its limit as the price gaps down through it; the stop would rest
    above the price, so the venue refuses it and its linked target with it. The position must not
    be left unprotected until the next decision: it is sold at market on the spot, as paper would."""
    k = 1440 + 5  # five minutes after the first decision's post-only buy
    m = _minutes([10_000.0] * k + [9_000.0] * (2 * 1440))
    m.iloc[k, m.columns.get_loc("open")] = m.iloc[k, m.columns.get_loc("high")] = 9_000.0
    res = _run(m, {"maker_wait_minutes": 15, "stop_loss": 0.04, "take_profit": 0.1}, instrument=instrument,
               risk_profile="aggressive")
    orders = sorted(res.journal.orders_.values(), key=lambda o: (o["ts"], o["id"]))
    assert [(o["intent"], o["order_type"], o["status"]) for o in orders] == [
        ("entry", "POST-ONLY LIMIT", "filled"), ("stop_loss", "STOP", "rejected"),
        ("take_profit", "LIMIT", "rejected"), ("stop_loss", "MARKET", "filled")]
    sell = orders[-1]
    assert sell["ts"] == orders[1]["ts"]  # the same minute, not the next decision
    assert "already through the 9,599.99 stop" in sell["reason"]
    assert sum(f["qty"] * (1 if f["side"] == "BUY" else -1) for f in res.journal.fills_) == 0
    assert any(e["kind"] == "stop_rejected" for e in res.journal.events_)
