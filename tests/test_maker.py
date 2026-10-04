"""Maker-first orders: post-only limits that pay the maker fee when filled, finished at market."""

import collections

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


def test_a_partly_filled_maker_entry_is_guarded_by_its_stop_through_the_wait(instrument):
    """Review round 9, M9-4: the backtest rested the stop only once the entry had filled whole, so a
    post-only entry filling in slices had no stop through its wait. Paper guards from the first slice; a
    25% fall inside the wait cost the backtest 3.5% of capital against 1.2% in paper. The stop now rests
    on the first slice and grows with each, and its fill cancels what is left of the entry, as paper's does."""
    day = [10_000.0] * 1440
    fall = [10_000.0 * (1 - 0.025 * k) for k in range(1, 11)]
    m = _minutes(day + [10_000.0] * 4 + fall + [7_500.0] * (2 * 1440), wiggle=1.0, volume=1.0)
    res = _run(m, {"maker_wait_minutes": 15, "stop_loss": 0.02}, instrument=instrument, risk_profile="aggressive",
               bar_minutes=1440, exec_minutes=1)
    j = res.journal
    entry = [f for f in j.fills_ if j.orders_[f["order_id"]]["intent"] == "entry"]
    stops = [f for f in j.fills_ if j.orders_[f["order_id"]]["intent"] == "stop_loss"]
    assert len(entry) >= 2 and stops  # the entry filled in slices, and the stop sold
    wait_ends = pd.Timestamp("2024-01-02 00:15", tz="UTC")
    assert all(pd.Timestamp(f["ts"]) < wait_ends for f in stops)
    held = sum(f["qty"] for f in entry)
    assert sum(f["qty"] for f in stops) == pytest.approx(held)
    assert min(f["price"] for f in stops) > 9_700  # out near the 2% stop, not after the 25% fall
    assert pd.Timestamp(stops[-1]["ts"]) >= max(pd.Timestamp(f["ts"]) for f in entry)  # nothing bought after
    stop_order = j.orders_[stops[0]["order_id"]]
    assert stop_order["qty"] > entry[0]["qty"] and "resized" in stop_order["message"]  # grew with the entry


def test_a_stop_and_target_resized_mid_flight_still_cover_the_whole_position(instrument):
    """An entry slice can fill while its stop and target are still being sent or resized. Those were
    skipped, so the stop sold a few slices and left most of the position with none; and resizing both
    linked orders made each undo the other's size. Every stop or target exit now leaves the book flat."""
    s = np.arange(16 * 3600)
    px = np.round(60_000 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.0004, len(s)))), 1)
    trades = pd.Series(px, index=pd.date_range("2025-10-03", periods=len(s), freq="1s", tz="UTC"))
    bars = trades.resample("1min", closed="left", label="right").ohlc()
    bars["volume"] = 60 * 0.0005  # thin: a post-only entry fills in slices
    dec = bars.resample("15min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    res = run_backtest("trend_filter", dec, instrument,
                       params={"fast": 3, "slow": 8, "maker_wait_minutes": 10, "stop_loss": 0.004, "take_profit": 0.02},
                       starting_capital=10_000, risk_profile="aggressive", bar_minutes=15, exec_prices=bars,
                       exec_minutes=1, half_spread=1e-4)
    j = res.journal
    held, after, exit_moments = 0.0, {}, set()
    for f in sorted(j.fills_, key=lambda f: f["id"]):
        held += f["qty"] if f["side"] == "BUY" else -f["qty"]
        assert held > -1e-9
        after[f["ts"]] = held  # the position once everything at that moment has filled
        if j.orders_[f["order_id"]]["intent"] in ("stop_loss", "take_profit"):
            exit_moments.add(f["ts"])
    # A slice that fills at the moment the stop does (its cancel still in flight) is sold with the rest.
    assert len(exit_moments) >= 3 and all(after[t] == pytest.approx(0, abs=1e-9) for t in exit_moments)
    assert res.handler_error_count == 0
    # Each resize is a command, and the venue matches its resting orders against the bar again on every one:
    # resizing on each slice once took a whole entry from one bar, 50 slices in a minute. At most the four
    # prints of a bar plus one re-match now.
    entry_slices = collections.Counter((f["order_id"], f["ts"]) for f in j.fills_
                                       if j.orders_[f["order_id"]]["order_type"] == "POST-ONLY LIMIT")
    assert entry_slices and max(entry_slices.values()) <= 5, entry_slices.most_common(3)


def test_without_quotes_a_post_only_order_rests_at_the_estimated_bid(instrument):
    """Review round 9, M9-3: paper joins the best bid, while the backtest rested a tick inside the last
    trade, nearer the market, and filled up to 34 points more often. Without quotes the backtest now
    estimates the bid as the last trade less the half spread it charges, in whole ticks."""
    m = _minutes([10_000.0] * (3 * 1440), wiggle=10.0)
    res = _run(m, {"maker_wait_minutes": 15}, instrument=instrument, half_spread=0.0005)
    fill = res.fills.iloc[0]
    assert fill["type"] == "LIMIT" and float(fill["avg_px"]) == pytest.approx(9_995.00)


def test_a_paper_maker_slice_is_charged_as_filled_at_its_limit():
    """Review round 9, M9-3: paper sends a kept post-only order's slices at market; the fee model charges
    each as filled at the order's limit with the maker fee, so the account matches the backtest's fill."""
    from decimal import Decimal

    from nautilus_trader.model import Price, Quantity

    from sleeve_fund.instruments import FeeSchedule, ScheduleFeeModel
    from sleeve_fund.venues import venue

    fm = ScheduleFeeModel(FeeSchedule(Decimal("0.004"), Decimal("0.008")))
    inst = venue("KRAKEN").instrument("BTC", "USD")
    fm.maker_slices = {"B": (Decimal("10000"), True), "S": (Decimal("10000"), False)}
    slice_ = lambda coid: type("O", (), {"client_order_id": coid, "is_post_only": False})()  # noqa: E731
    # A buy filled at 9,990 pays 10 more a unit; a sell filled at 10,010 gives 10 back: both at 10,000 net.
    buy = fm.get_commission(slice_("B"), Quantity(0.1, 8), Price(9_990, 2), inst).as_double()
    sell = fm.get_commission(slice_("S"), Quantity(0.1, 8), Price(10_010, 2), inst).as_double()
    assert buy == pytest.approx(0.1 * 10_000 * 0.004 + 1.0) and sell == pytest.approx(0.1 * 10_000 * 0.004 + 1.0)
    assert 0.1 * 9_990 + buy == pytest.approx(0.1 * 10_000 * 1.004)
    assert 0.1 * 10_010 - sell == pytest.approx(0.1 * 10_000 * 0.996)
