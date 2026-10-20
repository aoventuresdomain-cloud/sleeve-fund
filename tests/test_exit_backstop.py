"""B43-PRECISION 3(b) (Advisor 20 Oct; a replay feed at 2 dp into a 0.1-tick instrument never filled the exit, and
nothing said so): an exit still open at the close of its 2nd bar (a stop or other protective exit) or 3rd (a target or
the signal's exit), counted in the Strategy's own bars from the first that traded through its price, and at least 2
minutes after that bar opened, raises one ERROR event (sent by the alerts, not a reject, so outside F-RX-2). An alert
only: no order changes. The Advisor's cells 1-10 are marked (cN)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import BarType, OrderSide, OrderType
from test_long_short import PERP, _meta, _record

from sleeve_fund.data import FeedPrecisionError
from sleeve_fund.research.replay import replay
from sleeve_fund.store import Store
from sleeve_fund.strategies import base
from sleeve_fund.strategies.base import EXIT_BACKSTOP_FAILED, EXIT_UNFILLED, LongFlatStrategy

MIN = 60_000_000_000
ONE = "BTC/USD.KRAKEN-1-MINUTE-LAST-EXTERNAL"
QUARTER = "BTC/USD.KRAKEN-15-SECOND-LAST-EXTERNAL"


def _px(v):
    return SimpleNamespace(as_double=lambda: v)


def _order(coid, kind, side, level=None, sent=0):
    return SimpleNamespace(client_order_id=coid, order_type=kind, side=side, ts_init=sent,
                           trigger_price=_px(level), price=_px(level))


def _strat(orders, intents, bars=ONE, inflight=()):
    events = []
    strat = SimpleNamespace(
        runtime=SimpleNamespace(name="s", now=lambda: None, store=SimpleNamespace(
            event=lambda name, level, kind, msg, ts=None: events.append((level, kind, msg)))),
        strategy_id="S", decisions={c: {"intent": i} for c, i in intents.items()}, _unfilled={},
        _cfg=SimpleNamespace(bar_type=bars + "@1-MINUTE-EXTERNAL" if bars == ONE else bars),
        cache=SimpleNamespace(orders_open=lambda **k: list(orders), orders_inflight=lambda **k: list(inflight)))
    return strat, events


def _bar(n, low, high, bars=ONE, step=MIN):
    return SimpleNamespace(bar_type=BarType.from_str(bars), ts_event=n * step, low=_px(low), high=_px(high))


def _feed(strat, bars):
    for b in bars:
        LongFlatStrategy._exit_backstop(strat, b)


def _sell_stop():
    return _order("S1", OrderType.STOP_MARKET, OrderSide.SELL, 59_400.0)


def test_c1_c6_a_sell_stop_open_at_the_close_of_its_second_bar_through_raises_one_error():
    strat, events = _strat([_sell_stop()], {"S1": "stop_loss"})
    _feed(strat, [_bar(1, 59_500.0, 59_600.0)])  # above the trigger: not through, not counted
    _feed(strat, [_bar(2, 59_300.0, 59_450.0)])  # bar 1 through
    assert events == []
    _feed(strat, [_bar(3, 59_500.0, 59_600.0)])  # bar 2 (back above: still counts), 2 minutes from bar 1's open
    [(level, kind, msg)] = events
    assert (level, kind) == ("error", EXIT_UNFILLED) and "stop_loss order S1 (at 59,400)" in msg and "2 bars" in msg
    assert "the bar from 1 Jan 01:01 UK" in msg  # bar 1 opened 00:01 UTC on 1 Jan 1970 (BST then)
    _feed(strat, [_bar(n, 59_000.0, 59_100.0) for n in range(4, 7)])  # c6: bars 3-5 open: no second alert
    assert len(events) == 1


def test_c2_the_stop_filled_during_its_second_bar_raises_nothing():
    open_ = [_sell_stop()]
    strat, events = _strat(open_, {"S1": "stop_loss"})
    _feed(strat, [_bar(1, 59_300.0, 59_450.0)])
    open_.clear()  # filled before the second bar's close reaches the Strategy
    _feed(strat, [_bar(n, 59_300.0, 59_450.0) for n in range(2, 6)])
    assert events == [] and strat._unfilled == {}


def test_c3_a_target_only_touched_never_alerts():
    target = _order("T", OrderType.LIMIT, OrderSide.SELL, 60_600.0)
    strat, events = _strat([target], {"T": "take_profit"})
    _feed(strat, [_bar(n, 60_500.0, 60_600.0) for n in range(1, 6)])
    assert events == []


def test_c4_a_target_traded_through_is_open_at_the_close_of_its_third_bar_raises_one_error():
    target = _order("T", OrderType.LIMIT, OrderSide.SELL, 60_600.0)
    strat, events = _strat([target], {"T": "take_profit"})
    _feed(strat, [_bar(1, 60_500.0, 60_600.1), _bar(2, 60_500.0, 60_550.0)])
    assert events == []
    _feed(strat, [_bar(3, 60_500.0, 60_550.0)])
    assert [e[1] for e in events] == [EXIT_UNFILLED]


@pytest.mark.parametrize("side, low, high, through", [(OrderSide.BUY, 60_600.0, 60_700.0, False),  # a short's, touched
                                                      (OrderSide.BUY, 60_599.9, 60_700.0, True)])
def test_a_shorts_target_counts_only_a_trade_strictly_below_it(side, low, high, through):
    target = _order("T", OrderType.LIMIT, side, 60_600.0)
    strat, events = _strat([target], {"T": "take_profit"})
    _feed(strat, [_bar(n, low, high) for n in range(1, 4)])
    assert [e[1] for e in events] == ([EXIT_UNFILLED] if through else [])


def test_c5_on_15_second_bars_nothing_alerts_before_2_minutes_from_the_first_bar_through():
    strat, events = _strat([_sell_stop()], {"S1": "stop_loss"}, bars=QUARTER)
    q = MIN // 4
    _feed(strat, [_bar(n, 59_300.0, 59_450.0, QUARTER, q) for n in range(1, 3)])  # 2 bars, 30 seconds
    assert events == []
    _feed(strat, [_bar(n, 59_300.0, 59_450.0, QUARTER, q) for n in range(3, 8)])  # bar 7 closes 1:45 after bar 1 opened
    assert events == []
    _feed(strat, [_bar(8, 59_300.0, 59_450.0, QUARTER, q)])  # 2:00
    assert [e[1] for e in events] == [EXIT_UNFILLED]


def test_c7_a_market_exit_counts_from_the_first_bar_after_it_was_sent_and_an_entry_never_counts():
    exit_ = _order("X", OrderType.MARKET, OrderSide.SELL, sent=MIN + 1)
    entry = _order("E", OrderType.MARKET, OrderSide.BUY, sent=0)
    strat, events = _strat([exit_, entry], {"X": "exit", "E": "entry"})
    _feed(strat, [_bar(1, 1.0, 2.0), _bar(2, 1.0, 2.0), _bar(3, 1.0, 2.0)])  # sent in bar 1; bars 2-3 count
    assert events == []
    _feed(strat, [_bar(4, 1.0, 2.0)])  # the 3rd bar after sending
    [(level, kind, msg)] = events
    assert kind == EXIT_UNFILLED and "exit order X (at market)" in msg and set(strat._unfilled) == {"X"}


def test_an_exit_the_venue_never_answered_counts_too():
    exit_ = _order("X", OrderType.MARKET, OrderSide.SELL, sent=0)
    strat, events = _strat([], {"X": "pm_flatten"}, inflight=[exit_])
    _feed(strat, [_bar(n, 1.0, 2.0) for n in range(1, 3)])
    assert [e[1] for e in events] == [EXIT_UNFILLED]


def test_c8_the_alert_changes_nothing_it_only_journals():
    """The Strategy stand-in has no order, position, brake or entry method at all: the backstop can only read the
    cache and write the one event (anything else would raise here)."""
    strat, events = _strat([_sell_stop()], {"S1": "stop_loss"})
    _feed(strat, [_bar(n, 59_300.0, 59_450.0) for n in range(1, 4)])
    assert len(events) == 1
    assert set(vars(strat)) == {"runtime", "strategy_id", "decisions", "_unfilled", "_cfg", "cache"}


def test_c10_a_buy_stop_open_at_the_close_of_its_second_bar_at_or_above_the_trigger_raises_one_error():
    stop = _order("S1", OrderType.STOP_MARKET, OrderSide.BUY, 60_600.0)
    strat, events = _strat([stop], {"S1": "stop_loss"})
    _feed(strat, [_bar(1, 60_500.0, 60_599.9)])  # below: not through
    _feed(strat, [_bar(n, 60_500.0, 60_600.0) for n in range(2, 4)])
    assert [e[1] for e in events] == [EXIT_UNFILLED]


def test_the_same_bar_twice_counts_once_and_other_bars_dont_count():
    strat, events = _strat([_sell_stop()], {"S1": "stop_loss"})
    for _ in range(4):
        _feed(strat, [_bar(1, 59_000.0, 59_100.0)])  # a refill delivers it again
    _feed(strat, [_bar(2, 59_000.0, 59_100.0, QUARTER)])  # not the Strategy's own bars
    assert events == [] and strat._unfilled["S1"]["bars"] == 1


def test_nothing_is_counted_without_a_runtime():
    strat, _ = _strat([_sell_stop()], {"S1": "stop_loss"})
    strat.runtime = None
    _feed(strat, [_bar(n, 59_000.0, 59_100.0) for n in range(1, 5)])
    assert strat._unfilled == {}


# ---- end to end: exits that fill raise nothing (even at the strictest count); one left open by a precision mismatch
# raises the alert in a backtest, and nothing else in the run moves ----------------------------------------------------

UP = [(5, 0.0), (20, 0.015), (10, 0.03), (5, 0.0), (10, -0.03), (10, 0.0)]  # a long, its 1% target filled
DOWN = [(5, 0.0), (20, 0.008), (5, 0.0), (10, -0.03), (10, 0.0)]  # a long, its 1% stop filled
PARAMS = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.01, "take_profit": 0.01, **PERP}


@pytest.fixture
def watched(monkeypatch):
    """The strictest count (one bar, no floor), and a note of whether any exit was open on a bar at all, so a quiet
    run is one the backstop looked at."""
    seen = []
    real = LongFlatStrategy._exit_backstop

    def spy(self, bar):
        real(self, bar)
        seen.append(len(self._unfilled))

    for name in ("EXIT_UNFILLED_BARS", "TARGET_UNFILLED_BARS"):
        monkeypatch.setattr(base, name, 1)
    monkeypatch.setattr(base, "EXIT_UNFILLED_FLOOR_NS", 0)
    monkeypatch.setattr(LongFlatStrategy, "_exit_backstop", spy)
    return seen


@pytest.mark.parametrize("legs, stops", [(UP, False), (DOWN, False), (DOWN, True)])
def test_paper_exits_that_fill_raise_nothing(tmp_path, watched, legs, stops):
    _record(tmp_path / "a.jsonl.gz", _meta(10_000, {**PARAMS, **({"resting_stops": True} if stops else {})}), legs)
    store = Store.in_memory()
    replay(tmp_path / "a.jsonl.gz", with_fills=True, store=store)
    assert [o for o in store.orders("ping-pong-test") if o["intent"] in ("stop_loss", "take_profit")]
    assert not [e for e in store.events("ping-pong-test", limit=10_000) if e["kind"] == EXIT_UNFILLED]
    assert watched and (max(watched) > 0 if stops else True)


def _backtest(legs, monkeypatch=None, mismatch_after=None):
    """ping_pong on 1-minute bars. mismatch_after: from then on the bars carry one decimal more than the instrument's
    0.1 tick (the B-43 replay fault), so the simulated venue matches nothing and the resting stop stays open."""
    from sleeve_fund.research import runner
    from sleeve_fund.venues import venue

    if mismatch_after is not None:
        from nautilus_trader.model import Bar, Price, Quantity

        cut = pd.Timestamp(mismatch_after, tz="UTC").value

        def to_bars(df, instrument, bar_type):
            pp, sp = instrument.price_precision, instrument.size_precision
            return [Bar(bar_type, *(Price(y, pp + (t > cut)) for y in (o, h, lo, c)), Quantity(v, sp), int(t), int(t))
                    for o, h, lo, c, v, t in zip(df["open"], df["high"], df["low"], df["close"], df["volume"],
                                                 df.index.as_unit("ns").asi8)]

        monkeypatch.setattr(runner, "to_bars", to_bars)
    px, out = 60_000.0, []
    for minutes, move in legs:
        out += [px * (1 + move * (i + 1) / minutes) for i in range(minutes)]
        px = out[-1]
    c = np.array(out)
    bars = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": np.maximum(c, np.r_[c[0], c[:-1]]),
                         "low": np.minimum(c, np.r_[c[0], c[:-1]]), "close": c, "volume": 1e6},
                        index=pd.date_range("2026-10-01 00:01", periods=len(c), freq="1min", tz="UTC"))
    return runner.run_backtest("ping_pong", bars, venue("KRAKEN").instrument("BTC", "USD", price_precision=1),
                               params=PARAMS, starting_capital=10_000, risk_profile="aggressive", bar_minutes=1)


def _unfilled(res):
    return [e for e in res.journal.events_ if e["kind"] == EXIT_UNFILLED]


@pytest.mark.parametrize("legs", [UP, DOWN])
def test_backtest_exits_that_fill_raise_nothing(watched, legs):
    res = _backtest(legs)
    assert any(v.get("intent") in ("stop_loss", "take_profit") for v in res.decisions.values())
    assert max(watched) > 0  # the backtest's stop rests at the venue: the backstop saw it open
    assert res.journal.events_ and not _unfilled(res)


def test_c9_c8_a_backtest_stop_left_open_by_a_precision_mismatch_raises_the_alert_and_moves_nothing(monkeypatch):
    from sleeve_fund.research import runner

    with pytest.raises(FeedPrecisionError):  # (a) refuses this feed at load
        _backtest(DOWN, monkeypatch, mismatch_after="2026-10-01 00:30")
    # TEC-C9C8: let it past the load guard, modelling a mismatch that appears after load, which (b) is for
    monkeypatch.setattr(runner, "check_feed_precision", lambda instrument, data: None)
    res = _backtest(DOWN, monkeypatch, mismatch_after="2026-10-01 00:30")  # the long is in; its stop rests
    [e] = _unfilled(res)
    assert e["level"] == "error" and "stop_loss order" in e["message"]
    monkeypatch.setattr(LongFlatStrategy, "_exit_backstop", lambda self, bar: None)
    off = _backtest(DOWN, monkeypatch, mismatch_after="2026-10-01 00:30")
    assert not _unfilled(off)
    cols = ["side", "quantity", "filled_qty", "avg_px", "status", "type"]  # not the run's own ids and clocks
    assert res.fills[cols].equals(off.fills[cols]) and res.equity.equals(off.equity)
    assert res.exposure.equals(off.exposure) and res.decisions == off.decisions
    assert ([(x["kind"], x["message"]) for x in res.journal.events_ if x["kind"] != EXIT_UNFILLED]
            == [(x["kind"], x["message"]) for x in off.journal.events_])


def test_a_backstop_that_fails_never_stops_the_bars_trading(monkeypatch):
    """CR2 F1 (HoE MUST): it runs first on each bar, so a failure in it (a store write, an odd bar) becomes an ERROR
    event and the bar's targets, stops and decision go on as without it."""
    def boom(self, bar):
        raise RuntimeError("store down")

    before = _backtest(DOWN)
    monkeypatch.setattr(LongFlatStrategy, "_exit_backstop", boom)
    after = _backtest(DOWN)
    cols = ["side", "quantity", "filled_qty", "avg_px", "status", "type"]
    assert len(after.fills) > 1 and after.fills[cols].equals(before.fills[cols]) and after.equity.equals(before.equity)
    assert after.handler_error_count == before.handler_error_count == 0
    assert any(f["side"] == "SELL" for _, f in after.fills.iterrows())  # the long's stop went out and filled
    [failed] = [e for e in after.journal.events_ if e["kind"] == EXIT_BACKSTOP_FAILED]  # once, not every bar
    assert failed["level"] == "error" and "store down" in failed["message"]
