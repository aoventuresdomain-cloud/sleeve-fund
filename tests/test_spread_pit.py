"""SPREAD-PIT (Independent Quant Advisor 6 Oct 23:42; Data Architect rulings): the measured half spread is a
point-in-time series. Each measurement is in force from its measured_at (the end of its sampling window) until the
next; equal times go to the later row; before the first, the venue's assumption. A backtest charges each fill the
value in force when its bar opened, never a later measurement, and says which values it charged and from when;
paper refreshes the series as new measurements land."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sleeve_fund import spreads
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.spreads import SpreadQuote, SpreadSeries
from sleeve_fund.store import Store
from test_backtest import _path

UTC = timezone.utc
T0 = datetime(2026, 10, 1, 12, tzinfo=UTC)


def _ns(t: datetime) -> int:
    return int(pd.Timestamp(t).value)


def _measured(half: float, at: datetime) -> SpreadQuote:
    return SpreadQuote(half, "measured", 1_000, at)


def test_a_measurement_is_in_force_from_its_time_until_the_next():
    s = SpreadSeries(((_ns(T0), _measured(0.0002, T0)), (_ns(T0 + timedelta(hours=1)), _measured(0.0003, T0))),
                     SpreadQuote(0.0005, "assumed", 0, None))
    assert s.at(_ns(T0) - 1) == 0.0005  # before the first: the venue's assumption
    assert s.quote_at(_ns(T0) - 1).source == "assumed"
    assert s.at(_ns(T0)) == 0.0002  # in force from its own time
    assert s.at(_ns(T0 + timedelta(minutes=59))) == 0.0002  # never the next one early
    assert s.at(_ns(T0 + timedelta(hours=1))) == 0.0003
    assert s.at(_ns(T0 + timedelta(days=30))) == 0.0003
    assert s.peak(_ns(T0) - 1, _ns(T0 + timedelta(hours=2))) == 0.0005


def test_the_series_reads_the_store_oldest_first_and_the_later_row_wins_a_tie():
    store = Store.in_memory()
    store.record_spread("KRAKEN", "BTC/USD", 0.0003, samples=10, ts=T0 + timedelta(hours=1))
    store.record_spread("KRAKEN", "BTC/USD", 0.0002, samples=10, ts=T0)
    store.record_spread("KRAKEN", "BTC/USD", 0.0004, samples=10, ts=T0)  # same time, written later: it wins
    store.record_spread("KRAKEN", "ETH/USD", 0.0009, samples=10, ts=T0)  # another instrument: not read
    s = spreads.series("KRAKEN", "BTC/USD", store)
    assert [q.half_spread for _, q in s.points] == [0.0004, 0.0003]
    assert s.at(_ns(T0)) == 0.0004 and s.at(_ns(T0) - 1) == s.assumed.half_spread == 0.0005
    assert s.quote_at(_ns(T0)).measured_at == T0  # a naive time from the database is read as UTC
    assert spreads.series("KRAKEN", "SUI/USD", store).points == ()


def test_a_measurement_after_the_run_changes_nothing_in_it(prices, instrument):
    """No look-ahead: a measurement taken after the run ended is never charged in it."""
    closes = [100.0] * 10 + [100.0 * 0.99**i for i in range(1, 30)]
    df = _path(prices, closes)
    later = df.index[-1] + pd.Timedelta(days=1)
    s = SpreadSeries(((later.value, _measured(0.00001, later.to_pydatetime())),), SpreadQuote(0.001, "assumed", 0, None))
    pit = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.05}, half_spread=s)
    flat = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.05}, half_spread=0.001)
    assert list(pit.fills["avg_px"]) == list(flat.fills["avg_px"])
    assert pit.spread_paid == pytest.approx(flat.spread_paid)
    assert pit.spreads_used["measurements"] == 0 and pit.spreads_used["assumed"] == 0.001
    assert "assumed; no measurement was in force" in pit.spread_text


def test_each_fill_pays_the_spread_in_force_when_its_bar_opened(prices, instrument):
    closes = [100.0] * 10 + [100.0 * 0.99**i for i in range(1, 30)]
    df = _path(prices, closes)
    flat = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.05}, half_spread=0.001)
    buy_at, sell_at = flat.fills["ts_last"].iloc[0], flat.fills["ts_last"].iloc[1]
    mid = buy_at + (sell_at - buy_at) / 2  # measured after the buy, in force for the stop's bar
    s = SpreadSeries(((mid.value, _measured(0.002, mid.to_pydatetime())),), SpreadQuote(0.001, "assumed", 0, None))
    res = run_backtest("buy_and_hold", df, instrument, {"stop_loss": 0.05}, half_spread=s)
    buy, sell = (float(p) for p in res.fills["avg_px"])
    assert buy == pytest.approx(100.0 * 1.001)  # the assumption: the measurement came later
    assert float(flat.fills["avg_px"].iloc[0]) == pytest.approx(buy)  # so the same entry and stop level
    # The stop pays the measurement in force by then (both values above the stop's 0.05% floor).
    assert sell == pytest.approx(float(flat.fills["avg_px"].iloc[1]) / 0.999 * 0.998, rel=1e-5)  # to the fee's rounding cent
    used = res.spreads_used
    assert (used["assumed"], used["measurements"], used["at_start"], used["at_end"]) == (0.001, 1, 0.001, 0.002)
    assert used["assumed_until"] == used["measured_from"] == f"{mid:%Y-%m-%d %H:%M} UTC"
    assert res.half_spread == 0.001  # the one in force when the run began
    assert f"assumed until {mid:%Y-%m-%d %H:%M} UTC, then the 1 measurements" in res.spread_text


def test_a_measurement_before_the_run_is_charged_throughout(prices, instrument):
    df = _path(prices, [100.0] * 20)
    early = df.index[0] - pd.Timedelta(days=30)
    s = SpreadSeries(((early.value, _measured(0.0002, early.to_pydatetime())),), SpreadQuote(0.001, "assumed", 0, None))
    res = run_backtest("buy_and_hold", df, instrument, half_spread=s)
    assert float(res.fills["avg_px"].iloc[0]) == pytest.approx(100.02)
    assert res.spreads_used["assumed"] is None and res.spreads_used["measured_from"] == "2018-01-01 00:00 UTC"  # charged from the run's start


def test_a_series_value_outside_the_bounds_is_refused(prices, instrument):
    bad = SpreadSeries(((0, _measured(0.06, T0)),), SpreadQuote(0.0005, "assumed", 0, None))
    with pytest.raises(ValueError, match="outside"):
        run_backtest("buy_and_hold", _path(prices, [100.0] * 20), instrument, half_spread=bad)


def test_the_cost_ladder_widens_every_value_in_the_series():
    s = SpreadSeries(((_ns(T0), _measured(0.0002, T0)),), SpreadQuote(0.0005, "assumed", 0, None)).plus(0.0001)
    assert s.at(_ns(T0) - 1) == pytest.approx(0.0006) and s.at(_ns(T0)) == pytest.approx(0.0003)


def test_paper_reloads_the_series_when_an_hours_measurement_lands(tmp_path):
    from sleeve_fund.paper.runtime import SleeveRuntime

    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000)
    t = [T0]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.spread_loader = lambda: spreads.series("KRAKEN", "BTC/USD", store, strict=True)
    got = []
    for _ in range(190):
        t[0] += timedelta(seconds=20)
        fresh = rt.on_quote(99.99, 100.01, venue="KRAKEN")
        if fresh is not None:
            got.append((t[0], fresh))
    (at, fresh), = got  # once, when the hour's window closed and its measurement was written
    assert fresh.at(_ns(at)) == pytest.approx(0.0001) and fresh.at(_ns(T0)) == fresh.assumed.half_spread


def test_paper_keeps_its_series_when_the_reload_fails(tmp_path):
    from sleeve_fund.paper.runtime import SleeveRuntime

    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000)
    t = [T0]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])

    def broken():
        raise ConnectionError("database away")

    rt.spread_loader = broken
    for _ in range(190):
        t[0] += timedelta(seconds=20)
        assert rt.on_quote(99.99, 100.01, venue="KRAKEN") is None  # the strategy keeps what it has
    assert any(e["kind"] == "spreads_not_reloaded" for e in store.events("s1"))


def test_the_strategy_reads_the_value_in_force_where_it_has_no_quotes():
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    book = SimpleNamespace(_cfg=SimpleNamespace(assumed_half_spread=0.0005), spread_series=None, _bid=None, _ask=None,
                           clock=SimpleNamespace(timestamp_ns=lambda: _ns(T0)))
    book._assumed_spread = LongFlatStrategy._assumed_spread.__get__(book)
    assert LongFlatStrategy._half_spread(book) == 0.0005  # no series: the configured assumption
    book.spread_series = SpreadSeries(((_ns(T0), _measured(0.0002, T0)),), SpreadQuote(0.0005, "assumed", 0, None))
    assert LongFlatStrategy._half_spread(book) == 0.0002  # now
    assert book._assumed_spread(_ns(T0) - 1) == 0.0005  # a replayed minute before the measurement
    book._bid, book._ask = 99.99, 100.01
    assert LongFlatStrategy._half_spread(book) == pytest.approx(0.0001)  # live quotes win


def test_a_book_reset_leaves_the_spreads_untouched(tmp_path):
    """Data Architect 6 Oct: spreads are the instrument's, not a strategy's book: a reset never moves them."""
    from sleeve_fund.supervisor import Supervisor, seed

    store = Store(f"sqlite:///{tmp_path}/t.db")
    seed(store, ["configs/sleeves/ping_pong_test.toml"])
    name = "ping-pong-test"
    s = store.sleeve(name)
    store.record_spread(s.venue or "KRAKEN", s.instrument, 0.0002, samples=500, ts=T0)
    store.record_spread(s.venue or "KRAKEN", s.instrument, 0.0003, samples=500, ts=T0 + timedelta(hours=1))
    before = store.spread_series(s.venue or "KRAKEN", s.instrument)
    store.set_desired_state(name, "stopped")
    store.request_reset(name, "Test finished")
    Supervisor(store, python="true").reset_pending()
    assert store.pending_reset() is None and store.reset_runs()  # the reset went through
    assert store.spread_series(s.venue or "KRAKEN", s.instrument) == before


def test_a_take_profit_books_with_the_spread_in_force_when_it_fills_not_when_it_was_decided(prices, instrument,
                                                                                            monkeypatch):
    """HoE 7 Oct: the target is priced by the fee model at the fill, with the half spread in force then, so a spread
    that changes between the decision and the fill (a new bar's measurement) is the one booked."""
    from decimal import Decimal

    from sleeve_fund.instruments import target_fill_px
    from sleeve_fund.strategies.base import LongFlatStrategy

    df = _path(prices, [100.0] * 10 + [100.0 * 1.01**i for i in range(1, 30)])
    decided = {}
    real = LongFlatStrategy._bar_target

    def then_the_spread_moves(self, bar):
        sent = real(self, bar)
        if sent:  # the market order is out, priced at 0.1%; by its fill the spread in force is 0.3%
            decided["book"] = self.decisions[next(reversed(self.decisions))]["signal"]["book_px"]
            self.fee_model.half_spread = Decimal("0.003")
        return sent

    monkeypatch.setattr(LongFlatStrategy, "_bar_target", then_the_spread_moves)
    res = run_backtest("buy_and_hold", df, instrument, {"take_profit": 0.05}, half_spread=0.001)
    (coid, exit_px) = next((c, float(p)) for c, p, s in zip(res.fills.index, res.fills["avg_px"], res.fills["side"])
                           if s == "SELL")
    level = res.decisions[coid]["signal"]["target_px"]
    assert decided["book"] == pytest.approx(float(target_fill_px(level, True, 0.001)))  # the decision's estimate
    assert exit_px == pytest.approx(float(target_fill_px(level, True, 0.003)), rel=1e-6)  # what it booked
    assert res.decisions[coid]["signal"]["book_px"] == pytest.approx(exit_px, rel=1e-6)  # and journaled
    assert f"booked at {res.decisions[coid]['signal']['book_px']:,.6g}" in res.decisions[coid]["reason"]


def test_the_random_entry_baseline_pays_the_spread_in_force_at_each_bar_as_the_strategy_does():
    """QD 7 Oct: the strategy pays each fill's own spread, so the baseline it must beat is charged from the same
    series, at the bars each trip (the strategy's or a draw's) opens and closes on."""
    import numpy as np

    from sleeve_fund.research.random_entry import Trade, random_entry, random_side

    closes = np.linspace(100.0, 120.0, 40)
    trades, windows = [Trade(5, 15, 1), Trade(20, 30, -1)], [(0, 39)]
    flat = random_entry(closes, trades, windows, 0.001)
    assert random_entry(closes, trades, windows, np.full(40, 0.001)).strategy_return == pytest.approx(
        flat.strategy_return)
    cost = np.where(np.arange(40) < 18, 0.001, 0.004)  # the spread widened from bar 18
    pit = random_entry(closes, trades, windows, cost)
    long_leg = closes[15] / closes[5] - 1 - 0.002
    short_leg = -(closes[30] / closes[20] - 1) - 0.008
    assert pit.strategy_return == pytest.approx((1 + long_leg) * (1 + short_leg) - 1)
    side = random_side(closes, trades, windows, cost)
    assert side.strategy_return == pytest.approx(pit.strategy_return)


def test_series_at_many_matches_at():
    import numpy as np

    s = SpreadSeries(((_ns(T0), _measured(0.0002, T0)), (_ns(T0 + timedelta(hours=1)), _measured(0.0003, T0))),
                     SpreadQuote(0.0005, "assumed", 0, None))
    times = [_ns(T0) - 1, _ns(T0), _ns(T0 + timedelta(minutes=59)), _ns(T0 + timedelta(hours=1)), _ns(T0) + 10**15]
    assert list(s.at_many(np.array(times))) == [s.at(t) for t in times]
