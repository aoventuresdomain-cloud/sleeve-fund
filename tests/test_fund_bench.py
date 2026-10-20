"""The fund's benchmark (Advisor B34-BENCHMARK, 20 Oct): one buy-and-hold on the fund's capital, E0 x (1 - fee) x P / P0,
flows bought and sold at their time, funding on a perpetual's long, returns from E0."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.dashboard.fund_bench import BENCH_PAIR, chained_return, fund_hold, hold_values
from sleeve_fund.history import HistoryStore

FEE = 0.00055
T0 = pd.Timestamp("2026-10-10 00:00", tz="UTC")


def _prices(*points):
    return pd.Series([p for _, p in points], index=pd.DatetimeIndex([t for t, _ in points]), dtype=float)


def test_the_first_point_is_above_0_and_is_e0_net_of_one_entry_fee():
    v = hold_values(_prices((T0, 10_000.0), (T0 + timedelta(minutes=1), 10_000.0)), 100_000, FEE, T0)
    assert v.iloc[0] == pytest.approx(100_000 * (1 - FEE))
    assert v.iloc[0] > 0


def test_without_flows_the_hold_is_e0_net_of_the_fee_times_p_over_p0_and_its_return_is_from_e0():
    p = _prices((T0, 10_000.0), (T0 + timedelta(days=1), 11_000.0))
    v = hold_values(p, 100_000, FEE, T0)
    assert v.iloc[-1] == pytest.approx(100_000 * (1 - FEE) * 1.1)
    assert chained_return(v, 100_000) == pytest.approx((1 - FEE) * 1.1 - 1)  # the base is E0, not E0 net of the fee


def test_a_deposit_mid_window_leaves_the_excess_return_but_for_the_holds_entry_fee_on_it():
    days = [T0 + timedelta(days=i) for i in range(4)]
    p = _prices(*zip(days, (10_000.0, 10_500.0, 10_200.0, 11_000.0)))
    fund = pd.Series([103_000.0, 101_000.0, 104_000.0], index=pd.DatetimeIndex(days[1:]))
    bench = hold_values(p, 100_000, FEE, T0)
    plain = chained_return(fund, 100_000) - chained_return(bench, 100_000)
    flows = [(days[1], 10_000.0)]  # at day 1's close: the fund's equity carries it from then on, at the same returns
    fund2 = pd.Series([113_000.0, 101_000.0 * 113 / 103, 104_000.0 * 113 / 103], index=fund.index)
    bench2 = hold_values(p, 100_000, FEE, T0, flows)
    assert chained_return(fund2, 100_000, flows) == pytest.approx(chained_return(fund, 100_000))
    excess = chained_return(fund2, 100_000, flows) - chained_return(bench2, 100_000, flows)
    # The hold buys the deposit net of the one taker fee, the only thing a deposit moves (Advisor B34-BENCHMARK)
    assert excess - plain == pytest.approx(10_000 * FEE / 100_000 * 11_000 / 10_500)
    assert hold_values(p, 100_000, 0.0, T0, flows).pipe(chained_return, 100_000, flows) == pytest.approx(
        hold_values(p, 100_000, 0.0, T0).pipe(chained_return, 100_000))  # with no fee: exactly unchanged


def test_a_withdrawal_sells_units_at_its_price_and_is_no_loss():
    days = [T0 + timedelta(days=i) for i in range(3)]
    p = _prices(*zip(days, (10_000.0, 10_000.0, 10_000.0)))
    at = days[1] + timedelta(hours=1)
    v = hold_values(pd.concat([p, _prices((at, 10_000.0))]).sort_index(), 100_000, FEE, T0, [(at, -10_000.0)],
                    points=days[1:])
    assert v.iloc[-1] == pytest.approx(100_000 * (1 - FEE) - 10_000)
    assert chained_return(v, 100_000, [(at, -10_000.0)]) == pytest.approx(-FEE)


def test_a_perpetuals_hold_pays_settled_funding_on_what_it_holds():
    p = _prices((T0, 10_000.0), (T0 + timedelta(hours=8), 10_000.0), (T0 + timedelta(hours=9), 10_000.0))
    rates = pd.Series([0.0001], index=pd.DatetimeIndex([T0 + timedelta(hours=8)]))
    v = hold_values(p, 100_000, FEE, T0, funding=rates)
    held = 100_000 * (1 - FEE)
    assert v.iloc[-1] == pytest.approx(held - held * 0.0001)


def test_no_base_or_a_base_at_0_or_below_has_no_return():
    v = pd.Series([1.0], index=pd.DatetimeIndex([T0]))
    assert chained_return(v, 0) is None
    assert chained_return(v, -5) is None


def _hub(tmp_path, n_days=3):
    from sleeve_fund.venues import VENUES

    venue = next(v for v in VENUES.values() if v.hub)
    hist = HistoryStore(tmp_path)
    idx = pd.date_range(T0 - timedelta(hours=3), periods=(n_days * 1440) + 180, freq="1min", tz="UTC")
    c = 10_000.0 + np.arange(len(idx), dtype=float) * 0.5
    hist.append(venue.name, BENCH_PAIR, pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0},
                                                     index=idx), cursor="c")
    return hist, venue


def test_the_fund_hold_starts_at_e0_net_of_the_fee_and_ends_on_the_latest_stored_close(tmp_path):
    hist, venue = _hub(tmp_path)
    fb = fund_hold(None, T0, 100_000, hist=hist, fee=FEE)
    closes = hist.read(venue.name, BENCH_PAIR, minutes=1)["close"]
    p0 = float(closes[closes.index <= T0].iloc[-1])
    assert fb.e0 == 100_000
    assert fb.as_of == closes.index[-1] == fb.values.index[-1]  # never up to a day behind
    assert fb.values.iloc[-1] == pytest.approx(100_000 * (1 - FEE) * float(closes.iloc[-1]) / p0)
    t1 = fb.values.index[0]  # the first point: T0's day's last close, E0 net of the fee at that price
    assert t1 == closes[closes.index.floor("1D") == T0.floor("1D")].index[-1]
    assert fb.values.iloc[0] == pytest.approx(100_000 * (1 - FEE) * float(closes[t1]) / p0) and fb.values.iloc[0] > 0
    assert ("funding" in fb.source) if venue.perpetual else ("spot" in fb.source)


def test_a_fund_hold_on_e0_at_0_or_below_has_no_values_to_measure_from(tmp_path):
    hist, _ = _hub(tmp_path)
    assert fund_hold(None, T0, 0, hist=hist, fee=FEE).values.empty
    assert fund_hold(None, T0, -50, hist=hist, fee=FEE).values.empty


def test_no_stored_closes_reaching_the_start_is_no_fund_hold(tmp_path):
    hist, _ = _hub(tmp_path)
    assert fund_hold(None, T0 - timedelta(days=5), 100_000, hist=hist, fee=FEE) is None
    assert fund_hold(None, T0, 100_000, hist=HistoryStore(tmp_path / "empty"), fee=FEE) is None


# ---- the book's pages (CR MUST and the Advisor's cells): the fund's hold counted once ----------------------------------
def _fund_book(tmp_path, monkeypatch, pnl=(10.0, 20.0)):
    """Two fund-held Strategies on 100,000 (CR's probe), the hub's closes rising from 10,000 over three days."""
    from test_cut_a_fund_equity import HELD, _bridge, _store

    from sleeve_fund import history
    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.dashboard.metrics import sleeve_summary
    from sleeve_fund.paper.runtime import SleeveRuntime

    import test_cut_a_fund_equity as cut_a

    hub, venue = _hub(tmp_path / "hub")
    monkeypatch.setattr(history, "DEFAULT_ROOT", hub.root)
    monkeypatch.setattr(cut_a, "T0", T0.to_pydatetime())
    store = _store(tmp_path, True)
    _bridge(store)
    t = [T0.to_pydatetime() + timedelta(hours=2)]
    rts = []
    for n in ("s1", "s2"):
        store.create_sleeve(name=n, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                            starting_balance=0, risk_profile="balanced")
        rt = SleeveRuntime(store, n, now=lambda: t[0])
        rt.on_start(0.0005)
        rts.append(rt)
    for day, eq in enumerate(pnl):
        for rt in rts:
            rt.tick(equity=eq, price=10_000 + day, **HELD)
        t[0] += timedelta(days=1)
    sleeves = [store.sleeve(n) for n in ("s1", "s2")]
    frames = {s.name: bookm.daily(store, s.name) for s in sleeves}
    summaries = [bookm.sleeve_extras(store, sleeve_summary(store, s), frames[s.name]) for s in sleeves]
    return store, summaries, frames


def test_two_fund_held_strategies_put_one_hold_of_the_funds_capital_on_the_books_line(tmp_path, monkeypatch):
    """Advisor cell 1 and CR's: N Strategies give the book one fund hold, never N."""
    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.supervisor import bridge_start

    store, summaries, frames = _fund_book(tmp_path, monkeypatch)
    view = bookm.book_view(store, summaries, frames)
    start = view["start"]
    assert start == pytest.approx(100_000)  # the fund's capital: the Strategies hold 0
    one = fund_hold(store, bridge_start(store), start)
    assert view["bench_ret"] == pytest.approx(float(one.values.iloc[-1]) / start - 1)
    assert view["bench_source"] == one.source and view["bench_as_of"] == one.as_of
    curve = bookm.book_curve(summaries, frames, start, one.values)
    assert curve["benchmark"].iloc[-1] == pytest.approx(bookm.on_days(one.values, curve.index, start).iloc[-1])
    assert curve["benchmark"].max() < start * 1.5  # one hold of 100,000 on a market up about 22%; two would be over 2x


def test_a_reports_window_measures_the_hold_bought_at_its_start_on_the_books_equity_then(tmp_path, monkeypatch):
    """Advisor cell 3 and CR's: the base is E0, the book's equity at the window's start, never a negative mark."""
    from functools import partial

    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.dashboard import reports

    store, summaries, frames = _fund_book(tmp_path, monkeypatch)
    base = bookm.fund_cash(store)
    hold = partial(fund_hold, store)
    curve = bookm.book_curve(summaries, frames, base)
    start = curve.index[1]  # opens on the second day: E0 is the first day's close
    perf = reports.performance(summaries, frames, [], start.to_pydatetime(), None, base, hold)
    e0 = float(curve["equity"].iloc[0])
    assert not perf["empty"] and e0 > 0
    fb = fund_hold(store, curve.index[0] + pd.Timedelta(days=1), e0)
    assert perf["bench"] == pytest.approx(float(fb.values.iloc[-1]) / e0 - 1)
    month = reports.monthly(summaries, frames, {}, base, hold)
    assert all(r["bench_ret"] is not None and r["bench_ret"] > -0.01 for r in month["rows"])


def test_a_month_with_no_base_reads_n_a():
    """Advisor: Reports divide by E0; at 0 or below, n/a."""
    from sleeve_fund.dashboard.reports import _hold_ret

    assert _hold_ret(lambda *a: pytest.fail("never bought"), T0, 0.0, T0) is None
    assert _hold_ret(lambda *a: pytest.fail("never bought"), T0, -10.0, T0) is None


def test_b41_a_book_whose_peak_is_0_reads_no_drawdown_and_never_raises(tmp_path, monkeypatch):
    """B-41 (HoE): Money marks and a peak of 0 (no fund cash, every Strategy at 0) made 0/0 raise before `where`."""
    from sleeve_fund.dashboard import book as bookm

    store, summaries, frames = _fund_book(tmp_path, monkeypatch, pnl=(0.0, 0.0))
    assert frames["s1"]["equity"].dtype == object  # Money, as the journal reads it
    curve = bookm.book_curve(summaries, frames, 0.0)
    assert list(curve["drawdown"]) == [0.0] * len(curve)
    recent = bookm.recent_curve(store, [store.sleeve("s1")], 7, None, 0.0)
    assert set(recent["drawdown"]) <= {0.0}
