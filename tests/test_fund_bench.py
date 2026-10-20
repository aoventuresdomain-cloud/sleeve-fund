"""The fund's benchmark (Advisor B34-BENCHMARK, 20 Oct): one buy-and-hold on the fund's capital, E0 x (1 - fee) x P / P0,
flows bought and sold at their time, funding on a perpetual's long, returns from E0."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.dashboard.fund_bench import BENCH_PAIR, chained_return, fund_benchmark, hold_values
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


class _Store:
    """What fund_benchmark reads of the fund: its flows and its marks."""

    def __init__(self, flows, marks):
        self._flows, self._marks = flows, sorted(marks)

    def fund_flows(self):
        return self._flows

    def book_equity_at_or_before(self, ts):
        known = [e for t, e in self._marks if t <= ts]
        return known[-1] if known else None

    def first_book_mark(self):
        return {"ts": self._marks[0][0], "equity": self._marks[0][1]} if self._marks else None


def _hub(tmp_path, n_days=3):
    from sleeve_fund.venues import VENUES

    venue = next(v for v in VENUES.values() if v.hub)
    hist = HistoryStore(tmp_path)
    idx = pd.date_range(T0 - timedelta(hours=1), periods=(n_days * 1440) + 60, freq="1min", tz="UTC")
    c = 10_000.0 + np.arange(len(idx), dtype=float) * 0.5
    hist.append(venue.name, BENCH_PAIR, pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0},
                                                     index=idx), cursor="c")
    return hist, venue


def test_the_fund_benchmark_is_none_before_the_bridge(tmp_path):
    hist, _ = _hub(tmp_path)
    assert fund_benchmark(_Store([], [(T0.to_pydatetime(), 100_000)]), hist=hist, fee=FEE) is None


def test_the_fund_benchmark_starts_at_e0_net_of_the_fee_and_ends_on_the_latest_stored_close(tmp_path):
    hist, venue = _hub(tmp_path)
    bridge = (T0 - timedelta(minutes=30)).to_pydatetime()
    flows = [{"id": 1, "kind": "deposit", "amount": 100_000, "rebase": True, "applied_at": bridge}]
    fb = fund_benchmark(_Store(flows, [(T0.to_pydatetime(), 100_000)]), start=T0.to_pydatetime(), hist=hist, fee=FEE)
    closes = hist.read(venue.name, BENCH_PAIR, minutes=1)["close"]
    p0 = float(closes[closes.index <= T0].iloc[-1])
    assert fb.e0 == 100_000
    assert fb.as_of == closes.index[-1]  # never up to a day behind
    assert fb.values.index[-1] == closes.index[-1]
    assert fb.values.iloc[-1] == pytest.approx(100_000 * (1 - FEE) * float(closes.iloc[-1]) / p0)
    assert "funding" in fb.source if venue.perpetual else "spot" in fb.source
