"""R3: results by regime, report only (Advisor 6 Oct 14:05 UK): fixed backward-looking labels at entry (close vs the
200-day average, 30-day volatility vs its trailing-year median, ADX 14), per-bucket figures, "too few" under 20
trades and the regime-dependent flag over 80% of a positive net P&L."""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import regime_split as rs

DAYS = pd.date_range("2024-01-02", periods=700, freq="D", tz="UTC")


def _bars(close, spread=0.01):
    close = pd.Series(close, index=DAYS[: len(close)], dtype=float)
    return pd.DataFrame({"open": close.shift().fillna(close.iloc[0]), "high": close * (1 + spread),
                         "low": close * (1 - spread), "close": close})


def _walk(seed=0, n=700, vol=0.02):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, vol, n)))


def test_labels_never_look_ahead():
    bars = _bars(_walk(1))
    whole = rs.labels(rs.daily(bars))
    for cut in (250, 420, 600):
        part = rs.labels(rs.daily(bars.iloc[:cut]))
        assert part.equals(whole.iloc[:cut])


def test_bull_and_bear_need_200_closes_then_follow_the_average():
    up = np.linspace(100, 300, 260)
    lab = rs.labels(_bars(np.concatenate([up, np.linspace(300, 150, 60)])))
    assert (lab["direction"].iloc[:199] == rs.UNLABELLED).all()
    assert lab["direction"].iloc[220] == "bull" and lab["direction"].iloc[-1] == "bear"


def test_volatility_is_measured_against_its_own_trailing_year():
    calm, wild = _walk(2, 450, 0.005), None
    wild = calm[-1] * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.05, 100)))
    lab = rs.labels(_bars(np.concatenate([calm, wild])))
    assert (lab["volatility"].iloc[: rs.VOL_DAYS + rs.VOL_YEAR - 1] == rs.UNLABELLED).all()
    assert lab["volatility"].iloc[-1] == "high"


def test_adx_reads_trend_on_a_steady_climb_and_range_on_a_flat_chop():
    climb = rs.labels(_bars(np.linspace(100, 200, 120)))["trend"]
    chop = rs.labels(_bars(100 + np.tile([1.0, -1.0], 60)))["trend"]
    assert (climb.iloc[: 2 * rs.ADX_DAYS] == rs.UNLABELLED).all()
    assert climb.iloc[-1] == "trend" and chop.iloc[-1] == "range"


def test_intraday_bars_give_the_same_labels_as_their_daily_bars():
    d = _bars(_walk(4, 300))
    hourly = d["close"].resample("1h").ffill().iloc[1:]
    h = pd.DataFrame({"open": hourly, "high": hourly, "low": hourly, "close": hourly})
    h.loc[d.index[1:], "high"], h.loc[d.index[1:], "low"] = d["high"].iloc[1:], d["low"].iloc[1:]
    day = rs.daily(h)
    assert (day["close"] == d["close"].reindex(day.index)).all()


def _trip(day, pnl):
    return {"opened": DAYS[day] + pd.Timedelta(minutes=1), "pnl": pnl}


def test_a_trade_takes_the_labels_of_the_last_close_before_its_entry_and_buckets_add_up():
    bars = _bars(np.concatenate([np.linspace(100, 300, 260), np.linspace(300, 150, 120)]))
    lab = rs.labels(bars)
    bull_day = int(np.flatnonzero(lab["direction"] == "bull")[5])
    bear_day = int(np.flatnonzero(lab["direction"] == "bear")[-5])
    trips = [_trip(bull_day, 10.0)] * 25 + [_trip(bear_day, -2.0)] * 3 + [_trip(10, 1.0)]
    hold = bars["close"] * 10
    d = {s.dimension: s for s in rs.split(trips, bars, hold)}["direction"]
    b = {x.name: x for x in d.buckets}
    assert (b["bull"].trades, b["bear"].trades, b[rs.UNLABELLED].trades) == (25, 3, 1)
    assert b["bull"].net_pnl == 250.0 and not b["bull"].too_few and b["bear"].too_few
    assert sum(x.trades for x in d.buckets) == len(trips) and sum(x.net_pnl for x in d.buckets) == 245.0
    assert b["bull"].share == pytest.approx(250 / 245) and d.dependent == "bull"
    assert sum(x.days for x in d.buckets) == pytest.approx(1.0)
    on = lab.index[lab["direction"].shift(1) == "bear"]  # a day is the regime its previous close named
    assert b["bear"].hold_return == pytest.approx(float((1 + hold.pct_change().reindex(on).dropna()).prod() - 1))
    assert b["bear"].strategy_return is None  # no Strategy equity given


def test_no_flag_below_80_percent_or_when_the_total_is_a_loss():
    bars = _bars(np.concatenate([np.linspace(100, 300, 260), np.linspace(300, 150, 120)]))
    lab = rs.labels(bars)
    bull, bear = int(np.flatnonzero(lab["direction"] == "bull")[5]), int(np.flatnonzero(lab["direction"] == "bear")[-5])
    even = {s.dimension: s for s in rs.split([_trip(bull, 7.0), _trip(bear, 3.0)], bars)}["direction"]
    assert even.dependent is None and {b.name: b.share for b in even.buckets}["bull"] == pytest.approx(0.7)
    lost = {s.dimension: s for s in rs.split([_trip(bull, -7.0), _trip(bear, 3.0)], bars)}["direction"]
    assert lost.dependent is None and all(b.share is None for b in lost.buckets)
    assert all(b.hold_return is None for b in lost.buckets)


def test_the_tear_sheet_section_says_report_only_too_few_and_regime_dependent():
    bars = _bars(np.concatenate([np.linspace(100, 300, 260), np.linspace(300, 150, 120)]))
    lab = rs.labels(bars)
    bull = int(np.flatnonzero(lab["direction"] == "bull")[5])
    text = "\n".join(rs.render(rs.split([_trip(bull, 5.0)] * 3, bars)))
    assert "Results by regime (report only)" in text and "3 (too few)" in text
    assert "Regime-dependent" in text and "new counted variant" in text


def test_naive_fill_times_are_read_as_utc():
    bars = _bars(np.concatenate([np.linspace(100, 300, 260), np.linspace(300, 150, 120)]))
    lab = rs.labels(bars)
    bull = int(np.flatnonzero(lab["direction"] == "bull")[5])
    naive = {"opened": (DAYS[bull] + pd.Timedelta(minutes=1)).tz_localize(None).to_pydatetime(), "pnl": 1.0}
    d = {s.dimension: s for s in rs.split([naive], bars)}["direction"]
    assert {b.name: b.trades for b in d.buckets}["bull"] == 1


def test_a_trade_s_labels_never_change_with_the_prices_after_its_entry():
    # Point in time at entry: rewrite every close after the entry day and the trade lands in the same buckets.
    path = _walk(4)
    entry = 450
    trip = [{"opened": DAYS[entry] + pd.Timedelta(hours=5), "pnl": 1.0}]
    other = np.concatenate([path[: entry + 1], path[entry] * np.exp(np.cumsum(np.full(len(path) - entry - 1, 0.05)))])
    tag = lambda close: {s.dimension: next(b.name for b in s.buckets if b.trades)  # noqa: E731
                         for s in rs.split(trip, _bars(close))}
    assert tag(path) == tag(other) and rs.UNLABELLED not in tag(path).values()


def test_the_strategy_s_own_return_sits_beside_buy_and_hold_over_the_same_days():
    # Advisor 14 Oct 09:25 UK: like for like, day by day, the trade rows still by entry label.
    bars = _bars(np.concatenate([np.linspace(100, 300, 260), np.linspace(300, 150, 120)]))
    lab = rs.labels(bars)
    hold, equity = bars["close"] * 10, 1000 + pd.Series(np.arange(len(bars), dtype=float), index=bars.index)
    d = {s.dimension: s for s in rs.split([_trip(10, 1.0)], bars, hold, equity)}["direction"]
    on = lab.index[lab["direction"].shift(1) == "bull"]
    b = {x.name: x for x in d.buckets}
    assert b["bull"].strategy_return == pytest.approx(float((1 + equity.pct_change().reindex(on)).prod() - 1))
    assert b["bull"].hold_return == pytest.approx(float((1 + hold.pct_change().reindex(on)).prod() - 1))
    text = "\n".join(rs.render([d]))
    assert "Strategy over those days" in text and "price only, no fees" in text


def test_shares_add_up_with_the_unlabelled_and_read_n_a_when_the_total_is_not_positive():
    bars = _bars(np.concatenate([np.linspace(100, 300, 260), np.linspace(300, 150, 120)]))
    bull = int(np.flatnonzero(rs.labels(bars)["direction"] == "bull")[5])
    d = {s.dimension: s for s in rs.split([_trip(bull, 6.0), _trip(10, 2.0)], bars)}["direction"]
    assert sum(b.share for b in d.buckets) == pytest.approx(1.0)
    assert {b.name: b.share for b in d.buckets}[rs.UNLABELLED] == pytest.approx(0.25)
    lost = rs.split([_trip(bull, -6.0), _trip(10, 2.0)], bars)
    assert "n/a (total <= 0)" in "\n".join(rs.render(lost))


def test_a_bucket_over_80_percent_of_a_positive_total_is_flagged_regime_dependent():
    # F281-1 (CR): the threshold itself is pinned; 85% must be flagged, 75% must not.
    bars = _bars(np.concatenate([np.linspace(100, 300, 260), np.linspace(300, 150, 120)]))
    lab = rs.labels(bars)
    bull, bear = int(np.flatnonzero(lab["direction"] == "bull")[5]), int(np.flatnonzero(lab["direction"] == "bear")[-5])
    at = {s.dimension: s for s in rs.split([_trip(bull, 85.0), _trip(bear, 15.0)], bars)}["direction"]
    under = {s.dimension: s for s in rs.split([_trip(bull, 75.0), _trip(bear, 25.0)], bars)}["direction"]
    assert at.dependent == "bull" and under.dependent is None
