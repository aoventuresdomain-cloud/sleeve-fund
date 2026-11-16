"""Family C, step (a) (PM 16 Nov 21:06 UTC, backtest only): the relative strength rank block. Each week, at the close at
Monday 00:00 UTC, the basket's assets are ranked by their return over the trailing lookback_days, and the ranking holds
through the week. Done when:

1. it reads the close at the decision time and the one lookback_days before it, never a later one (look-ahead);
2. a tie keeps the order the tied assets held at the last ranking, and with none the basket's listed order;
3. a week where any asset lacks either close has no value, is counted, and leaves the last ranking for the next tie;
4. it ranks at Monday 00:00 UTC closes only, the same on daily and 4-hour bars (alignment);
5. it is a studies-only block that takes no timeframe or input, warming up over its lookback and a week;
6. only a basket's run hands it every asset's closes; one asset alone is refused."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.study import block_warmup
from sleeve_fund.strategies import study_only_uses
from sleeve_fund.strategies.definitions import (
    DAY_NS,
    RelativeStrengthRank,
    check_definition,
    is_weekly_decision,
    relative_strength_ranks,
    to_params,
)

ORDER = ("BTC", "ETH", "SOL")
MONDAY = pd.Timestamp("2024-01-08", tz="UTC")  # a Monday
D = pd.Timedelta(days=1)


START = pd.Timestamp("2024-01-01", tz="UTC")


def _closes(days: int = 120, seed: int = 1, minutes: int = 1440, start=START) -> dict:
    """{asset: Series of closes by close time}: random walks on bars of `minutes`, the first closing a bar after start."""
    rng = np.random.default_rng(seed)
    step = pd.Timedelta(minutes=minutes)
    idx = pd.date_range(start + step, periods=days * 1440 // minutes, freq=step)
    return {a: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03 * (minutes / 1440) ** 0.5, len(idx)))), index=idx)
            for a in ORDER}


def _source(own: str, closes: dict) -> tuple:
    return own, {a: (s.index.as_unit("ns").asi8.copy(), s.to_numpy(dtype=float).copy()) for a, s in closes.items()}, ORDER


def _block(own: str, closes: dict, lookback: int = 28) -> RelativeStrengthRank:
    b = RelativeStrengthRank(lookback)
    b.source = _source(own, closes)
    return b


def _feed(b: RelativeStrengthRank, index) -> list:
    """Feeds each close time in turn; the block's (rank, score) after each, None where it has no value."""
    out = []
    for t in index:
        b.update_ohlcv(1.0, 1.0, 1.0, 1.0, 1.0, ts_ns=int(t.value))
        out.append((b.values["rank"], b.values["score"]) if b.initialized else None)
    return out


def _by_hand(closes: dict, t: pd.Timestamp, lookback: int = 28) -> dict:
    return {a: float(closes[a][t] / closes[a][t - lookback * D] - 1) for a in ORDER}


# ---- 1. no look-ahead -------------------------------------------------------------------------------------------

def test_it_scores_the_close_at_the_decision_over_the_close_lookback_days_before_and_ranks_by_it():
    closes = _closes()
    t = MONDAY + 5 * 7 * D
    want = _by_hand(closes, t)
    order = sorted(ORDER, key=lambda a: -want[a])
    for own in ORDER:
        b = _block(own, closes)
        got = _feed(b, [t])[0]
        assert got == (float(order.index(own) + 1), want[own])


def test_a_close_after_the_decision_never_changes_its_ranking_and_the_two_it_reads_do():
    closes = _closes()
    t = MONDAY + 5 * 7 * D
    base = {own: _feed(_block(own, closes), [t])[0] for own in ORDER}
    later = {a: s.copy() for a, s in closes.items()}
    for a in ORDER:  # every close after the decision moved, the minute after it included
        later[a][later[a].index > t] *= 1 + 9 * (ORDER.index(a) + 1)
    assert {own: _feed(_block(own, later), [t])[0] for own in ORDER} == base
    weakest = min(ORDER, key=lambda a: base[a][1])
    for when in (t, t - 28 * D):  # the weakest pushed to the top through either close it reads
        moved = {a: s.copy() for a, s in closes.items()}
        moved[weakest][when] *= 10 if when == t else 0.1
        assert _feed(_block(weakest, moved), [t])[0][0] == 1.0
    moved = {a: s.copy() for a, s in closes.items()}
    moved[weakest][t - 27 * D] *= 10  # a close between the two is never read
    moved[weakest][t - 29 * D] *= 10
    assert _feed(_block(weakest, moved), [t])[0] == base[weakest]


# ---- 2. ties ----------------------------------------------------------------------------------------------------

def test_a_tie_keeps_the_last_rankings_order_and_with_none_the_listed_order():
    assert relative_strength_ranks({"BTC": 0.1, "ETH": 0.1, "SOL": 0.0}, None, ORDER) == {"BTC": 1, "ETH": 2, "SOL": 3}
    assert relative_strength_ranks({"BTC": 0.1, "ETH": 0.1, "SOL": 0.0}, {"ETH": 1, "SOL": 2, "BTC": 3},
                                   ORDER) == {"ETH": 1, "BTC": 2, "SOL": 3}
    assert relative_strength_ranks({"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}, {"SOL": 1, "BTC": 2, "ETH": 3},
                                   ORDER) == {"SOL": 1, "BTC": 2, "ETH": 3}
    # a tie between the old 1 and 3 keeps them in that order; the score still outranks any old order
    assert relative_strength_ranks({"BTC": -0.2, "ETH": 0.3, "SOL": -0.2}, {"SOL": 1, "ETH": 2, "BTC": 3},
                                   ORDER) == {"ETH": 1, "SOL": 2, "BTC": 3}


def test_the_block_breaks_a_tie_by_its_last_weeks_ranking():
    closes = _closes()
    t1, t2 = MONDAY + 5 * 7 * D, MONDAY + 6 * 7 * D
    first = {own: _feed(_block(own, closes), [t1])[0][0] for own in ORDER}
    top = min(ORDER, key=first.get)
    tied = {a: s.copy() for a, s in closes.items()}
    for a in ORDER:  # every asset up 5% over the lookback to t2: a three-way tie
        tied[a][t2] = tied[a][t2 - 28 * D] * 1.05
    for own in ORDER:
        b = _block(own, tied)
        assert _feed(b, [t1, t2])[1][0] == first[own]  # the order held at t1 stands
    b = _block(top, tied)
    assert _feed(b, [t2])[0][0] == float(ORDER.index(top) + 1)  # no ranking before: the listed order


# ---- 3. a missing close -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("gap", [0, 28])
def test_a_week_missing_any_assets_close_has_no_value_and_is_counted(gap):
    closes = _closes()
    t1, t2, t3 = (MONDAY + k * 7 * D for k in (5, 6, 7))
    holed = {a: s.copy() for a, s in closes.items()}
    holed["SOL"] = holed["SOL"].drop(t2 - gap * D)  # SOL's close at the decision, or lookback days before it
    for own in ORDER:
        b = _block(own, holed)
        got = _feed(b, [t1, t1 + D, t2, t2 + D, t3])
        assert got[0] is not None and got[1] == got[0]
        assert got[2] is None and got[3] is None  # that week reads nothing, for every asset, not only SOL
        assert got[4] == _feed(_block(own, closes), [t3])[0]  # ranks again the week after
        assert (b.weeks, b.no_rank, b.unranked) == (2, 1, [t2.date().isoformat()])


def test_after_a_missing_week_a_tie_falls_back_to_the_ranking_before_it():
    closes = _closes()
    t1, t2, t3 = (MONDAY + k * 7 * D for k in (5, 6, 7))
    first = {own: _feed(_block(own, closes), [t1])[0][0] for own in ORDER}
    holed = {a: s.copy() for a, s in closes.items()}
    holed["ETH"] = holed["ETH"].drop(t2)
    for a in ORDER:
        holed[a][t3] = holed[a][t3 - 28 * D] * 0.9  # a three-way tie at t3
    for own in ORDER:
        assert _feed(_block(own, holed), [t1, t2, t3])[2][0] == first[own]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
def test_a_week_whose_score_isnt_finite_has_no_value_and_is_dated(bad):
    closes = _closes()
    t1, t2 = MONDAY + 5 * 7 * D, MONDAY + 6 * 7 * D
    holed = {a: s.copy() for a, s in closes.items()}
    holed["BTC"][t2 - 28 * D] = bad
    b = _block("ETH", holed)
    assert _feed(b, [t1, t2])[1] is None
    assert (b.weeks, b.no_rank, b.unranked) == (1, 1, [t2.date().isoformat()])


def test_with_no_source_it_never_reads_and_counts_every_week():
    b = RelativeStrengthRank()
    assert _feed(b, [MONDAY, MONDAY + D, MONDAY + 7 * D]) == [None, None, None]
    assert (b.weeks, b.no_rank, b.unranked) == (0, 2, ["2024-01-08", "2024-01-15"])


# ---- 4. alignment -----------------------------------------------------------------------------------------------

def test_only_a_close_at_monday_00_utc_is_a_decision():
    assert is_weekly_decision(MONDAY.value) and is_weekly_decision((MONDAY + 7 * D).value)
    for off in (pd.Timedelta(hours=4), pd.Timedelta(minutes=1), D, 6 * D, -pd.Timedelta(hours=4)):
        assert not is_weekly_decision((MONDAY + off).value)
    assert pd.Timestamp(MONDAY.value).day_name() == "Monday"


def test_it_ranks_at_monday_closes_only_and_holds_that_ranking_through_the_week():
    closes = _closes()
    idx = closes["BTC"].index[(closes["BTC"].index > MONDAY + 3 * 7 * D) & (closes["BTC"].index <= MONDAY + 8 * 7 * D)]
    for own in ORDER:
        got = _feed(_block(own, closes), idx)
        for t, g in zip(idx, got):
            monday = t if is_weekly_decision(t.value) else t.floor("D") - ((t.floor("D").dayofweek) * D)
            if t < MONDAY + 4 * 7 * D:  # before its first Monday with 28 days behind it
                continue
            want = _by_hand(closes, monday)
            assert g == (float(sorted(ORDER, key=lambda a: -want[a]).index(own) + 1), want[own]), t
    first = _feed(_block("BTC", closes), idx[: list(idx).index(MONDAY + 4 * 7 * D)])
    assert all(g is None for g in first)  # nothing before the first Monday close


def test_daily_and_4_hour_bars_rank_the_same_weeks_the_same():
    daily = _closes(seed=4)
    # 4-hour bars whose midnight closes are the daily closes, others in between
    four = {}
    for a, s in daily.items():
        idx = pd.date_range(s.index[0] - D + pd.Timedelta(hours=4), s.index[-1], freq="4h")
        f = pd.Series(np.nan, index=idx)
        f[s.index] = s.to_numpy()
        four[a] = f.interpolate() * (1 + 0.01 * np.sin(np.arange(len(idx))))
        four[a][s.index] = s.to_numpy()
    mondays = [t for t in daily["BTC"].index if is_weekly_decision(t.value) and t >= MONDAY + 4 * 7 * D]
    for own in ORDER:
        d = dict(zip(daily["BTC"].index, _feed(_block(own, daily), daily["BTC"].index)))
        h = dict(zip(four["BTC"].index, _feed(_block(own, four), four["BTC"].index)))
        assert [d[t] for t in mondays] == [h[t] for t in mondays]
        assert all(h[t + pd.Timedelta(hours=k)] == h[t] for t in mondays[:-1] for k in (4, 8, 20))


# ---- 5. the definition ------------------------------------------------------------------------------------------

def _defn(**block) -> dict:
    return {"version": 1, "name": "rs", "title": "rs", "reason": "test",
            "blocks": {"rs": {"kind": "relative_strength_rank", **block}},
            "long": {"entry": {"left": "rs.rank", "op": "<=", "right": 1}, "exit": {"left": "rs.rank", "op": ">=", "right": 2}}}


def test_it_is_a_studies_only_block_with_no_timeframe_or_input():
    defn = _defn()
    checked = check_definition(defn, bar_minutes=1440)
    assert checked.blocks["rs"]["settings"] == {"lookback_days": 28}
    assert study_only_uses({"definition": defn}) == ["the relative strength rank block"]
    with pytest.raises(ValueError, match="takes no timeframe"):
        check_definition({**defn, "blocks": {"rs": {"kind": "relative_strength_rank", "timeframe": "1d"}}},
                         bar_minutes=240)
    with pytest.raises(ValueError, match="reads several prices"):
        check_definition({**defn, "blocks": {"rs": {"kind": "relative_strength_rank", "input": "close"}}},
                         bar_minutes=1440)
    with pytest.raises(ValueError, match="lookback_days must be a whole number, at least 7"):
        check_definition({**defn, "blocks": {"rs": {"kind": "relative_strength_rank", "lookback_days": 6}}},
                         bar_minutes=1440)


def test_it_warms_up_over_its_lookback_and_a_week():
    assert RelativeStrengthRank.ranking_warmup_minutes({}) == 35 * 1440
    assert block_warmup("relative_strength_rank", {"lookback_days": 28}, 240) == 35 * 6 + 1
    assert block_warmup("relative_strength_rank", {"lookback_days": 14}, 1440) == 21 + 1
    assert DAY_NS == 86_400 * 10**9


def test_a_long_lookback_passes_the_check_with_or_without_the_candle_size():
    # the block is fed no candles, so its lookback is no candle warm-up: a check that doesn't know the candle (as
    # to_params runs it) must not count 35 days as 1-minute candles against MAX_WARMUP_BARS
    for days in (28, 365):
        defn = {**WEEKLY_LONG, "blocks": {"rs": {"kind": "relative_strength_rank", "lookback_days": days}}}
        assert check_definition(defn).warmup[None] == 0 and check_definition(defn, bar_minutes=240).warmup[None] == 0
        assert to_params(defn, "BINANCE")["definition"]["blocks"]["rs"]["lookback_days"] == days


# ---- 6. only a basket hands it closes ---------------------------------------------------------------------------

WEEKLY_LONG = {"version": 1, "name": "rs-long", "title": "rs", "reason": "test",
               "blocks": {"rs": {"kind": "relative_strength_rank"}},
               "long": {"entry": {"all": [{"left": "rs.rank", "op": "<=", "right": 1},
                                          {"left": {"time": "weekday"}, "op": "<=", "right": 0},
                                          {"left": {"time": "hour"}, "op": "<=", "right": 0}]},
                        "exit": {"left": "rs.rank", "op": ">=", "right": 2}}}


def _legs(days: int = 150, warm: int = 40) -> list:
    from decimal import Decimal

    from nautilus_trader.model import Venue

    from sleeve_fund.instruments import FeeSchedule, spot_pair
    from sleeve_fund.research.portfolio_run import LegSpec

    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01 01:00", periods=days * 24, freq="1h", tz="UTC")  # hourly, closing on the hour
    out = []
    for a in ORDER:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
        open_ = np.concatenate([[100.0], close[:-1]])
        h = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001,
                          "low": np.minimum(open_, close) * 0.999, "close": close, "volume": 1_000.0},
                         index=pd.DatetimeIndex(idx, name="timestamp"))
        day = (h.index - pd.Timedelta(1, "ns")).floor("D") + D
        d = h.groupby(day).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        d = d.rename_axis("timestamp")
        w, window = d.iloc[:warm], d.iloc[warm:]
        inst = spot_pair(a, "USD", FeeSchedule(Decimal("0.001"), Decimal("0.002")), Venue("KRAKEN"))
        out.append(LegSpec(a, "rules", inst, window, h[h.index > w.index[-1]], w,
                           params={"definition": WEEKLY_LONG}))
    return out


def test_one_asset_alone_is_refused_wherever_it_would_run():
    from sleeve_fund.research.placebo import fast_path
    from sleeve_fund.research.portfolio_run import RANK_NEEDS, run_portfolio
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.risk import PortfolioProfile

    s = _legs(days=60)[0]
    loose = PortfolioProfile(gross=100, net_instrument=100, margin=100, open_risk=100, drawdown=0.99, daily_loss=0.99)
    with pytest.raises(ValueError, match="needs 2 or more"):
        run_portfolio([s], loose)
    with pytest.raises(ValueError, match="needs 2 or more"):
        run_backtest("rules", s.prices, s.instrument, params=s.params, exec_prices=s.exec_prices, exec_minutes=60,
                     warmup_prices=s.warmup_prices)
    with pytest.raises(ValueError, match="needs 2 or more"):
        fast_path(WEEKLY_LONG, s.prices, 1440, 0, 0.0)
    assert "needs 2 or more" in RANK_NEEDS


def test_a_baskets_run_hands_every_leg_the_same_closes_and_its_entries_follow_the_ranking():
    from sleeve_fund.research.portfolio_run import rank_closes, run_portfolio
    from sleeve_fund.risk import PortfolioProfile

    specs = _legs()
    closes = rank_closes(specs)
    for s in specs:  # warm-up and window, by close time, nothing else
        bars = pd.concat([s.warmup_prices, s.prices])
        assert list(closes[s.name][0]) == list(bars.index.as_unit("ns").asi8)
        assert list(closes[s.name][1]) == list(bars["close"])
    loose = PortfolioProfile(gross=100, net_instrument=100, margin=100, open_risk=100, drawdown=0.99, daily_loss=0.99)
    run = run_portfolio(specs, loose)
    daily = {s.name: pd.concat([s.warmup_prices, s.prices])["close"] for s in specs}
    mondays = [t for t in specs[0].prices.index if is_weekly_decision(t.value)]
    top = {}
    for m in mondays:
        sc = _by_hand(daily, m)
        top[m] = max(ORDER, key=lambda a: sc[a])
    entries = sorted((pd.Timestamp(d["bar_ts"]), d["strategy"]) for d in run.decisions
                     if d["kind"] == "entry" and d["outcome"] == "approved")
    want = sorted((m, top[m]) for n, m in enumerate(mondays) if n == 0 or top[mondays[n - 1]] != top[m])
    assert len(want) >= 3 and entries == want  # every entry on a Monday close, by the asset ranked first there
    weeks = sum(1 for t in pd.concat([specs[0].warmup_prices, specs[0].prices]).index if is_weekly_decision(t.value))
    for s in specs:  # the four Mondays in the first 28 days of history can't rank; every later one does
        assert run.legs[s.name].rule_counts == {"rank_weeks": weeks - 4, "rank_no_rank": 4,
                                                "rank_unranked": ["2024-01-08", "2024-01-15", "2024-01-22",
                                                                  "2024-01-29"]}
