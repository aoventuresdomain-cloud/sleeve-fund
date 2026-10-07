"""The degraded-bar entry gate (board 5a, Data Architect ruling (a) on #144): a bar built with more than 10% of
its minutes missing opens nothing, while its exits still run and the indicators still update on it."""

import pandas as pd
import pytest

from sleeve_fund import bars
from sleeve_fund.research.runner import run_backtest
from test_backtest import _path

PERP = {"market": "perp", "allow_short": True}


def _degrade(df: pd.DataFrame, at, missing: int = 3) -> pd.DataFrame:
    """The frame with the bars at these close times flagged as the store flags a bar missing `missing` minutes."""
    df = df.assign(missing=0, degraded=False)
    df.loc[at, "missing"] = missing
    df.loc[at, "degraded"] = True
    return df


def test_the_stores_columns_are_stable_on_every_path():
    """1-minute bars and an empty read carry missing and degraded too, so a consumer can rely on them (DA minor)."""
    idx = pd.date_range("2026-10-01", periods=30, freq="1min", tz="UTC")
    minutes = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    one = bars.build_bars(minutes, 1)
    assert list(one.columns) == bars.COLUMNS and (one["missing"] == 0).all() and not one["degraded"].any()
    assert list(bars.build_bars(minutes.iloc[:0], 15).columns) == bars.COLUMNS
    thin = bars.build_bars(minutes.drop(idx[3:5]), 15)  # 2 of 15 missing: over 10%
    assert list(thin["missing"]) == [2, 0] and list(thin["degraded"]) == [True, False]


def test_no_entry_on_a_degraded_bar_and_the_next_whole_bar_enters(prices, instrument):
    """Buy and hold enters on its first bar; with the first three degraded it enters on the fourth's close."""
    df = _path(prices, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    whole = run_backtest("buy_and_hold", df, instrument, half_spread=0)
    assert whole.fills.sort_values("ts_last")["ts_last"].iloc[0] == df.index[0]
    thin = run_backtest("buy_and_hold", _degrade(df, df.index[:3]), instrument, half_spread=0)
    fills = thin.fills.sort_values("ts_last")
    assert fills["ts_last"].iloc[0] == df.index[3]
    assert float(fills["avg_px"].iloc[0]) == pytest.approx(103.0)


def test_a_degraded_bar_still_exits_but_opens_no_new_side(prices, instrument):
    """Ping-pong long from 100 sells at +1% (101.5) and would short there. With that bar degraded the long still
    closes, but no short opens on it: the next entry comes on a whole bar."""
    closes = [100.0, 100.5, 100.8, 101.5, 101.3, 101.2, 100.9, 101.0, 101.5, 102.0, 102.0]
    df = _path(prices, closes)
    whole = run_backtest("ping_pong", df, instrument, PERP, half_spread=0)
    w = whole.fills.sort_values("ts_last")
    assert [whole.decisions[o]["intent"] for o in w.index][:3] == ["entry", "exit", "entry"]
    assert w["ts_last"].iloc[2] == df.index[3]  # the short opened on the 101.5 bar

    thin = run_backtest("ping_pong", _degrade(df, [df.index[3]]), instrument, PERP, half_spread=0)
    t = thin.fills.sort_values("ts_last")
    intents = [thin.decisions[o]["intent"] for o in t.index]
    assert intents[:2] == ["entry", "exit"] and t["ts_last"].iloc[1] == df.index[3]  # the exit ran
    entries = t[[i == "entry" for i in intents]]
    assert df.index[3] not in set(entries["ts_last"])  # nothing opened on the degraded bar
    assert len(entries) >= 2 and entries["ts_last"].iloc[1] > df.index[3]


def test_degraded_bars_inside_a_hold_change_nothing(prices, instrument):
    """Bars degraded only while a position is held (never an entry bar) leave every fill as it was: the
    indicators still update on them and the exits still run on them."""
    params = {}
    whole = run_backtest("trend_filter", prices, instrument, params, half_spread=0)
    f = whole.fills.sort_values("ts_last")
    entry_ts = {ts for o, ts in zip(f.index, f["ts_last"]) if whole.decisions[o]["intent"] == "entry"}
    held = whole.exposure.reindex(prices.index).fillna(0) > 0
    inside = [ts for ts in prices.index[held.to_numpy()] if ts not in entry_ts]
    assert len(entry_ts) >= 2 and len(inside) > 50  # the case has trades to compare
    thin = run_backtest("trend_filter", _degrade(prices, inside), instrument, params, half_spread=0)
    t = thin.fills.sort_values("ts_last")
    assert list(t["ts_last"]) == list(f["ts_last"]) and list(t["side"]) == list(f["side"])
    assert [float(x) for x in t["avg_px"]] == pytest.approx([float(x) for x in f["avg_px"]])
