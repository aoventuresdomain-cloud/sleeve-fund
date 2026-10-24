"""Improve-loop round 1 (Researcher 24 Oct, Advisor cleared): qr23 with qr01b V4's shorts-below-SMA200 rule and qr21
with an SMA200 slope check on each entry, each round 0's definition exactly plus that one change; the lag block the
slope reads; the BTC list on round 0's dates; and the Advisor's two qr21 pins, entries are states (a cross taken while
the slope disagrees is entered once it agrees) and a reversal's other leg passes its own slope check or goes flat."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from sleeve_fund.research import rescore
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import check_definition, definition_hash, load_definition_file, to_params
from sleeve_fund.strategies.indicators import Lag, make_block

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "qr-improve-r1-list.md"


def _load(key: str) -> tuple[dict, dict]:
    v = PREREGISTERED[key]
    return load_definition_file(definition_path(v["definition"])), load_definition_file(definition_path(v["parent"]))


def test_qr23_r1_is_round_0_plus_v4s_short_rule():
    mine, r0 = _load("QR23_R1")
    v4 = load_definition_file(definition_path("qr01b_trend_multispeed_ls_v4_short_sma200"))
    assert mine["blocks"] == {**r0["blocks"], "close1d": v4["blocks"]["close1d"], "sma200d": v4["blocks"]["sma200d"]}
    assert mine["long"] == r0["long"] and "exits" not in mine
    assert mine["short"] == {**r0["short"], "entry": {"all": [r0["short"]["entry"],
                                                              {"left": "close1d", "op": "<", "right": "sma200d"}]}}
    check_definition(mine, bar_minutes=1440)
    assert definition_hash(mine) != definition_hash(r0)


def test_qr21_r1_is_round_0_plus_a_slope_check_on_each_entry():
    mine, r0 = _load("QR21_R1")
    assert mine["blocks"] == {**r0["blocks"], "sma_ago": {"kind": "lag", "period": 20, "input": "sma"}}
    assert mine["exits"] == r0["exits"]
    for side, op in (("long", ">"), ("short", "<")):
        assert mine[side] == {**r0[side], "entry": {"all": [r0[side]["entry"],
                                                            {"left": "sma", "op": op, "right": "sma_ago"}]}}
    checked = check_definition(mine, bar_minutes=1440)
    assert checked.warmup[None] >= 200 + 21  # the average's days, then 20 more of it
    assert definition_hash(mine) != definition_hash(r0)


def test_the_lag_reads_its_input_period_values_back():
    lag = make_block("lag", period=3)
    seen = []
    for x in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]:
        lag.update_raw(x)
        seen.append(lag.value if lag.initialized else None)
    assert seen == [None, None, None, 10.0, 11.0, 12.0] and Lag(20).warmup_bars == 21


def test_the_list_is_two_btc_rows_on_round_0s_dates_on_the_perpetual_with_shorts():
    from sleeve_fund.research.pinned import parse

    text = LIST.read_text()
    r0 = (ROOT / "configs" / "studies" / "qr-improve-r0-list.md").read_text()
    assert rescore.rows_of(text) == [(1, "qr23_r1", 1440), (2, "qr21_r1", 1440)]
    assert parse(text) == parse(r0) and parse(text).holdout_days == rescore.HOLDOUT_DAYS
    for name in ("qr23_r1", "qr21_r1"):
        req = rescore.request(name, 1440, "binance", "BTC/USDT", parse(text))
        assert req.params["definition"] == _load(name.upper())[0]
        assert req.params["market"] == "perp" and req.params["allow_short"] is True
        assert (req.use_holdout, req.holdout_days, req.train_days, req.test_days) == (False, 365, 730, 365)
        req.validate()


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--dry-run"], run=lambda *a, **k: None, journal=lambda: None) == 0
    assert "dry run: 2 checked, 0 run" in capfd.readouterr().out


# --- the Advisor's qr21 pins, on a short-period copy so a made-up path turns it ------------------------------------

SMA, AGO = 10, 5


def _small_qr21() -> dict:
    d = copy.deepcopy(_load("QR21_R1")[0])
    d["blocks"]["sma"]["period"], d["blocks"]["sma_ago"]["period"] = SMA, AGO
    del d["exits"]  # the stops are round 0's and unchanged; the pins are about entries and the reversal
    return d


def _entries(defn: dict, close: np.ndarray, instrument) -> tuple[list, list, pd.DataFrame]:
    from sleeve_fund.research.runner import run_backtest

    idx = pd.date_range("2024-01-01", periods=len(close), freq="1D", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1e3}, index=idx)
    res = run_backtest("rules", df, instrument, params={**to_params(defn), "market": "perp", "allow_short": True},
                       bar_minutes=1440, half_spread=0)
    f = res.fills
    rows = sorted((f.loc[o, "ts_last"], str(f.loc[o, "side"]), res.decisions[o]["intent"]) for o in f.index)
    sma = df["close"].rolling(SMA).mean()
    df["sma"], df["ago"] = sma, sma.shift(AGO)
    return ([(ts, side) for ts, side, intent in rows if intent == "entry"],
            [(ts, side) for ts, side, intent in rows if intent == "exit"], df)


# Down, a turn up whose first cross comes while the average is still falling, up, then a fall whose first cross comes
# while the average is still rising, a bounce, and a long fall.
PATH = np.concatenate([np.linspace(130, 100, 30), np.linspace(100, 104, 4), np.linspace(104, 140, 25),
                       np.linspace(140, 120, 6), np.linspace(120, 150, 20), np.linspace(150, 90, 30)])


def test_entries_are_states_a_cross_against_the_slope_is_entered_once_the_slope_agrees(instrument):
    entries, _, df = _entries(_small_qr21(), PATH, instrument)
    longs = [ts for ts, side in entries if side == "BUY"]
    assert longs
    for ts in longs:  # every long entry holds both conditions on its candle
        assert df.loc[ts, "close"] > df.loc[ts, "sma"] and df.loc[ts, "sma"] > df.loc[ts, "ago"]
    first = longs[0]
    above = df["close"] > df["sma"]
    cross = above[(above & ~above.shift(1, fill_value=False))].index
    cross = cross[cross < first][-1]  # the close went above the average before the entry...
    assert df.loc[cross, "sma"] <= df.loc[cross, "ago"]  # ...while the average was still falling
    assert above.loc[cross:first].all()  # and stayed above: the entry is the first candle the slope agreed
    agree = df["sma"] > df["ago"]
    assert agree.loc[first] and not agree.loc[cross:first].iloc[:-1].any()


# A seeded random walk: its crosses come both with and against the average's slope, so reversals of both kinds happen.
WALK = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, 300)))


def test_every_entry_holds_its_close_and_slope_conditions(instrument):
    entries, _, df = _entries(_small_qr21(), WALK, instrument)
    assert len(entries) >= 10
    for ts, side in entries:
        c, m, ago = df.loc[ts, "close"], df.loc[ts, "sma"], df.loc[ts, "ago"]
        assert (c > m > ago) if side == "BUY" else (c < m < ago)


def test_a_reversal_whose_other_leg_fails_its_slope_goes_flat(instrument):
    """Advisor 24 Oct: on a reversal the other leg must pass its own slope check, or the Strategy goes flat."""
    entries, exits, df = _entries(_small_qr21(), WALK, instrument)
    opened = set(entries)
    for exit_side, other, falling in (("SELL", "SELL", True), ("BUY", "BUY", False)):  # a long's exit sells, a short's buys
        legs = [ts for ts, side in exits if side == exit_side]
        passes = [ts for ts in legs if bool(df.loc[ts, "sma"] < df.loc[ts, "ago"]) is falling]
        fails = [ts for ts in legs if ts not in passes]
        assert passes and fails
        assert all((ts, other) in opened for ts in passes)  # passed: the other side opens on the same candle
        assert not any((ts, other) in opened for ts in fails)  # failed: out, and flat
