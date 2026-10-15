"""Research tool gaps G1-G4 (Quant Researcher, build cards 14 Oct; tool-gaps.md): what the rule builder lacked to
write the shortlist ideas as one definition each. G1 per-side exits, G2 the rolling max and min of a block's output,
G3 the day of the month in a rule, G4 the previous calendar week's (or day's) high, low and close."""

import copy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.ablation import describe, remove, stopless, units
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.definitions import Compiled, Env, check_definition, definition_hash, to_params
from sleeve_fund.strategies.indicators import BLOCKS, Bollinger

NS = 1_000_000_000
BASE = {"version": 1, "reason": "A test case, written before it is run.",
        "blocks": {"avg": {"kind": "sma", "period": 20}, "atr": {"kind": "atr", "period": 14}},
        "long": {"entry": {"left": "close", "op": "crosses_above", "right": "avg"}},
        "short": {"entry": {"left": "close", "op": "crosses_below", "right": "avg"}}}
LONG_STOP = {"level": {"sub": ["close", {"mul": [3.0, "atr"]}]}}
SHORT_STOP = {"level": {"add": ["close", {"mul": [3.0, "atr"]}]}}


def _with(**changes):
    d = copy.deepcopy(BASE)
    d.update(changes)
    return d


def _walk(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    idx = pd.date_range("2025-10-03 00:01", periods=n, freq="1min", tz="UTC")
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995,
                         "close": c, "volume": 1e3}, index=idx)


def _entries(res):
    f = res.fills.sort_values("ts_last")
    return [str(f.loc[o, "side"]) for o in f.index if res.decisions[o]["intent"] == "entry"]


# --- G1: per-side exits --------------------------------------------------------------------------------------------

PER_SIDE = _with(exits={"time_stop": {"minutes": 600}, "long": {"stop": LONG_STOP}, "short": {"stop": SHORT_STOP}})


def test_g1_one_definition_carries_a_stop_on_each_sides_own_side():
    checked = check_definition(PER_SIDE, bar_minutes=1)
    rules = Compiled(checked)
    assert set(rules.exits[1]["levels"]) == set(rules.exits[-1]["levels"]) == {"stop"}
    assert rules.exits[1]["time_stop"] == rules.exits[-1]["time_stop"] == ("minutes", 600)  # the shared one
    blocks = rules.blocks
    for _ in range(400):
        blocks["atr"].update_raw(101.0, 99.0, 100.0)
    env = Env(ohlcv=(100.0, 101.0, 99.0, 100.0, 1.0), blocks=blocks)
    assert rules.exits[1]["levels"]["stop"](env) < 100.0 < rules.exits[-1]["levels"]["stop"](env)


def test_g1_a_sides_own_exit_replaces_the_shared_one_of_its_kind_for_that_side_only():
    defn = _with(exits={"time_stop": {"minutes": 600}, "stop": LONG_STOP,
                        "short": {"stop": SHORT_STOP, "time_stop": {"bars": 30}}})
    rules = Compiled(check_definition(defn, bar_minutes=1))
    assert rules.exits[1]["time_stop"] == ("minutes", 600) and rules.exits[-1]["time_stop"] == ("bars", 30)


def test_g1_both_legs_trade_with_per_side_stops_where_a_shared_long_stop_holds_every_short_back(instrument):
    df = _walk()
    params = {"market": "perp", "allow_short": True}
    per_side = run_backtest("rules", df, instrument, params={**to_params(PER_SIDE), **params}, bar_minutes=1,
                            half_spread=0)
    sides = _entries(per_side)
    assert any("BUY" in s for s in sides) and any("SELL" in s for s in sides)
    shared = run_backtest("rules", df, instrument, params={**to_params(_with(exits={"stop": LONG_STOP})), **params},
                          bar_minutes=1, half_spread=0)
    assert not any("SELL" in s for s in _entries(shared))  # the long's stop sits above every short's entry


@pytest.mark.parametrize("exits, words", [
    ({"long": {"stop": LONG_STOP}, "short": {"stop": SHORT_STOP}, "middle": {}}, "exits has no middle"),
    ({"long": {"stop": LONG_STOP, "target": LONG_STOP}, "short": {"stop": SHORT_STOP}}, "exits.long has no target"),
    ({"long": "tight"}, "exits.long is a table"),
    ({"long": {"stop": LONG_STOP}}, "long has a stop and the short has none"),
    ({"long": {"exit_at_level": {"level": "avg"}}, "short": {"stop": SHORT_STOP}}, "long has none"),
    ({"short": {"time_stop": {"minutes": 0}}}, "time_stop"),
    ({"long": {"stop": SHORT_STOP}, "short": {"stop": LONG_STOP}}, None),  # wrong way round: allowed, held back
    ({"short": {"stop": {"level": "low"}}, "long": {"stop": LONG_STOP}}, "exits.short.stop: low is never above"),
])
def test_g1_per_side_exits_are_refused_with_their_reason(exits, words):
    if words is None:
        check_definition(_with(exits=exits), bar_minutes=1)
        return
    with pytest.raises(ValueError, match=words):
        check_definition(_with(exits=exits), bar_minutes=1)


def test_g1_a_side_table_needs_its_side():
    long_only = {k: v for k, v in BASE.items() if k != "short"}
    with pytest.raises(ValueError, match="exits.short is for the short side, and the definition has none"):
        check_definition({**long_only, "exits": {"long": {"stop": LONG_STOP}, "short": {"stop": SHORT_STOP}}},
                         bar_minutes=1)


def test_g1_shared_exits_keep_their_hash_and_shape():
    """A definition without side tables is what it was: the same hash (the trials register's key) and exits."""
    defn = _with(exits={"stop": LONG_STOP})
    defn.pop("short")
    assert definition_hash(defn) == definition_hash(copy.deepcopy(defn))
    rules = Compiled(check_definition(defn, bar_minutes=1))
    assert set(rules.exits) == {1} and set(rules.exits[1]["levels"]) == {"stop"}


def test_g1_ablation_takes_each_per_side_exit_out_as_its_own_unit():
    assert [u for u in units(PER_SIDE) if u.startswith("exits")] == [
        "exits.time_stop", "exits.long.stop", "exits.short.stop"]
    status, variant, _ = remove(PER_SIDE, "exits.time_stop")
    assert status == "ran" and "time_stop" not in variant["exits"] and variant["exits"]["long"] == {"stop": LONG_STOP}
    # a single side's stop out would leave the sides disagreeing on a stop, which the builder refuses
    status, _, note = remove(PER_SIDE, "exits.long.stop")
    assert status == "not ablatable" and "has none" in note
    assert describe(PER_SIDE, "exits.short.stop") == "short stop exit"
    assert not stopless(PER_SIDE) and stopless(_with(exits={"short": {"time_stop": {"minutes": 5}}}))


# --- G2: rolling max and min of a block's output -------------------------------------------------------------------

def test_g2_rolling_reads_the_max_and_min_of_its_inputs_last_n_values_the_one_just_closed_included():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 1, 600))
    bb, lo = Bollinger(period=20), BLOCKS["rolling"](period=120)
    widths, mins, maxes = [], [], []
    for c in closes:
        bb.update_raw(c)
        if bb.settled:
            widths.append(bb.values["width"])
            lo.update_raw(widths[-1])
            mins.append(lo.values["min"] if lo.settled else None)
            maxes.append(lo.values["max"] if lo.settled else None)
    ref = pd.Series(widths).rolling(120)
    assert mins[:119] == [None] * 119 and maxes[118] is None
    assert np.allclose(mins[119:], ref.min()[119:], rtol=0, atol=1e-15)
    assert np.allclose(maxes[119:], ref.max()[119:], rtol=0, atol=1e-15)


SQUEEZE = {"version": 1, "reason": "Bollinger width at its 120-candle low, written before it is tested.",
           "blocks": {"bb": {"kind": "bollinger", "period": 20}, "lo": {"kind": "rolling", "period": 120,
                                                                         "input": "bb.width"}},
           "long": {"entry": {"left": "bb.width", "op": "<=", "right": "lo.min"}}}


def test_g2_a_definition_chains_it_on_a_block_output_and_its_warm_up_adds_up():
    checked = check_definition(SQUEEZE, bar_minutes=60)
    assert checked.warmup[None] == 20 + 120
    Compiled(checked)
    with pytest.raises(ValueError, match="period"):
        check_definition({**SQUEEZE, "blocks": {**SQUEEZE["blocks"], "lo": {"kind": "rolling", "period": 0,
                                                                              "input": "bb.width"}}})


def test_g2_squeeze_definition_backtests(instrument):
    res = run_backtest("rules", _walk(1200), instrument, params=to_params(SQUEEZE), bar_minutes=1, half_spread=0)
    assert _entries(res)


# --- G3: day of the month ------------------------------------------------------------------------------------------

def _clock(what, ts, tz=None):
    x = {"time": what, **({"tz": tz} if tz else {})}
    defn = {**BASE, "long": {"entry": {"left": x, "op": ">=", "right": 0}}}
    rules = Compiled(check_definition(defn, bar_minutes=1))
    fn = rules.operand(x)[0]
    return fn(Env(ts=int(ts.timestamp()) * NS))


@pytest.mark.parametrize("close, day, left", [
    (datetime(2026, 10, 15, 12, tzinfo=timezone.utc), 15, 16),
    (datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc), 31, 0),
    (datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc), 1, 29),  # at the candle's close, as hour and weekday are
    (datetime(2026, 2, 28, 6, tzinfo=timezone.utc), 28, 0),
    (datetime(2028, 2, 28, 6, tzinfo=timezone.utc), 28, 1),  # a leap year
    (datetime(2028, 2, 29, 6, tzinfo=timezone.utc), 29, 0),
    (datetime(2026, 4, 30, 6, tzinfo=timezone.utc), 30, 0),
])
def test_g3_day_of_month_and_days_left_in_it(close, day, left):
    assert _clock("day", close) == day and _clock("days_to_month_end", close) == left


def test_g3_follows_a_named_zone_across_a_month_end():
    close = datetime(2026, 5, 31, 23, 30, tzinfo=timezone.utc)  # 00:30 on 1 June in London (summer time)
    assert (_clock("day", close), _clock("day", close, "Europe/London")) == (31, 1)
    assert _clock("days_to_month_end", close, "Europe/London") == 29


def test_g3_turn_of_month_definition_checks_and_builds():
    defn = {"version": 1, "reason": "Month-end flows lift prices from the last day to the 3rd, written first.",
            "long": {"entry": {"any": [{"left": {"time": "days_to_month_end"}, "op": "<=", "right": 0},
                                       {"left": {"time": "day"}, "op": "<=", "right": 3}]},
                     "exit": {"all": [{"left": {"time": "day"}, "op": ">", "right": 3},
                                      {"left": {"time": "days_to_month_end"}, "op": ">", "right": 0}]}}}
    Compiled(check_definition(defn, bar_minutes=1440))
    with pytest.raises(ValueError, match="time is one of hour, weekday, day, days_to_month_end"):
        check_definition({**defn, "long": {"entry": {"left": {"time": "month"}, "op": ">", "right": 0}}})


# --- G4: the previous calendar week's and day's levels -------------------------------------------------------------

def _hourly(start, n, seed=2):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")  # each candle stamped at its close
    return pd.DataFrame({"high": c + rng.uniform(0, 1, n), "low": c - rng.uniform(0, 1, n), "close": c}, index=idx)


def _feed(block, df):
    out = []
    for ts, r in df.iterrows():
        block.update_ohlcv(r.close, r.high, r.low, r.close, 1.0, ts_ns=ts.value)
        out.append(dict(block.values) if block.settled else None)
    return out


@pytest.mark.parametrize("period", ["week", "day"])
def test_g4_previous_period_levels_match_a_calendar_grouping_and_never_read_the_current_one(period):
    df = _hourly("2026-09-02 15:00", 24 * 40)  # starts mid-week, mid-day: the first part period is never published
    got = _feed(BLOCKS["prior_period"](period=period), df)
    # A candle belongs to the period its close falls in: the one closing at 00:00 ends the old one. Weeks start Monday.
    day = (df.index - pd.Timedelta(1, "ns")).floor("D")
    key = day - pd.to_timedelta(day.weekday, unit="D") if period == "week" else day
    agg = df.groupby(key).agg({"high": "max", "low": "min", "close": "last"})
    order = list(agg.index)
    seen_whole = 0
    for i, k in enumerate(order.index(x) for x in key):
        if k < 2:
            assert got[i] is None, df.index[i]  # the first period is a part one, so nothing whole is behind it yet
            continue
        prev = agg.iloc[k - 1]
        assert got[i] == {"high": prev.high, "low": prev.low, "close": prev.close}, df.index[i]
        seen_whole += 1
    assert seen_whole > 24 * 20


def test_g4_week_in_a_named_zone_and_its_warm_up_by_candle_size():
    blk = BLOCKS["prior_period"](period="week", tz="Europe/London")
    assert blk.tz == "Europe/London"
    level = {"version": 1, "reason": "Last calendar week's high, written before it is tested.",
             "blocks": {"wk": {"kind": "prior_period", "period": "week"}},
             "long": {"entry": {"left": "close", "op": ">", "right": "wk.high"}}}
    assert check_definition(level, bar_minutes=60).warmup[None] == 2 * 7 * 24 + 1
    on_days = {**level, "blocks": {"wk": {"kind": "prior_period", "period": "week", "timeframe": "1d"}}}
    assert check_definition(on_days, bar_minutes=60).warmup[1440] == 2 * 7 + 1
    with pytest.raises(ValueError, match="Mars/Olympus"):
        check_definition({**level, "blocks": {"wk": {"kind": "prior_period", "tz": "Mars/Olympus"}}})
    with pytest.raises(ValueError, match="week, day"):
        check_definition({**level, "blocks": {"wk": {"kind": "prior_period", "period": "month"}}})
