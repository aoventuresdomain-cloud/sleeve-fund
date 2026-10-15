"""Research tool gaps G1-G4 (Quant Researcher, build cards 14 Oct; tool-gaps.md): what the rule builder lacked to
write the shortlist ideas as one definition each. G1 per-side exits, G2 the rolling max and min of a block's output,
G3 the day of the month in a rule, G4 the previous calendar week's (or day's) high, low and close."""

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.ablation import describe, remove, stopless, units
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.definitions import (Compiled, Env, check_definition, definition_hash, load_definition_file,
                                                to_params)
from sleeve_fund.strategies.indicators import BLOCKS, Bollinger

NS = 1_000_000_000
DEFINITIONS = Path(__file__).resolve().parent.parent / "configs" / "definitions"
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

PER_SIDE = _with(exits={"long": {"time_stop": {"minutes": 600}, "stop": LONG_STOP},
                         "short": {"time_stop": {"bars": 30}, "stop": SHORT_STOP}})


def test_g1_one_definition_carries_a_stop_on_each_sides_own_side():
    checked = check_definition(PER_SIDE, bar_minutes=1)
    rules = Compiled(checked)
    assert set(rules.exits[1]["levels"]) == set(rules.exits[-1]["levels"]) == {"stop"}
    assert rules.exits[1]["time_stop"] == ("minutes", 600) and rules.exits[-1]["time_stop"] == ("bars", 30)
    blocks = rules.blocks
    for _ in range(400):
        blocks["atr"].update_raw(101.0, 99.0, 100.0)
    env = Env(ohlcv=(100.0, 101.0, 99.0, 100.0, 1.0), blocks=blocks)
    assert rules.exits[1]["levels"]["stop"](env) < 100.0 < rules.exits[-1]["levels"]["stop"](env)


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
    ({"long": {"stop": LONG_STOP}}, "the long has a stop and the short has none: split into two definitions, a "
                                    "long-only one with a stop and a short-only one without, held at 1x"),
    ({"long": {"exit_at_level": {"level": "avg"}}, "short": {"stop": SHORT_STOP}}, "long has none"),
    ({"short": {"time_stop": {"minutes": 0}}}, "exits.short.time_stop"),
    ({"time_stop": {"minutes": 60}, "long": {"stop": LONG_STOP}, "short": {"stop": SHORT_STOP}}, "not both"),
    ({"stop": LONG_STOP, "short": {"stop": SHORT_STOP}}, "not both"),
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


# The Researcher's definitions' content hashes on main 14ed240, before G1-G4: the trials register's keys.
QR_HASHES = {
    "qr01_trend_ema_long": "db887a6fa3a01f1b", "qr01_trend_ema_short": "8ff28a74df7b0372",
    "qr01b_trend_multispeed_long": "e122ddf1851edf73", "qr01b_trend_multispeed_short": "ebeb628bca57ab35",
    "qr01c_trend_donchian_long": "a64268b388237a66", "qr01c_trend_donchian_short": "170c176d9440913e",
    "qr02_dip_long": "ce126dc7f0c4275c", "qr02_dip_short": "d6a770a1cc522b77",
    "qr03_squeeze_long": "95545de0d400e19f", "qr03_squeeze_short": "8bb0e50ac6fd3c0f",
    "qr10_capitulation_long": "62a1ee8da26544c1", "qr11_exhaustion_long": "8ad588bcd437b26a",
    "qr11_exhaustion_short": "daa1bea4ae612937", "qr12_overnight_breakout_long": "aa3a666c41d9381c",
    "qr13_level_continuation_long": "18e1ac88db4b05b8", "qr13_level_continuation_short": "8ee488792d38fdd4",
}


def test_g1_existing_definitions_keep_their_hashes():
    got = {p.stem: definition_hash(load_definition_file(p)) for p in sorted(DEFINITIONS.glob("qr*.toml"))}
    assert got == QR_HASHES


def _fills_digest(res) -> tuple[int, str]:
    f = res.fills.sort_values("ts_last")
    rows = [[str(f.loc[o, "ts_last"]), res.decisions[o]["intent"], str(f.loc[o, "side"]), str(f.loc[o, "filled_qty"]),
             str(f.loc[o, "avg_px"]), str(f.loc[o].get("trigger_price"))] for o in f.index]
    return len(rows), hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


SHARED = {**{k: v for k, v in BASE.items() if k != "short"},
          "exits": {"time_stop": {"minutes": 90}, "stop": LONG_STOP,
                    "trail": {"level": {"sub": ["close", {"mul": [4.0, "atr"]}]}, "ratchet": True}}}


@pytest.mark.parametrize("defn, params, want", [
    (SHARED, {}, (122, "4c9caef668e53419")),
    ({**BASE, "exits": {"time_stop": {"bars": 45}}}, {"market": "perp", "allow_short": True}, (107, "8365dd00a2b62c2d")),
])
def test_g1_a_definition_with_shared_exits_loads_and_trades_as_it_did_before(defn, params, want, instrument):
    """HoQA: the fills (time, intent, side, quantity, price, trigger) of shared-exit definitions, digested on main
    14ed240 before G1-G4, are unchanged."""
    res = run_backtest("rules", _walk(), instrument, params={**to_params(defn), **params}, bar_minutes=1,
                       half_spread=0)
    assert _fills_digest(res) == want


def test_g1_ablation_takes_each_per_side_exit_out_as_its_own_unit():
    assert [u for u in units(PER_SIDE) if u.startswith("exits")] == [
        "exits.long.time_stop", "exits.long.stop", "exits.short.time_stop", "exits.short.stop"]
    status, variant, _ = remove(PER_SIDE, "exits.long.time_stop")
    assert status == "ran" and variant["exits"]["long"] == {"stop": LONG_STOP}
    # a single side's stop out would leave the sides disagreeing on a stop, which the builder refuses
    status, _, note = remove(PER_SIDE, "exits.long.stop")
    assert status == "not ablatable" and "has none" in note
    assert describe(PER_SIDE, "exits.short.stop") == "short stop exit"
    assert not stopless(PER_SIDE) and stopless(_with(exits={"short": {"time_stop": {"minutes": 5}}}))


# --- G2: rolling max and min of a block's output -------------------------------------------------------------------

def test_g2_rolling_reads_the_max_and_min_of_its_inputs_last_n_values_the_one_just_closed_included():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 1, 600))
    bb, lo, hi = Bollinger(period=20), BLOCKS["rolling_min"](period=120), BLOCKS["rolling_max"](period=120)
    widths, mins, maxes = [], [], []
    for c in closes:
        bb.update_raw(c)
        if bb.settled:
            widths.append(bb.values["width"])
            lo.update_raw(widths[-1])
            hi.update_raw(widths[-1])
            mins.append(lo.value if lo.settled else None)
            maxes.append(hi.value if hi.settled else None)
    ref = pd.Series(widths).rolling(120)
    assert mins[:119] == [None] * 119 and maxes[:119] == [None] * 119  # fewer than N values: no value
    assert np.array_equal(mins[119:], ref.min()[119:]) and np.array_equal(maxes[119:], ref.max()[119:])


SQUEEZE = {"version": 1, "reason": "Bollinger width at its 120-candle low, written before it is tested.",
           "blocks": {"bb": {"kind": "bollinger", "period": 20}, "lo": {"kind": "rolling_min", "period": 120,
                                                                         "input": "bb.width"}},
           "long": {"entry": {"left": "bb.width", "op": "<=", "right": "lo"}}}


def test_g2_a_definition_chains_it_on_a_block_output_and_its_warm_up_adds_up():
    checked = check_definition(SQUEEZE, bar_minutes=60)
    assert checked.warmup[None] == 20 + 120
    Compiled(checked)
    for bad, words in (({"kind": "rolling_min", "period": 0, "input": "bb.width"}, "period"),
                       ({"kind": "rolling_max", "period": 5, "input": "bb.middle"}, "no output 'middle'"),
                       ({"kind": "rolling_max", "period": 5, "input": "nothing"}, "not a block")):
        with pytest.raises(ValueError, match=words):
            check_definition({**SQUEEZE, "blocks": {**SQUEEZE["blocks"], "lo": bad}})


def test_g2_squeeze_definition_backtests(instrument):
    res = run_backtest("rules", _walk(1200), instrument, params=to_params(SQUEEZE), bar_minutes=1, half_spread=0)
    assert _entries(res)


# --- G3: day of the month ------------------------------------------------------------------------------------------

def _clock(what, ts, tz=None):
    x = {"time": what, **({"tz": tz} if tz else {})}
    defn = {**BASE, "long": {"entry": {"left": x, "op": ">=", "right": 0 if what != "day_of_month" else 1}}}
    rules = Compiled(check_definition(defn, bar_minutes=1))
    fn = rules.operand(x)[0]
    return fn(Env(ts=int(ts.timestamp()) * NS))


@pytest.mark.parametrize("close, day, left", [
    (datetime(2026, 10, 15, 12, tzinfo=timezone.utc), 15, 16),
    (datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc), 31, 0),
    # Advisor: a candle closing at month end is read at its close, as hour and weekday are: 00:00 is the 1st
    (datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc), 1, 29),
    (datetime(2026, 2, 28, 6, tzinfo=timezone.utc), 28, 0),
    (datetime(2028, 2, 28, 6, tzinfo=timezone.utc), 28, 1),  # a leap year
    (datetime(2028, 2, 29, 6, tzinfo=timezone.utc), 29, 0),
    (datetime(2026, 4, 30, 6, tzinfo=timezone.utc), 30, 0),
])
def test_g3_day_of_month_and_days_left_in_it(close, day, left):
    assert _clock("day_of_month", close) == day and _clock("days_to_month_end", close) == left


def test_g3_follows_a_named_zone_across_a_month_end_and_the_bst_to_gmt_change():
    close = datetime(2026, 5, 31, 23, 30, tzinfo=timezone.utc)  # 00:30 on 1 June in London (summer time)
    assert (_clock("day_of_month", close), _clock("day_of_month", close, "Europe/London")) == (31, 1)
    assert _clock("days_to_month_end", close, "Europe/London") == 29
    # Summer time ends at 01:00 UTC on 25 Oct 2026: 00:30 UTC is 01:30 BST, 23:30 UTC on the 31st is 23:30 GMT
    assert _clock("day_of_month", datetime(2026, 10, 24, 23, 30, tzinfo=timezone.utc), "Europe/London") == 25
    assert _clock("day_of_month", datetime(2026, 10, 25, 23, 30, tzinfo=timezone.utc), "Europe/London") == 25
    assert _clock("days_to_month_end", datetime(2026, 10, 31, 23, 30, tzinfo=timezone.utc), "Europe/London") == 0


TURN = {"version": 1, "reason": "Month-end flows lift prices from the last day to the 3rd, written first.",
        "long": {"entry": {"any": [{"left": {"time": "days_to_month_end"}, "op": "<=", "right": 0},
                                   {"left": {"time": "day_of_month"}, "op": "<=", "right": 3}]},
                 "exit": {"all": [{"left": {"time": "day_of_month"}, "op": ">", "right": 3},
                                  {"left": {"time": "days_to_month_end"}, "op": ">", "right": 0}]}}}


def test_g3_turn_of_month_definition_checks_and_builds():
    Compiled(check_definition(TURN, bar_minutes=1440))


@pytest.mark.parametrize("left, right, words", [
    ({"time": "month"}, 0, "time is one of hour, weekday, day_of_month, days_to_month_end"),
    ({"time": "day_of_month"}, 0, "day_of_month runs 1 to 31"),
    ({"time": "day_of_month"}, 32, "day_of_month runs 1 to 31"),
    ({"time": "day_of_month"}, -1, "day_of_month runs 1 to 31"),
    ({"time": "days_to_month_end"}, 31, "days_to_month_end runs 0 to 30"),
    ({"time": "days_to_month_end"}, -1, "days_to_month_end runs 0 to 30"),
    (5, {"time": "days_to_month_end"}, None),  # either way round
    (40, {"time": "days_to_month_end"}, "days_to_month_end runs 0 to 30"),
])
def test_g3_a_number_outside_the_month_is_refused(left, right, words):
    defn = {**TURN, "long": {"entry": {"left": left, "op": ">", "right": right}}}
    if words is None:
        check_definition(defn)
        return
    with pytest.raises(ValueError, match=words):
        check_definition(defn)


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


@pytest.mark.parametrize("period, tz, start", [
    ("week", None, "2026-09-02 15:00"),
    ("day", None, "2026-09-02 15:00"),
    ("week", "Europe/London", "2026-09-30 15:00"),  # through the BST to GMT change, 25 Oct 2026
    ("day", "Europe/London", "2026-10-15 15:00"),  # 25 Oct in London has 25 hourly candles
])
def test_g4_prior_period_is_exact_from_the_first_whole_period_has_no_value_before_and_never_reads_the_current_one(
        period, tz, start):
    """HoE (TEC-G4-WARMUP): from the first whole period on it is exact, against a calendar grouping; before it, no
    value. The period the data starts in is a part one, so it is never published."""
    df = _hourly(start, 24 * 40)
    got = _feed(BLOCKS["prior_period"](period=period, **({"tz": tz} if tz else {})), df)
    # A candle belongs to the period its close falls in: the one closing at 00:00 ends the old one. Weeks start Monday.
    local = df.index - pd.Timedelta(1, "ns")
    day = (local.tz_convert(tz) if tz else local).tz_localize(None).floor("D")
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
    if tz and period == "day":
        assert (key == pd.Timestamp("2026-10-25")).sum() == 25


def test_g4_a_candle_closing_at_monday_0000_utc_belongs_to_the_week_before():
    """Advisor: the candle closing exactly at 00:00 UTC on a Monday ends the old week, so the levels move on the
    next close, not on it."""
    df = _hourly("2026-09-30 01:00", 24 * 20)
    got = _feed(BLOCKS["prior_period"](period="week"), df)
    at = {ts: v for ts, v in zip(df.index, got)}
    monday = pd.Timestamp("2026-10-19 00:00", tz="UTC")
    week = df[(df.index > pd.Timestamp("2026-10-12 00:00", tz="UTC")) & (df.index <= monday)]
    assert at[monday] == at[monday - pd.Timedelta(1, "h")] != at[monday + pd.Timedelta(1, "h")]
    assert at[monday + pd.Timedelta(1, "h")] == {"high": week.high.max(), "low": week.low.min(),
                                                  "close": df.loc[monday, "close"]}


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
    with pytest.raises(ValueError, match="day, week"):
        check_definition({**level, "blocks": {"wk": {"kind": "prior_period", "period": "month"}}})
