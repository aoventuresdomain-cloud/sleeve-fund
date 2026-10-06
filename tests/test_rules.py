"""v2 P1-5: the rule builder's own checks, beside QA's acceptance file (test_p1_5_xfails.py): what a definition is
refused for, what its hash follows, confirm-within-N, warm-up, and legacy decisions on unsettled indicators."""

import copy

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.definitions import Compiled, Env, check_definition, definition_hash, to_params
from sleeve_fund.strategies.rules import Rules, RulesConfig

M = 60_000_000_000
BASE = {"version": 1, "reason": "A test case, written before it is run.", "blocks": {"rsi": {"kind": "rsi"}},
        "long": {"entry": {"left": "rsi", "op": "<", "right": 30}}}


def _with(**changes):
    d = copy.deepcopy(BASE)
    d.update(changes)
    return d


@pytest.mark.parametrize("defn, words", [
    (_with(reason=" "), "reason"),
    (_with(blocks={"a": {"kind": "atr", "input": "close"}}), "reads several prices"),
    (_with(blocks={"r": {"kind": "rsi", "timeframe": "1h"}, "s": {"kind": "sma", "input": "r"}}), "another timeframe"),
    (_with(blocks={"close": {"kind": "rsi"}}), "not a price field"),
    (_with(long={"entry": {"left": {"time": "hour", "tz": "Mars/Olympus"}, "op": ">", "right": 8}}), "Mars/Olympus"),
    (_with(exits={"time_stop": {"bars": 6, "count": "hours"}}), "time_stop"),
    (_with(exits={"time_stop": {"minutes": 0}}), "time_stop"),
    (_with(costs={"breakout_slippage_bp": 25}), "breakout = true"),
    (_with(long={"entry": {"left": "rsi", "op": "<", "right": 30},
                 "exit": {"setup": {"left": "rsi", "op": ">", "right": 70}, "trigger": {"left": "rsi", "op": "<",
                          "right": 60}, "expire_after": {"bars": 4, "timeframe": "15m"}}}), "entry rule"),
    (_with(blocks={"r": {"kind": "rsi", "timeframe": "5m"}}), "whole multiple"),
    (_with(blocks={"rsi": {"kind": "rsi"}, "dc": {"kind": "donchian", "period": 20}},
           exits={"stop": {"level": "dc.upper"}}),
     "never below the close, so as the long side's stop"),
    (_with(short={"entry": {"left": "close", "op": ">", "right": 0}}, exits={"trail": {"level": "low"}}),
     "never above the close, so as the short side's stop"),
    (_with(blocks={"rsi": {"kind": "rsi"}, "dc": {"kind": "donchian", "period": 20}},
           short={"entry": {"left": "close", "op": ">", "right": 0}},
           exits={"exit_at_level": {"level": "dc.lower"}}), "never above the close, so as the short side's stop"),
])
def test_a_definition_is_refused_with_its_reason(defn, words):
    with pytest.raises(ValueError, match=words.replace("(", r"\(")):
        check_definition(defn, bar_spec="15-MINUTE-LAST-INTERNAL")


def test_the_hash_ignores_name_reason_and_number_spelling_but_not_settings():
    same = definition_hash(BASE)
    assert definition_hash({**BASE, "name": "other", "reason": "Reworded."}) == same
    assert definition_hash(_with(long={"entry": {"left": "rsi", "op": "<", "right": 30.0}})) == same
    assert definition_hash(_with(long={"entry": {"left": "rsi", "op": "<", "right": 31}})) != same
    assert definition_hash(BASE, venue="KRAKEN") == same  # a venue whose day starts at 00:00 UTC


def test_a_strategy_refuses_a_definition_changed_after_it_was_copied_in(instrument):
    from sleeve_fund.data import bar_type_for

    params = to_params(BASE)
    params["definition"]["long"]["entry"]["right"] = 25
    with pytest.raises(ValueError, match="changed after it was copied in"):
        RulesConfig(instrument_id=instrument.id, bar_type=bar_type_for(instrument, 1), assumed_taker_fee=0.001,
                    **params)


def test_confirm_within_n_counts_on_the_event_candle_or_the_next_n_and_only_once():
    defn = _with(blocks={}, long={"entry": {"event": {"left": "close", "op": ">", "right": 100},
                                            "confirm": {"left": "volume", "op": ">", "right": 5}, "within": 2}})
    rules = Compiled(check_definition(defn))
    node, env = rules.sides[1]["entry"], Env()
    got = []
    for k, (close, vol) in enumerate([(101, 1), (99, 1), (99, 9), (99, 9), (101, 1), (99, 1), (99, 1), (99, 9)]):
        env.ohlcv, env.ts, env.bar = (close, close, close, close, vol), (k + 1) * M, k + 1
        node.tick(env)
        if node.test(env):
            got.append(k)
            node.consume()  # the leg it opened spends the event
    assert got == [2]  # confirmed two candles after the event; the event at 4 is never confirmed within 2


def test_warm_up_counts_chains_and_slower_candles_and_the_model_loads_enough_for_both():
    defn = _with(blocks={"rsi": {"kind": "rsi", "period": 14}, "avg": {"kind": "sma", "period": 3, "input": "rsi"},
                         "trend": {"kind": "sma", "period": 50, "timeframe": "4h"}},
                 long={"entry": {"left": "avg", "op": "crosses_above", "right": 30}})
    checked = check_definition(defn, bar_spec="15-MINUTE-LAST-INTERNAL")
    assert checked.warmup == {None: 140 + 3 + 1, 240: 50}  # the chain, plus the candle a cross looks back to
    params = {"definition": defn}
    assert Rules.slower_needs(params) == {240: 50}
    assert Rules.warmup_needed(params, 15) == 51 * 240 // 15  # one more: the part candle the warm-up starts in


def _walk(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    idx = pd.date_range("2025-10-03 00:01", periods=n, freq="1min", tz="UTC")
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995,
                         "close": c, "volume": 1e3}, index=idx)


def test_legacy_decisions_before_the_warm_up_are_flagged_and_a_preloaded_warm_up_has_none(instrument):
    df = _walk()
    cold = run_backtest("rsi_bands", df, instrument, bar_minutes=1, half_spread=0)
    flagged = [o for o in cold.fills.index if cold.decisions[o].get("unsettled")]
    early = [o for o in cold.fills.index if cold.fills.loc[o, "ts_last"] < df.index[139]]
    assert flagged == early and cold.unsettled_fills == len(early) > 0
    warm = run_backtest("rsi_bands", df.iloc[300:], instrument, bar_minutes=1, half_spread=0,
                        warmup_prices=df.iloc[:300])
    assert warm.unsettled_fills == 0 and len(warm.fills) > 0
    assert warm.fills["ts_last"].min() >= df.index[300]  # the preload is never traded
    with pytest.raises(ValueError, match="before the backtest"):
        run_backtest("rsi_bands", df.iloc[300:], instrument, bar_minutes=1, warmup_prices=df.iloc[:301])


def test_an_entry_held_back_for_its_stop_level_still_carries_the_lineage_payload(instrument):
    """The rule holds from the first candle, the stop's channel needs 20 to settle: the entry waits for it, and the
    order says which rule it carries out, with the values it was sent on."""
    defn = _with(blocks={"dc": {"kind": "donchian", "period": 20}},
                 long={"entry": {"left": "close", "op": ">", "right": 0}}, exits={"stop": {"level": "dc.lower"}})
    df = _walk(300)
    res = run_backtest("rules", df, instrument, params=to_params(defn), bar_minutes=1, half_spread=0)
    entry = next(o for o in res.fills.sort_values("ts_last").index if res.decisions[o]["intent"] == "entry")
    sig = res.decisions[entry]["signal"]
    assert res.fills.loc[entry, "ts_last"] == df.index[19]
    assert sig["v"] == 1 and sig["rule"] == "long.entry" and set(sig["blocks"]) == {"dc.upper", "dc.lower", "dc.mid"}


def test_the_rules_model_stays_off_the_pickers_until_the_form_can_choose_a_definition(tmp_path):
    from sleeve_fund.dashboard import pipeline
    from sleeve_fund.dashboard.app import _strategy_choices

    assert "rules" not in {s["name"] for s in _strategy_choices()}
    assert "rules" not in {s["name"] for s in pipeline.strategies(tmp_path, [])}


def test_a_channel_is_the_n_most_recent_closed_bars_the_one_just_closed_included():
    """Advisor 17:07: at bar i's close the channel takes in bar i, so a level resting in bar i+1 is that bar's n
    predecessors; the library block (as the donchian model trades it) still leaves bar i out."""
    from sleeve_fund.strategies.definitions import BUILDER_BLOCKS
    from sleeve_fund.strategies.indicators import Donchian

    built, library = BUILDER_BLOCKS["donchian"](3), Donchian(3)
    for high, low in [(10, 9), (12, 8), (11, 7), (15, 10)]:
        built.update_raw(high, low, low)
        library.update_raw(high, low, low)
    assert (built.values["upper"], built.values["lower"]) == (15, 7)  # bars 1-3
    assert (library.values["upper"], library.values["lower"]) == (12, 7)  # bars 0-2
    defn = _with(blocks={"dc": {"kind": "donchian", "period": 3}},
                 long={"entry": {"left": "close", "op": ">", "right": "dc.upper"}})
    assert built.settled and check_definition(defn).warmup[None] == 3


def test_the_slower_warm_up_covers_its_candles_when_it_starts_inside_one():
    """The warm-up Rules asks for builds the slower candles its blocks take even when it starts part way through a
    slower candle, which SlowerCandles drops (Code Reviewer on a69618a), at a midnight and a non-midnight anchor."""
    from sleeve_fund.strategies.timeframes import SlowerCandles

    defn = _with(blocks={"rsi": {"kind": "rsi"}, "trend": {"kind": "sma", "period": 50, "timeframe": "4h"}})
    need = Rules.warmup_needed({"definition": defn}, 15)
    for anchor in (0, 90):
        for start in range(0, 240, 15):  # every 15-minute offset into a 4h candle
            s = SlowerCandles(240, 15, anchor=anchor)
            first = (start + 15) * M
            closed = sum(s.update(1, 1, 1, 1, 1, first + i * 15 * M) is not None for i in range(need))
            assert closed >= 50, (anchor, start, closed)


def test_a_restarted_leg_times_its_stop_from_the_journals_entry_candle(instrument):
    """After a restart the wall-clock time stop runs from the entry candle the journal kept, not from the candles
    counted back from the next decision, which comes out late when candles were missing during the hold (5.2)."""
    from sleeve_fund.data import bar_type_for

    defn = _with(exits={"time_stop": {"minutes": 60}})
    s = Rules(RulesConfig(instrument_id=instrument.id, bar_type=bar_type_for(instrument, 15), assumed_taker_fee=0.001,
                          **to_params(defn)))
    entry = 1_000 * 15 * M
    s._resume_entry_ns = entry
    s.resume_leg(1, 1)  # one candle held by the count, though three were due: two went missing
    assert s._leg_ts == entry
    s.env.ts = entry + 60 * M
    assert s._time_up(15 * M)


def _flat(n=30, spike=10, at=106.0):
    """1-minute bars flat at 100 but one closing at `at`."""
    idx = pd.date_range("2025-10-03 00:01", periods=n, freq="1min", tz="UTC")
    c = np.full(n, 100.0)
    c[spike] = at
    return pd.DataFrame({"open": 100.0, "high": np.maximum(c, 100.0), "low": np.minimum(c, 100.0), "close": c,
                         "volume": 1e3}, index=idx)


def _trades(res):
    f = res.fills.sort_values("ts_last")
    return [(res.decisions[o]["intent"] == "entry", f.loc[o, "side"].name if hasattr(f.loc[o, "side"], "name")
             else str(f.loc[o, "side"]), f.loc[o, "ts_last"]) for o in f.index]


@pytest.mark.parametrize("exits", [{}, {"time_stop": {"bars": 10, "count": "bars"}}])
def test_a_leg_ended_at_the_close_reopens_its_side_no_sooner_than_the_next_close(exits, instrument):
    """Advisor 22:30 (#161 MAJOR): an always-true entry whose exit fires on one candle's close makes one round trip
    and opens again on the next close, never closing and reopening at one price on the exit candle. The same for a
    time stop counted in candles (rsi_cross's port)."""
    df = _flat()
    exit_rule = {"exit": {"left": "close", "op": ">", "right": 105}} if not exits else {}
    defn = _with(blocks={}, long={"entry": {"left": "close", "op": ">", "right": 0}, **exit_rule}, exits=exits)
    res = run_backtest("rules", df, instrument, params=to_params(defn), bar_minutes=1, half_spread=0)
    trades = [(entry, ts) for entry, _, ts in _trades(res)]
    assert trades[:3] == [(True, df.index[0]), (False, df.index[10]), (True, df.index[11])]
    assert res.reentries_on_exit_candle == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_the_other_side_opens_on_the_exit_candle_only_where_it_declares_reverse(reverse, instrument):
    df = _flat()
    defn = _with(blocks={}, long={"entry": {"left": "close", "op": "<", "right": 105},
                                  "exit": {"left": "close", "op": ">", "right": 105}},
                 short={"entry": {"left": "close", "op": ">", "right": 105}, "reverse": reverse})
    res = run_backtest("rules", df, instrument, params={**to_params(defn), "market": "perp", "allow_short": True},
                       bar_minutes=1, half_spread=0)
    sells = [ts for _, side, ts in _trades(res) if "SELL" in side]
    assert (sells[0], len(sells)) == (df.index[10], 2 if reverse else 1)  # with reverse: the long's exit, then the short


@pytest.mark.parametrize("defn, words", [
    (_with(long={"entry": {"left": "rsi", "op": "<", "right": 30}, "reverse": True}), "needs a short side"),
    (_with(long={"entry": {"left": "rsi", "op": "<", "right": 30}, "reverse": "yes"}), "reverse is true or false"),
])
def test_reverse_is_refused_without_the_side_it_reverses_from(defn, words):
    with pytest.raises(ValueError, match=words):
        check_definition(defn, bar_spec="15-MINUTE-LAST-INTERNAL")


def test_re_entries_on_an_exit_candle_count_entries_in_a_candle_where_an_exit_filled():
    from sleeve_fund.research.runner import reentries_on_exit_candle

    t = pd.Timestamp("2025-10-03 00:15", tz="UTC")
    fills = pd.DataFrame({"ts_last": [t, t + pd.Timedelta(minutes=7), t + pd.Timedelta(minutes=15),
                                      t + pd.Timedelta(minutes=45)]}, index=["e1", "stop", "e2", "e3"])
    decisions = {"e1": {"intent": "entry"}, "stop": {"intent": "stop_loss"}, "e2": {"intent": "entry"},
                 "e3": {"intent": "entry"}}
    assert reentries_on_exit_candle(fills, decisions, 15) == 1  # e2: the stop filled inside its candle
    assert reentries_on_exit_candle(fills, decisions, 5) == 0
    assert reentries_on_exit_candle(None, {}, 15) == 0


def test_after_a_stop_inside_a_candle_the_rules_may_enter_again_at_its_close_and_it_is_counted(instrument):
    """Advisor 22:30 (QA P1-5-X2): a stop or target filled inside the candle ends the leg; the entry rules are read
    afresh at that candle's close and may open the same side there. The backtest counts it."""
    df = _flat(n=30, spike=10, at=100.0)
    df.iloc[10, df.columns.get_loc("low")] = 98.0  # through the 1 % stop inside the minute, back to 100 at its close
    defn = _with(blocks={}, long={"entry": {"left": "close", "op": ">", "right": 0}})
    res = run_backtest("rules", df, instrument, params={**to_params(defn), "stop_loss": 0.01}, bar_minutes=1,
                       half_spread=0)
    entries = [ts for entry, _, ts in _trades(res) if entry]
    assert entries[:2] == [df.index[0], df.index[10]] and res.reentries_on_exit_candle == 1
