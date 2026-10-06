"""QA Tester 2: acceptance tests for R2, the rule builder's first-touch condition, written BEFORE the build as STRICT
xfails. Platform Engineer 1 makes each pass, then removes its mark, before QA round 1. Copy this file into tests/ to
run it (it uses the suite's own fixtures: `instrument`).

Sources (every trading-behaviour expectation cites one):
- [D]   v2/r2-first-touch-design.md (PE1): form, levels fixed at candle k-1's close, which way "reached" is read,
        order, window, missing minutes, report, hash, lineage.
- [HoE] the HoE's Done-when (22:01): (1) judged on the 1m bars inside the decision candle; (2) both levels in one 1m
        bar -> the adverse one first, and the report shows how often; (3) checked, hashed, lineage-tagged;
        (4) look-ahead safe, pinned by a case where Y is hit after the decision time; (5) hand-computed reference.
- [ADV] advisor-rulings.md, 6 Oct 22:11 (R2 first-touch edges), 19:40 (exact touch); and 14:10 (R3-R6 line): "First-touch within one bar: assume adverse first."

Interfaces ASSUMED (PE1 to confirm or correct the names; the behaviour is what is pinned):
- the rule is `{"first_touch": {"reach": X, "before": Y}}` anywhere a condition is allowed;
- `run_backtest(...)` result carries `res.first_touch`: {rule path: {"judged", "true", "same_minute", "unknown"}} (the
  share of candles where either level was reached is derived by the test: same_minute / (reached candles));
- an order's `signal` carries `signal["first_touch"][rule path]` = {"x", "y", "x_minute", "y_minute", "same_minute"}
  with minutes as ISO strings (the minute's CLOSE time, as the bars are indexed) or None when never reached;
- running on a decision candle longer than a minute without `exec_prices` at 1 minute raises ValueError naming
  "1-minute".
Every interface lookup that is not built yet is turned into an AssertionError ("not built: ..."), so each test fails
strictly on its own assertion and the mark is `xfail(strict=True, raises=AssertionError)`.

Missing minutes follow the Advisor's 22:11 ruling (advisor-rulings.md "R2 first-touch edges"), which supersedes the
design doc: the candle's own high/low first, a gap matters only when both levels are in range and it comes before the
first reach; unknown and same-minute are FALSE on the entry side and TRUE in exits.

Layout of the fixed scenario: 15-minute decision candles over flat 100.0 prices. Candle c0 (minutes closing 00:01-00:15)
is the warm-up candle; c1.. are judged. X = close + 1 = 101, Y = close - 1 = 99, read at the PREVIOUS candle's close.
A candle "fires" when an entry fills at its close (the entry is placed on the decision candle's close).
"""

import copy

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.definitions import check_definition, definition_hash, to_params

R2 = pytest.mark.xfail(strict=True, raises=AssertionError, reason="R2 first-touch not built")

T0 = pd.Timestamp("2025-10-03 00:00", tz="UTC")
SPEC = "15-MINUTE-LAST-INTERNAL"
UP, DOWN = {"add": ["close", 1]}, {"sub": ["close", 1]}
BASE = {"version": 1, "reason": "An R2 acceptance case, written before it is built.", "blocks": {}}


def _defn(rule=None, **side):
    rule = rule or {"first_touch": {"reach": UP, "before": DOWN}}
    return {**BASE, "long": {"entry": rule, **side}}


def _minutes(n_candles, tf=15, spikes=(), closes=None):
    """Flat 100.0 minutes closing 00:01.. ; spikes {(candle, minute 1..tf): (high, low)}; closes {candle: last close}
    sets that candle's final minute close (default 100). The decision candles are built from THESE minutes."""
    idx = pd.date_range(T0 + pd.Timedelta(minutes=1), periods=n_candles * tf, freq="1min")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e3}, index=idx)
    for (k, m), (hi, lo) in dict(spikes).items():
        i = k * tf + m - 1
        df.iloc[i, df.columns.get_loc("high")] = hi
        df.iloc[i, df.columns.get_loc("low")] = lo
    for k, c in (closes or {}).items():
        i = k * tf + tf - 1
        df.iloc[i, df.columns.get_loc("close")] = c
        df.iloc[i, df.columns.get_loc("high")] = max(df.iloc[i]["high"], c)
        df.iloc[i, df.columns.get_loc("low")] = min(df.iloc[i]["low"], c)
    return df


def _without(m, drop, tf=15):
    """The minutes the engine is actually given: `drop` {(candle, minute)} never arrive. The candle's own high and
    low (decision bars) still include them, as they come from the venue's candle, not from the minutes."""
    return m.iloc[[i for i in range(len(m)) if (i // tf, i % tf + 1) not in set(drop)]]


def _decisions(m, tf=15):
    return m.resample(f"{tf}min", closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})


def _run(defn, m, instrument, tf=15, drop=(), exec_prices=None, perp=False, **kw):
    ex = exec_prices if exec_prices is not None else _without(m, drop, tf)
    try:
        params = {**to_params(defn), **({"market": "perp", "allow_short": True} if perp else {})}  # shorts need a perp
        return run_backtest("rules", _decisions(m, tf), instrument, params=params, bar_minutes=tf,
                            exec_prices=ex if tf > 1 else None, exec_minutes=1, half_spread=0, **kw)
    except ValueError as e:  # an unknown rule is refused by the checker today
        raise AssertionError(f"not built: {e}") from e


def _fired(res):
    """Candle numbers (c1 = 1) on whose close an entry filled."""
    return sorted(int((res.fills.loc[o, "ts_last"] - T0) / pd.Timedelta(minutes=15)) - 1  # candle k closes T0 + 15(k+1)
                  for o in res.fills.index if res.decisions[o]["intent"] == "entry")


def _ft(res, rule="long.entry"):
    got = getattr(res, "first_touch", None)
    assert got is not None and rule in got, "not built: res.first_touch[rule]"
    return got[rule]


def _lineage(res, rule="long.entry"):
    o = next(o for o in res.fills.sort_values("ts_last").index if res.decisions[o]["intent"] == "entry")
    sig = res.decisions[o]["signal"]
    assert "first_touch" in sig and rule in sig["first_touch"], "not built: signal['first_touch'][rule]"
    return sig["first_touch"][rule]


def _checked(defn):
    try:
        return check_definition(defn, bar_spec=SPEC)
    except ValueError as e:
        raise AssertionError(f"not built: {e}") from e


def _at(k, minute):
    return (T0 + pd.Timedelta(minutes=15 * k + minute)).isoformat()


# ------------------------------------------------------------------------------------------------------ order (D, HoE 1)
@pytest.mark.parametrize("name, spikes, fires", [
    ("X then Y -> true", {(1, 3): (101.5, 100.0), (1, 7): (100.0, 98.5)}, True),
    ("Y then X -> false", {(1, 3): (100.0, 98.5), (1, 7): (101.5, 100.0)}, False),
    ("only X -> true", {(1, 9): (101.2, 100.0)}, True),
    ("only Y -> false", {(1, 9): (100.0, 98.8)}, False),
    ("neither level reached -> false", {(1, 5): (100.9, 99.1)}, False),
    ("touching the level exactly counts as reached", {(1, 5): (101.0, 100.0)}, True),
])
def test_the_first_minute_to_reach_a_level_decides(name, spikes, fires, instrument):
    res = _run(_defn(), _minutes(2, spikes=spikes), instrument)
    assert _fired(res) == ([1] if fires else []), name


def test_both_levels_in_one_minute_is_false_and_counted_as_same_minute(instrument):
    """[ADV] adverse first. The same minute reaches 101.5 and 98.5: the rule is false, and the report says so."""
    m = _minutes(2, spikes={(1, 6): (101.5, 98.5)})
    res = _run(_defn(), m, instrument)
    ft = _ft(res)
    assert _fired(res) == [] and (ft["judged"], ft["true"], ft["same_minute"], ft["unknown"]) == (1, 0, 1, 0)


def test_a_decision_candle_of_one_minute_is_its_own_minute(instrument):
    """[D] 1-minute decision candle: a bar that reaches both is a same-minute case; one that reaches only X is true.
    The levels are the previous 1-minute close +/- 1."""
    m = _minutes(4, tf=1, spikes={(2, 1): (101.5, 98.5), (3, 1): (101.5, 100.0)}, )
    res = _run(_defn(), m, instrument, tf=1)
    ft = _ft(res)
    fired = sorted(res.fills.loc[o, "ts_last"] for o in res.fills.index if res.decisions[o]["intent"] == "entry")
    assert fired == [T0 + pd.Timedelta(minutes=4)]  # candle 3 closes T0 + 4 minutes and ft["same_minute"] == 1 and ft["true"] == 1


@pytest.mark.parametrize("name, rule, spikes, fires", [
    ("X below the close, reached by a low, first -> true",
     {"first_touch": {"reach": DOWN, "before": UP}}, {(1, 4): (100.0, 98.9), (1, 8): (101.2, 100.0)}, True),
    ("X below the close, Y above reached first by a high -> false",
     {"first_touch": {"reach": DOWN, "before": UP}}, {(1, 4): (101.2, 100.0), (1, 8): (100.0, 98.9)}, False),
    ("a level below the close is not reached by a high",
     {"first_touch": {"reach": DOWN, "before": {"add": ["close", 5]}}}, {(1, 4): (100.4, 99.6)}, False),
    ("a level above the close is not reached by a low",
     {"first_touch": {"reach": UP, "before": {"sub": ["close", 5]}}}, {(1, 4): (100.4, 99.6)}, False),
], ids=lambda v: v if isinstance(v, str) else "")
def test_reached_is_read_by_the_high_above_the_close_and_the_low_below_it(name, rule, spikes, fires, instrument):
    res = _run(_defn(rule), _minutes(2, spikes=spikes), instrument)
    assert _fired(res) == ([1] if fires else []), name


# ------------------------------------------------------------------------------------- levels and window (D, HoE 4)
def test_the_levels_are_fixed_at_the_previous_close_not_the_decision_candles_own_close(instrument):
    """[D] c0 closes at 100, so X = 101. c1 reaches 101.2 at minute 3 and then runs on to close at 110. Using c1's own
    close (X = 111) would call this false."""
    m = _minutes(2, spikes={(1, 3): (101.2, 100.0)}, closes={1: 110.0})
    assert _fired(_run(_defn(), m, instrument)) == [1]


def test_the_levels_follow_each_candles_previous_close(instrument):
    """c1 closes at 100.8 (reaching neither 101 nor 99), so for c2 X = 101.8 and Y = 99.8: a minute at 101.5 is NOT X,
    one at 101.9 is. (v3: the earlier 105 close reached c1's own X, so c1 fired and held the position.)"""
    m = _minutes(3, spikes={(2, 3): (101.5, 100.0)}, closes={1: 100.8})
    assert _fired(_run(_defn(), m, instrument)) == []
    m2 = _minutes(3, spikes={(2, 3): (101.9, 100.0)}, closes={1: 100.8})
    assert _fired(_run(_defn(), m2, instrument)) == [2]


def test_a_y_reached_only_after_the_decision_time_changes_nothing(instrument):
    """[HoE 4] c1 reaches X at minute 5 and nothing else. c2 is the future: whatever it does (Y, X, both, flat), c1's
    decision and the entry's price and time are the same as with a flat c2."""
    ref = _run(_defn(), _minutes(3, spikes={(1, 5): (101.2, 100.0)}), instrument)
    assert _fired(ref)[:1] == [1]
    for future in ({(2, 1): (100.0, 90.0)}, {(2, 1): (110.0, 100.0)}, {(2, 1): (110.0, 90.0)}):
        res = _run(_defn(), _minutes(3, spikes={(1, 5): (101.2, 100.0), **future}), instrument)
        assert _fired(res)[:1] == [1] and _lineage(res)["y_minute"] is None
        assert res.fills.loc[res.fills.index[0], ["ts_last", "avg_px"]].tolist() == \
            ref.fills.loc[ref.fills.index[0], ["ts_last", "avg_px"]].tolist()


def test_an_x_reached_only_after_the_decision_time_is_not_read_either(instrument):
    """The other way round: c1 reaches nothing, c2 reaches X at its minute 1. c1 must not fire."""
    res = _run(_defn(), _minutes(3, spikes={(2, 1): (101.5, 100.0)}), instrument)
    assert 1 not in _fired(res)


def test_the_window_is_the_minutes_closing_after_the_previous_close_up_to_the_decision_close(instrument):
    """[D] Window. The minute closing exactly at c0's close belongs to c0: a Y spike there is not read for c1. The
    minute closing exactly at c1's close is read: an X spike there fires, and a same-minute close spike is Y-first."""
    before = _minutes(2, spikes={(0, 15): (100.0, 98.5), (1, 5): (101.2, 100.0)})
    assert _fired(_run(_defn(), before, instrument)) == [1]
    last_only = _minutes(2, spikes={(1, 15): (101.2, 100.0)})
    assert _fired(_run(_defn(), last_only, instrument)) == [1]
    last_both = _minutes(2, spikes={(1, 15): (101.5, 98.5)})
    assert _fired(_run(_defn(), last_both, instrument)) == []


# Advisor 6 Oct 22:11 ("R2 first-touch edges", supersedes the design's missing-minute rule): the candle's own high and
# low come first. Only one of X, Y in range -> decided without minutes. Neither -> false. BOTH in range -> the minutes
# decide, and a missing minute matters only if it comes BEFORE the first minute that reaches a level -> "unknown".
BOTH = {(1, 5): (101.5, 100.0), (1, 12): (100.0, 98.5)}  # X at minute 5, then Y at minute 12 (X first)


@pytest.mark.parametrize("name, spikes, drop, fires, unknown", [
    ("a gap after the first reach does not matter", BOTH, {(1, 9)}, True, 0),
    ("the missing minute is the LAST and a level was reached earlier", BOTH, {(1, 15)}, True, 0),
    ("a gap before the first reach -> unknown", BOTH, {(1, 2)}, False, 1),
    ("the missing minute is the first one, both levels in range -> unknown", BOTH, {(1, 1)}, False, 1),
    ("the minute that held the first reach is the one missing -> unknown", BOTH, {(1, 5)}, False, 1),
    ("the last minute holds both levels and is missing, nothing reached earlier -> unknown",
     {(1, 15): (101.5, 98.5)}, {(1, 15)}, False, 1),
    ("only X in range: decided without the minutes even with gaps -> true", {(1, 9): (101.5, 100.0)},
     {(1, 2), (1, 3), (1, 9)}, True, 0),
    ("only Y in range: false without the minutes, and not unknown", {(1, 9): (100.0, 98.5)}, {(1, 2), (1, 3)},
     False, 0),
    ("neither in range: false without the minutes, and not unknown", {}, {(1, 2), (1, 3)}, False, 0),
])
def test_a_missing_minute_matters_only_when_both_levels_are_in_the_range_and_it_comes_first(
        name, spikes, drop, fires, unknown, instrument):
    res = _run(_defn(), _minutes(3, spikes=spikes), instrument, drop=drop)  # c2 follows so c1 closes even if its last minute is missing
    ft = _ft(res)
    assert _fired(res) == ([1] if fires else []) and ft["unknown"] == unknown, name


def test_a_missing_minute_in_an_earlier_candle_does_not_spoil_the_next_one(instrument):
    m = _minutes(3, spikes={(2, 3): (101.5, 100.0)})
    res = _run(_defn(), m, instrument, drop={(1, 9)})
    assert _fired(res) == [2] and _ft(res)["unknown"] == 0


def test_minutes_that_show_no_reach_while_the_candle_range_does_are_inconsistent_and_false(instrument):
    """[ADV 22:11] The candle's range holds both levels (minute 6 reached 101.5 and 98.5) but the minutes given for
    that minute are flat: not a path anyone can trust. False on the entry side, counted as unknown (design, amended)."""
    m = _minutes(2, spikes={(1, 6): (101.5, 98.5)})
    flat_exec = _minutes(2)
    res = _run(_defn(), m, instrument, exec_prices=flat_exec)
    ft = _ft(res)
    assert _fired(res) == [] and ft["unknown"] == 1 and ft["true"] == 0


def test_an_unknown_path_in_the_paper_runtime_stays_unknown_and_false(instrument):
    """[ADV 22:11] 'unknown, late minute': in paper the minute that is not there when the candle closes is unknown at
    that decision; the decision is not revised when the minute turns up later. Driven here with the paper runtime and
    the minute never arriving; a hub-fed late arrival is PE1's to add once its entry point is named."""
    m = _minutes(2, spikes=BOTH)
    res = _run(_defn(), m, instrument, drop={(1, 2)}, risk_profile="balanced")
    assert _fired(res) == [] and _ft(res)["unknown"] == 1


# ---------------------------------------------------- unknown and same-minute: false on the entry side, TRUE on exits
SAME = {(1, 6): (101.5, 98.5)}
UNKNOWN = {"spikes": BOTH, "drop": {(1, 2)}}


@pytest.mark.parametrize("where", ["setup", "trigger", "confirm"])
@pytest.mark.parametrize("case", ["same minute", "unknown"])
def test_same_minute_and_unknown_are_false_in_every_entry_side_position(where, case, instrument):
    d = {**BASE, **POSITIONS[where]({"first_touch": {"reach": UP, "before": DOWN}})}
    spikes, drop = (SAME, ()) if case == "same minute" else (BOTH, {(1, 2)})
    res = _run(d, _minutes(3, spikes=spikes), instrument, drop=drop)
    assert 1 not in _fired(res) and _ft(res, f"long.entry.{where}")["judged"] >= 1


def _exit_defn(reenter=False):
    """reenter=True is the original data: entry `close > 0` is true on every candle, so the side re-enters on the candle
    the exit fires (#161 P1-5). The default entry is true on c0 only."""
    return {**BASE, "long": {"entry": {"left": "close", "op": ">", "right": 0} if reenter else
                             {"left": "close", "op": "<", "right": 99.9},
                             "exit": {"first_touch": {"reach": UP, "before": DOWN}}}}


def _exit_fills(res):
    return sorted(int((res.fills.loc[o, "ts_last"] - T0) / pd.Timedelta(minutes=15)) - 1
                  for o in res.fills.index if res.decisions[o]["intent"] != "entry")


@pytest.mark.parametrize("name, spikes, drop, exits", [
    ("same minute resolves TRUE in an exit", SAME, (), True),
    ("unknown resolves TRUE in an exit: missing data never blocks an exit", BOTH, {(1, 2)}, True),
    ("a clean X-first path exits", {(1, 3): (101.5, 100.0), (1, 9): (100.0, 98.5)}, (), True),
    ("a clean Y-first path does not exit", {(1, 3): (100.0, 98.5), (1, 9): (101.5, 100.0)}, (), False),
])
def test_in_an_exit_condition_same_minute_and_unknown_resolve_true(name, spikes, drop, exits, instrument):
    """The entry (a plain rule) opens the position at c0's close; the exit is judged on c1."""
    res = _run(_exit_defn(), _minutes(2, spikes=spikes, closes={0: 99.8}), instrument, drop=drop)
    assert (1 in _exit_fills(res)) is exits and _ft(res, "long.exit")["judged"] >= 1, name




@pytest.mark.parametrize("name, spikes, drop, exits", [
    pytest.param("same minute resolves TRUE in an exit", SAME, (), True),
    pytest.param("unknown resolves TRUE in an exit", BOTH, {(1, 2)}, True),
    pytest.param("a clean X-first path exits", {(1, 3): (101.5, 100.0), (1, 9): (100.0, 98.5)}, (), True),
    # a control, not a pin: nothing exits on a Y-first path, with or without the re-entry bug
    pytest.param("a clean Y-first path does not exit", {(1, 3): (100.0, 98.5), (1, 9): (101.5, 100.0)}, (), False),
])
def test_the_same_exit_cases_on_the_original_data_where_the_entry_is_true_on_every_candle(
        name, spikes, drop, exits, instrument):
    """[Advisor 22:30, #161 same-candle re-entry = MAJOR] The original data, kept: entry `close > 0` holds on every
    candle. An exit that fires at c1's close must show as an exit fill and the same side must NOT re-enter on c1."""
    res = _run(_exit_defn(reenter=True), _minutes(2, spikes=spikes), instrument, drop=drop)
    assert (1 in _exit_fills(res)) is exits and _ft(res, "long.exit")["judged"] >= 1, name
    assert 1 not in _fired(res)[1:] or not exits, "re-entered on the candle the exit fired"


# ------------------------------------------------------------ #161 same-candle re-entry (Advisor 22:30), plain rules
# An exit fired at candle k's CLOSE: no same-direction re-entry on k, earliest k+1 close, in backtest and in paper.
# Opposite direction on k only if the definition explicitly declares a reversal. After an INTRABAR exit (stop,
# market-on-touch target during k) re-entry at k's close is allowed, and the report counts it.
EXIT_UP = {"left": "close", "op": ">", "right": 100.5}  # true on c1 only (c1 closes at 101), so the exit fires at c1's close


def _always_long(**extra):
    return {**BASE, "long": {"entry": {"left": "close", "op": ">", "right": 0}, "exit": EXIT_UP}, **extra}


def _at_close(res, k):
    """(intent, side) of every fill whose time is candle k's close, in order."""
    t = T0 + pd.Timedelta(minutes=15 * (k + 1))
    rows = res.fills[res.fills["ts_last"] == t].sort_values("ts_init")
    return [(res.decisions[o]["intent"], str(rows.loc[o, "side"])) for o in rows.index]


@pytest.mark.parametrize("profile", [None, "balanced"], ids=["backtest", "paper"])
def test_an_exit_fired_at_a_candles_close_is_one_round_trip_not_a_close_and_reopen(profile, instrument):
    """(a) always-true entry + an exit that fires at c1's close. c0 opens; c1 closes it and does NOTHING else; the
    earliest re-entry is c2's close (the exit rule is false there). Not close + reopen at the same price on c1."""
    m = _minutes(4, closes={1: 101.0})
    res = _run(_always_long(), m, instrument, risk_profile=profile)
    assert [i for i, _ in _at_close(res, 0)] == ["entry"]
    assert [i for i, _ in _at_close(res, 1)] == ["exit"], "the exit candle must hold only the exit"
    assert [i for i, _ in _at_close(res, 2)] == ["entry"], "the earliest re-entry is the next candle's close"
    assert getattr(res, "reentries_on_exit_candle", None) == 0, "not built: res.reentries_on_exit_candle"


def test_an_explicitly_declared_reversal_may_open_the_opposite_way_on_the_exit_candle(instrument):
    """(b) The long exit and a short entry are both true at c1's close, and the definition declares the reversal
    (`{"short": {"entry": ..., "reverse": true}}`, named by PE1). Both fills are allowed on c1, exit first, and the
    report counts one re-entry on the exit candle (an entry filled on a candle where a non-entry order also filled)."""
    d = {**BASE, "long": {"entry": {"left": "close", "op": ">", "right": 0}, "exit": EXIT_UP},
         "short": {"entry": EXIT_UP, "reverse": True}}
    try:
        res = _run(d, _minutes(3, closes={1: 101.0}), instrument, perp=True)
    except AssertionError:
        raise
    assert [i for i, _ in _at_close(res, 1)] == ["exit", "entry"]
    assert _at_close(res, 1)[1][1] == "SELL"
    assert getattr(res, "reentries_on_exit_candle", None) == 1, "not built: res.reentries_on_exit_candle"


def test_the_same_definition_without_a_declared_reversal_does_not_open_the_opposite_way_on_the_exit_candle(instrument):
    """(1) Refused = no opposite fill on the exit candle and the count stays 0 (the ruling defines no refusal row)."""
    d = {**BASE, "long": {"entry": {"left": "close", "op": ">", "right": 0}, "exit": EXIT_UP},
         "short": {"entry": EXIT_UP}}
    res = _run(d, _minutes(3, closes={1: 101.0}), instrument, perp=True)
    assert [i for i, _ in _at_close(res, 1)] == ["exit"], "an undeclared reversal on the exit candle is refused"
    assert getattr(res, "reentries_on_exit_candle", None) == 0, "not built: res.reentries_on_exit_candle"


def test_an_undeclared_opposite_entry_still_fills_at_the_next_candle_if_its_condition_still_holds(instrument):
    """[HoQA 22:42] The refusal is for the exit candle only; a fix that suppresses the signal for good must fail. The
    long enters while close < 100.5 (c0 only), exits when close > 100.5, and the short enters on the same condition.
    c1 and c2 both close at 101: c1 holds only the exit (the short is refused there), c2 holds the short entry."""
    d = {**BASE, "long": {"entry": {"left": "close", "op": "<", "right": 100.5}, "exit": EXIT_UP},
         "short": {"entry": EXIT_UP}}
    res = _run(d, _minutes(4, closes={1: 101.0, 2: 101.0}), instrument, perp=True)
    assert [i for i, _ in _at_close(res, 0)] == ["entry"]
    assert [i for i, _ in _at_close(res, 1)] == ["exit"], "an undeclared reversal on the exit candle is refused"
    assert _at_close(res, 2) == [("entry", "SELL")], "the short fills at the next candle, its condition still true"
    assert getattr(res, "reentries_on_exit_candle", None) == 0, "not built: res.reentries_on_exit_candle"


def test_after_an_intrabar_exit_re_entry_at_the_same_candles_close_is_allowed_and_counted(instrument):
    """(c) A resting stop at the previous close minus 1 (99) is hit by minute 6 of c1 (low 98.5). The entry rule is
    true at c1's close, so the side re-enters there, and the report counts one 're-entry on the exit candle'."""
    d = {**BASE, "long": {"entry": {"left": "close", "op": ">", "right": 0}}, "exits": {"stop": {"level": DOWN}}}
    res = _run(d, _minutes(3, spikes={(1, 6): (100.0, 98.5)}), instrument)
    stopped = res.fills[(res.fills["ts_last"] > T0 + pd.Timedelta(minutes=15)) &
                        (res.fills["ts_last"] < T0 + pd.Timedelta(minutes=30))]
    assert len(stopped) >= 1 and all(res.decisions[o]["intent"] != "entry" for o in stopped.index), \
        "the stop (a volume-capped stop may fill in two parts) must fire inside c1"
    assert [i for i, _ in _at_close(res, 1)] == ["entry"], "re-entry at the exit candle's close is allowed"
    count = getattr(res, "reentries_on_exit_candle", None)
    assert count == 1, "not built: res.reentries_on_exit_candle (assumed name)"



# ----------------------------------------------------------------------------------------- report (D, HoE 2) + reference
def test_hand_computed_reference_over_a_fixed_minute_path_and_the_report(instrument):
    """[HoE 5] Five judged candles, hand computed (X = 101, Y = 99 every time as every close is 100):
       c1  minute 6 reaches both                         -> false, same minute
       c2  minute 4 reaches Y (98.7), minute 9 reaches X -> false (Y first)
       c3  nothing reaches either                        -> false, not reached
       c4  X at 3 and Y at 12 in range, minute 1 missing -> false, unknown (a gap before the first reach)
       c5  minute 2 reaches X (101.1), minute 12 Y       -> TRUE, the only entry
    Report: judged 5, true 1, same_minute 1, unknown 1. Candles where either level was reached: c1, c2, c4, c5 = 4, so
    same-minute + unknown is 2 of 4 = 50% (above the 5% G1 line). Lineage on the entry: x 101, y 99, x_minute = c5
    minute 2, y_minute = c5 minute 12."""
    m, drop = _ref_path()
    res = _run(_defn(), m, instrument, drop=drop)
    ft = _ft(res)
    assert _fired(res) == [5]
    assert (ft["judged"], ft["true"], ft["same_minute"], ft["unknown"]) == (5, 1, 1, 1)
    assert (ft["same_minute"] + ft["unknown"]) / 4 == pytest.approx(0.5)
    lin = _lineage(res)
    assert (lin["x"], lin["y"], lin["same_minute"]) == (101.0, 99.0, False)
    assert (lin["x_minute"], lin["y_minute"]) == (_at(5, 2), _at(5, 12))


def _ref_path():
    return (_minutes(6, spikes={(1, 6): (101.5, 98.5), (2, 4): (100.0, 98.7), (2, 9): (101.3, 100.0),
                                (3, 7): (100.5, 99.5), (4, 3): (101.3, 100.0), (4, 12): (100.0, 98.6),
                                (5, 2): (101.1, 100.0), (5, 12): (100.0, 98.9)}), {(4, 1)})


@pytest.mark.parametrize("y_only, same, unknown, rerun", [
    (19, 1, 0, False),  # 1 flagged of 20 reached = 5%: not above
    (18, 2, 0, True),   # 10%
    (18, 1, 1, True),   # same-minute + unknown = 10%
    (20, 0, 0, False),
])
def test_the_g1_rerun_trigger_is_same_minute_plus_unknown_above_five_percent_of_candles_where_a_level_was_reached(
        y_only, same, unknown, rerun, instrument):
    """[ADV 22:11 (2)] 20 candles reach a level (Y-only candles are false and keep the strategy flat), `same` of them
    are same-minute and `unknown` are unknown. The report's `ambiguous_share` is (same-minute + unknown) / candles where either level was reached."""
    n = y_only + same + unknown
    spikes, drop = {}, set()
    for k in range(1, n + 1):
        if k <= y_only:
            spikes[(k, 5)] = (100.0, 98.5)
        elif k <= y_only + same:
            spikes[(k, 5)] = (101.5, 98.5)
        else:
            spikes[(k, 5)], spikes[(k, 12)] = (101.5, 100.0), (100.0, 98.5)
            drop.add((k, 1))
    ft = _ft(_run(_defn(), _minutes(n + 1, spikes=spikes), instrument, drop=drop))
    assert (ft["same_minute"], ft["unknown"]) == (same, unknown)
    assert ft["ambiguous_share"] == pytest.approx((same + unknown) / n)  # of the candles where a level was reached
    assert (ft["ambiguous_share"] > 0.05) is rerun  # above 5% G1 re-runs with first_touch_flip=True


def test_both_flags_are_reported_for_every_judged_candle_even_when_another_condition_decides_the_group(instrument):
    """[ADV 22:11 (2)] `all` with a plain condition that is never true: the group is false whatever first_touch says,
    and first_touch's same-minute (c1) and unknown (c2) are still counted."""
    rule = {"all": [{"left": "close", "op": "<", "right": 0}, {"first_touch": {"reach": UP, "before": DOWN}}]}
    m = _minutes(3, spikes={(1, 6): (101.5, 98.5), (2, 5): (101.5, 100.0), (2, 12): (100.0, 98.5)})
    res = _run(_defn(rule), m, instrument, drop={(2, 2)})
    ft = _ft(res, "long.entry[1]")
    assert _fired(res) == [] and (ft["judged"], ft["same_minute"], ft["unknown"]) == (2, 1, 1)


def test_the_same_minute_flag_and_both_minutes_are_in_the_lineage_when_another_condition_carries_the_entry(instrument):
    """The rule is false on a same-minute candle, so it cannot place the order alone; inside an `any` with a plain
    condition the order is placed, and the payload still names the rule and flags the same minute."""
    rule = {"any": [{"first_touch": {"reach": UP, "before": DOWN}}, {"left": "close", "op": ">", "right": 99.7}]}
    res = _run(_defn(rule), _minutes(2, spikes={(1, 6): (101.5, 98.5)}, closes={0: 99.5}), instrument)
    lin = _lineage(res, "long.entry[0]")
    assert lin["same_minute"] is True and lin["x_minute"] == lin["y_minute"] == _at(1, 6)


# --------------------------------------------------------------------------------- checked, hashed, tagged (D, HoE 3)
POSITIONS = {
    "entry": lambda r: {"long": {"entry": r}},
    "exit": lambda r: {"long": {"entry": {"left": "close", "op": ">", "right": 0}, "exit": r}},
    "setup": lambda r: {"long": {"entry": {"setup": r, "trigger": {"left": "close", "op": ">", "right": 0},
                                           "expire_after": {"bars": 4, "timeframe": "15m"}}}},
    "trigger": lambda r: {"long": {"entry": {"setup": {"left": "close", "op": ">", "right": 0}, "trigger": r,
                                             "expire_after": {"bars": 4, "timeframe": "15m"}}}},
    "confirm": lambda r: {"long": {"entry": {"event": {"left": "close", "op": ">", "right": 0}, "confirm": r,
                                             "within": 3}}},
}


@pytest.mark.parametrize("where", sorted(POSITIONS))
def test_first_touch_is_accepted_wherever_a_condition_is(where):
    _checked({**BASE, **POSITIONS[where]({"first_touch": {"reach": UP, "before": DOWN}})})
    # the same place with a mistaken operand must be refused: the pair proves the rule is parsed, not skipped.
    with pytest.raises(ValueError, match="nope"):
        check_definition({**BASE, **POSITIONS[where]({"first_touch": {"reach": "nope", "before": DOWN}})}, bar_spec=SPEC)


@pytest.mark.parametrize("rule, words", [
    ({"first_touch": {"reach": UP}}, "before"),
    ({"first_touch": {"before": DOWN}}, "reach"),
    ({"first_touch": {"reach": UP, "before": DOWN, "extra": 1}}, "extra"),
    ({"first_touch": {"reach": "nope", "before": DOWN}}, "nope"),
    ({"first_touch": {"reach": UP, "before": "nope"}}, "nope"),
    ({"first_touch": {"reach": {"add": ["close"]}, "before": DOWN}}, "two operands"),
    ({"first_touch": {"reach": float("nan"), "before": DOWN}}, "finite"),
    ({"first_touch": {"reach": True, "before": DOWN}}, "not a number"),
    ({"first_touch": [UP, DOWN]}, "reach.*before"),
    ({"not": {"first_touch": {"reach": UP, "before": DOWN}}}, "negat"),  # Advisor 22:11: no 'not first_touch'
    ({"first_touch": {"reach": UP, "before": UP}}, "same level|equal"),  # X == Y
    ({"first_touch": {"reach": 101, "before": 101.0}}, "same level|equal"),
])
def test_a_malformed_first_touch_is_refused_with_its_reason(rule, words):
    with pytest.raises(ValueError, match=words):
        check_definition(_defn(rule), bar_spec=SPEC)


def test_first_touch_counts_as_reading_the_previous_candle_for_warm_up():
    """[D] +1, as a cross has (the plain comparison needs 0, a cross 1)."""
    plain = _checked(_defn({"left": "close", "op": ">", "right": 0}))
    ft = _checked(_defn())
    assert plain.warmup[None] == 0 and ft.warmup[None] == 1


def test_first_touch_is_part_of_the_definition_hash():
    _checked(_defn())  # the hash of a definition the checker refuses proves nothing
    base = definition_hash(_defn())
    same = definition_hash(_defn({"first_touch": {"reach": {"add": ["close", 1.0]}, "before": DOWN}}))
    assert same == base  # number spelling, as every other rule
    assert definition_hash(_defn({"first_touch": {"reach": {"add": ["close", 2]}, "before": DOWN}})) != base
    assert definition_hash(_defn({"first_touch": {"reach": DOWN, "before": UP}})) != base  # swapped
    assert definition_hash(_defn({"left": "close", "op": ">", "right": 0})) != base


def test_the_hash_and_the_lineage_agree(instrument):
    d = _defn()
    res = _run(d, _minutes(2, spikes={(1, 3): (101.5, 100.0)}), instrument)
    o = res.fills.index[0]
    assert res.decisions[o]["signal"]["definition"] == definition_hash(d)


# ----------------------------------------------------------------------------- minutes in each mode (D "Same minutes")
def test_without_1m_bars_a_run_stops_with_the_reason_only_when_a_candle_actually_needed_the_minutes(instrument):
    """[design, amended] The run starts. A candle with at most one level in range is decided by the range alone; the
    first candle with both levels in range needs minutes it does not have, and the run stops saying so."""
    easy = _minutes(2, spikes={(1, 3): (101.5, 100.0)})  # only X in range
    try:
        res = run_backtest("rules", _decisions(easy), instrument, params=to_params(_defn()), bar_minutes=15,
                           half_spread=0)
    except ValueError as err:
        raise AssertionError(f"not built: {err}") from err
    assert _fired(res) == [1]
    hard = _minutes(2, spikes=BOTH)
    with pytest.raises(ValueError, match="1-minute") as e:
        run_backtest("rules", _decisions(hard), instrument, params=to_params(_defn()), bar_minutes=15, half_spread=0)
    assert "not a rule" not in str(e.value), "not built"


def test_the_flip_option_resolves_ambiguous_candles_the_other_way(instrument):
    """[design, amended] G1 re-runs above 5% with first_touch_flip=True and judges the worse result: same-minute and
    unknown become TRUE on the entry side and FALSE in an exit rule."""
    for spikes, drop in ((SAME, ()), (BOTH, {(1, 2)})):
        m = _minutes(2, spikes=spikes)
        assert _fired(_run(_defn(), m, instrument, drop=drop)) == []
        assert _fired(_run(_defn(), m, instrument, drop=drop, first_touch_flip=True)) == [1]
        e = _minutes(2, spikes=spikes, closes={0: 99.8})
        assert 1 in _exit_fills(_run(_exit_defn(), e, instrument, drop=drop))
        assert 1 not in _exit_fills(_run(_exit_defn(), e, instrument, drop=drop, first_touch_flip=True))


def test_a_one_minute_decision_candle_carries_a_note_that_the_share_rests_on_the_assumption(instrument):
    m = _minutes(4, tf=1, spikes={(2, 1): (101.5, 98.5)})
    note = _ft(_run(_defn(), m, instrument, tf=1)).get("note")
    assert note and "assum" in note


def test_a_definition_using_first_touch_with_1m_bars_that_do_not_cover_the_run_is_refused_or_counted_unknown(instrument):
    """Minutes for c1 only: c2 has none at all. Either the run refuses at the start (naming the missing minutes) or c2
    is unknown. It must never be judged as 'neither reached' or silently true."""
    m = _minutes(3, spikes={(1, 3): (101.5, 100.0), (2, 3): (101.5, 100.0), (2, 9): (100.0, 98.5)})
    short = m.iloc[:30]
    try:
        res = run_backtest("rules", _decisions(m), instrument, params=to_params(_defn()), bar_minutes=15,
                           exec_prices=short, exec_minutes=1, half_spread=0)
    except ValueError as e:
        assert "not a rule" not in str(e), f"not built: {e}"
        assert "minute" in str(e)
        return
    assert _ft(res)["unknown"] >= 1 and 2 not in _fired(res)


def test_the_paper_runtime_judges_the_same_candles_the_same_way(instrument):
    """[D] backtest == paper on the same minutes. The paper runtime (a throwaway journal) is fed the same minutes and
    must reach the same entries, prices and report. A hub-fed replay is the PE1's to add once its entry point is named."""
    m, drop = _ref_path()
    a = _run(_defn(), m, instrument, drop=drop)
    b = _run(_defn(), m, instrument, drop=drop, risk_profile="balanced")
    assert _fired(a) == _fired(b) == [5]
    assert _ft(a) == _ft(b)
    assert a.fills["ts_last"].tolist() == b.fills["ts_last"].tolist()


D13 = pytest.mark.xfail(strict=True, raises=AssertionError, reason="D13")


@D13
def test_fill_prices_match_between_research_and_the_paper_runtime_to_the_rounding_cent(instrument):
    """[HoQA 22:30, one-taker-rule ruling 20:55] Same minutes, same decisions, same booked price. Tolerance is the #161
    rounding cent: abs 0.02 divided by the smallest filled quantity. Today research books ~100.0049 and the paper runtime
    100.00 (D13 gap, also pinned in d13-xfails). Never widen the tolerance to hide it."""
    m, drop = _ref_path()
    a = _run(_defn(), m, instrument, drop=drop)
    b = _run(_defn(), m, instrument, drop=drop, risk_profile="balanced")
    assert _fired(a) == _fired(b) == [5], "not built: first_touch"  # the entry must exist before its price can be compared
    q = min(float(x) for x in [*a.fills["filled_qty"], *b.fills["filled_qty"]])
    for pa, pb in zip(a.fills["avg_px"], b.fills["avg_px"]):
        assert abs(float(pa) - float(pb)) <= 0.02 / q
