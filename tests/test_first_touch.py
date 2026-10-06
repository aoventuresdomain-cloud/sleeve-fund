"""R2: the rule builder's first_touch rule, "the price reaches X before it reaches Y" inside the decision candle's own
1-minute bars (design v2/r2-first-touch-design.md; HoE Done-when 22:01; Advisor 14:05: within one minute the adverse
level is taken as first)."""

import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.definitions import Touch, check_definition, definition_hash, first_touch, to_params

M = 60_000_000_000
START = pd.Timestamp("2025-10-03", tz="UTC")
FT = {"first_touch": {"reach": {"add": ["close", 1]}, "before": {"sub": ["close", 1]}}}  # the close before +/- 1
DEFN = {"version": 1, "reason": "A test case, written before it is run.", "blocks": {},
        "long": {"entry": FT, "exit": {"left": "close", "op": "<", "right": 0}}}


# --- the path, by hand ---------------------------------------------------------------------------------------------

def _m(*bars):
    """Minutes (close ns, open, high, low, close) from (high, low) pairs, one minute apart."""
    return [(i * M, (h + lo) / 2, h, lo, (h + lo) / 2) for i, (h, lo) in enumerate(bars, 1)]


@pytest.mark.parametrize("minutes, expect", [
    (_m((100.5, 99.5), (101.0, 100.0), (100.2, 98.0)), Touch(True, 2 * M, 3 * M, "minutes", reached=True)),  # both minutes, for the lineage
    (_m((100.5, 99.5), (100.2, 99.0), (101.5, 100.0)), Touch(False, 3 * M, 2 * M, "minutes", reached=True)),
    (_m((100.5, 99.5), (101.2, 98.9)), Touch(False, 2 * M, 2 * M, "minutes", same_minute=True, reached=True)),
    (_m((100.9, 99.1), (100.5, 99.5)), Touch(False, by="minutes")),  # neither: 101 and 99 are never reached
])
def test_first_touch_follows_the_minutes_in_order_and_takes_the_adverse_level_first_within_one(minutes, expect):
    """Levels X 101 and Y 99 against the candle before's close of 100: X is reached by a high, Y by a low."""
    assert first_touch(minutes, 101.0, 99.0, 100.0) == expect


def test_the_candles_range_settles_it_when_one_level_is_outside_and_minutes_must_agree_with_it():
    assert first_touch((), 101.0, 99.0, 100.0, high=101.5, low=99.5) == Touch(True, reached=True)  # only X in range
    assert first_touch((), 101.0, 99.0, 100.0, high=100.5, low=99.5) == Touch(False)  # neither
    both = dict(high=101.5, low=98.5, start=0, step=2)
    assert first_touch(_m((100.5, 99.5)), 101.0, 99.0, 100.0, **both).unknown == "missing"  # minute 2 not to hand
    assert first_touch(_m((100.5, 99.5), (100.5, 99.5)), 101.0, 99.0, 100.0, **both).unknown == "inconsistent"
    assert first_touch(_m((101.0, 99.5)), 101.0, 99.0, 100.0, **both).holds  # X in minute 1: a later gap can't matter


def test_a_level_below_the_close_before_is_reached_by_a_low_and_one_above_by_a_high():
    """A short's question: down to 99 before up to 101."""
    assert first_touch(_m((100.4, 99.0)), 99.0, 101.0, 100.0).holds
    assert not first_touch(_m((101.0, 99.6)), 99.0, 101.0, 100.0).holds


# --- the definition ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rule, words", [
    ({"first_touch": {"reach": 101}}, "first_touch is {reach, before}"),
    ({"first_touch": {"reach": 101, "before": "nope"}}, "unknown reference 'nope'"),
    ({"first_touch": [101, 99]}, "first_touch is {reach, before}"),
    ({"first_touch": {"reach": {"add": ["close", 1]}, "before": {"add": ["close", 1.0]}}}, "the same level"),
])
def test_a_malformed_first_touch_is_refused_with_its_reason(rule, words):
    with pytest.raises(ValueError, match=words.replace("{", r"\{").replace("}", r"\}")):
        check_definition({**DEFN, "long": {"entry": rule}}, bar_minutes=15)


def test_first_touch_is_hashed_and_reads_the_candle_before():
    assert check_definition(DEFN, bar_minutes=15).warmup[None] == 1  # its levels are the candle before's
    other = {**DEFN, "long": {"entry": {"first_touch": {"reach": {"add": ["close", 2]},
                                                        "before": {"sub": ["close", 1]}}}}}
    assert definition_hash(other) != definition_hash(DEFN)


# --- a backtest on 15-minute candles -------------------------------------------------------------------------------

def _frames(candle_b, after=(), drop=()):
    """Candle A (minutes 1-15) flat at 100; candle B (16-30) from (high, low) pairs, its close the midpoint of the
    last; then flat at that close, the first minutes of candle C from `after`. Returns (15-minute, 1-minute) bars,
    each stamped at its close; `drop` leaves those minutes (1-based) out of the 1-minute bars."""
    pairs = [(100.0, 100.0)] * 15 + list(candle_b) + list(after)
    end = (candle_b[-1][0] + candle_b[-1][1]) / 2
    pairs += [(end, end)] * (60 - len(pairs))
    rows, prev = [], 100.0
    for h, lo in pairs:
        c = (h + lo) / 2
        rows.append({"open": prev, "high": max(h, prev), "low": min(lo, prev), "close": c, "volume": 10.0})
        prev = c
    idx = pd.date_range(START + pd.Timedelta(minutes=1), periods=len(rows), freq="1min")
    minutes = pd.DataFrame(rows, index=idx)
    candles = minutes.resample("15min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return candles, minutes.drop(minutes.index[[i - 1 for i in drop]])


def _run(instrument, candle_b, **kw):
    candles, minutes = _frames(candle_b, **{k: v for k, v in kw.items() if k in ("after", "drop")})
    return run_backtest("rules", candles, instrument, params=to_params(DEFN), bar_minutes=15, half_spread=0,
                        exec_prices=minutes, exec_minutes=1)


def _entered(res) -> bool:
    return any(d["intent"] == "entry" for d in res.decisions.values())


FLAT = [(100.0, 100.0)]


def test_reached_x_first_inside_the_candle_enters_on_its_close_and_says_when(instrument):
    """Candle B: up to 101.2 in its 3rd minute, down to 98.8 in its 10th. The levels are 101 and 99 from candle A's
    close (100); candle B's own close (99.0) would put them at 100 and 98, which this path also reaches in that order,
    but the hand reference is the candle before's."""
    b = FLAT * 2 + [(101.2, 100.0)] + FLAT * 6 + [(100.0, 98.8)] + [(99.0, 99.0)] * 5
    res = _run(instrument, b)
    entry = next(d for d in res.decisions.values() if d["intent"] == "entry")
    ft = entry["signal"]["first_touch"]["long.entry"]
    assert ft == {"x": 101.0, "y": 99.0, "x_minute": "2025-10-03T00:18:00+00:00",
                  "y_minute": "2025-10-03T00:25:00+00:00", "same_minute": False, "held": True, "by": "minutes",
                  "unknown": None}
    stats = res.first_touch["long.entry"]
    assert stats["true"] == 1 and stats["same_minute"] == 0


def test_reached_y_first_doesnt_enter(instrument):
    b = FLAT * 2 + [(100.0, 98.9)] + FLAT * 6 + [(101.5, 100.0)] + [(100.0, 100.0)] * 5
    res = _run(instrument, b)
    assert not _entered(res)


def test_both_in_one_minute_takes_the_adverse_level_first_and_the_report_counts_it(instrument):
    b = FLAT * 4 + [(101.5, 98.5)] + FLAT * 10
    res = _run(instrument, b)
    assert not _entered(res)
    (stats,) = res.first_touch.values()
    assert stats["same_minute"] == 1 and stats["reached"] == 1


def test_a_level_reached_only_after_the_decision_time_never_counts(instrument):
    """Look-ahead: inside candle B neither level is reached; the first minute after its close reaches X (101.3).
    Judged at B's close, nothing held. Candle C then holds on its own range (only X inside it)."""
    b = FLAT * 15
    res = _run(instrument, b, after=[(101.3, 100.0)])
    stats = next(iter(res.first_touch.values()))
    entries = [d for d in res.decisions.values() if d["intent"] == "entry"]
    assert stats["true"] == 1 and len(entries) == 1
    assert entries[0]["signal"]["first_touch"][next(iter(res.first_touch))]["by"] == "range"
    coid = next(o for o, d in res.decisions.items() if d["intent"] == "entry")
    assert res.fills.loc[coid, "ts_last"] == START + pd.Timedelta(minutes=45)  # decided at C's close, not B's


def test_one_level_inside_the_candle_settles_it_without_minutes(instrument):
    """Advisor ~22:07: only X inside candle B's range: true from its high and low alone, even with its minutes
    missing; only Y inside: false."""
    res = _run(instrument, FLAT * 2 + [(101.2, 100.0)] + FLAT * 12, drop=[22])
    assert _entered(res) and next(iter(res.first_touch.values()))["by_minutes"] == 0
    assert not _entered(_run(instrument, FLAT * 2 + [(100.0, 98.8)] + FLAT * 12))


def test_with_both_inside_a_minute_missing_before_the_first_reach_leaves_it_unknown(instrument):
    """X at B's 5th minute, Y at its 10th. Its 3rd minute missing: the order can't be told, so no entry (the entry
    side's safe answer), counted. Its 7th missing instead: X already came first, so it holds."""
    b = FLAT * 4 + [(101.2, 100.0)] + FLAT * 4 + [(100.0, 98.8)] + FLAT * 5
    res = _run(instrument, b, drop=[18])
    assert not _entered(res) and next(iter(res.first_touch.values()))["missing"] == 1
    assert _entered(_run(instrument, b, drop=[22]))


def test_an_ambiguous_candle_never_blocks_an_exit(instrument):
    """In an exit rule a same-minute candle resolves to true (Advisor ~22:07: entry-skip, exit-never). Long from
    candle A (close 100 <= 100.5); candle B reaches both levels in its 5th minute and closes at 100.8."""
    exit_ft = {**DEFN, "long": {"entry": {"left": "close", "op": "<=", "right": 100.5}, "exit": FT}}
    candles, minutes = _frames(FLAT * 4 + [(101.5, 98.5)] + [(100.8, 100.8)] * 10)
    res = run_backtest("rules", candles, instrument, params=to_params(exit_ft), bar_minutes=15, half_spread=0,
                       exec_prices=minutes, exec_minutes=1)
    exits = [d for d in res.decisions.values() if d["intent"] == "exit"]
    assert exits and next(iter(res.first_touch.values()))["resolved"] == "true when ambiguous"


def test_the_g1_check_can_resolve_ambiguous_candles_the_other_way(instrument):
    candles, minutes = _frames(FLAT * 4 + [(101.5, 98.5)] + FLAT * 10)
    res = run_backtest("rules", candles, instrument, params=to_params(DEFN), bar_minutes=15, half_spread=0,
                       exec_prices=minutes, exec_minutes=1, first_touch_flip=True)
    (stats,) = res.first_touch.values()
    assert _entered(res) and stats["same_minute"] == 1 and stats["ambiguous_share"] == 1.0


def test_a_slower_candle_backtest_without_minutes_runs_until_a_candle_needs_them(instrument):
    candles, _ = _frames(FLAT * 2 + [(101.2, 100.0)] + FLAT * 12)  # only X inside: no minutes needed
    run_backtest("rules", candles, instrument, params=to_params(DEFN), bar_minutes=15, half_spread=0)
    candles, _ = _frames(FLAT * 4 + [(101.5, 98.5)] + FLAT * 10)
    with pytest.raises(ValueError, match="needed the 1-minute bars of 1 candles"):
        run_backtest("rules", candles, instrument, params=to_params(DEFN), bar_minutes=15, half_spread=0)


def test_on_one_minute_candles_the_report_says_the_share_is_the_assumptions(instrument):
    _, minutes = _frames(FLAT * 4 + [(101.5, 98.5)] + FLAT * 10)
    res = run_backtest("rules", minutes, instrument, params=to_params(DEFN), bar_minutes=1, half_spread=0)
    (stats,) = res.first_touch.values()
    assert stats["same_minute"] == 1 and "assumption" in stats["note"]


def test_paper_judges_the_same_minutes_a_backtest_does():
    """Paper's first_touch reads the minutes the hub client built each bar from (HubStatus.minutes); a backtest reads
    its own 1-minute bars (runner.minutes_from). Fed the same minutes, both give the same list for every candle, and
    a minute the hub refilled late into a bar still being built is in it, as it is in the store."""
    from sleeve_fund.paper.hub_client import Decoder, HubStatus
    from sleeve_fund.research.runner import minutes_from

    iid = "BTCUSDT-PERP.BINANCE"
    _, minutes = _frames(FLAT * 2 + [(101.2, 100.0)] + FLAT * 6 + [(100.0, 98.8)] + [(99.0, 99.0)] * 5)
    status = HubStatus()
    d = Decoder("15-MINUTE-LAST-EXTERNAL", status=status)
    rows = list(minutes.itertuples())
    order = list(range(len(rows)))
    order[20], order[21] = order[21], order[20]  # minute 22 lands before minute 21, which the hub refilled
    for i in order:
        r = rows[i]
        ts = r.Index.as_unit("ns").value
        d({"t": "bar", "id": iid, "o": f"{r.open:.2f}", "h": f"{r.high:.2f}", "l": f"{r.low:.2f}",
           "c": f"{r.close:.2f}", "v": f"{r.volume:.3f}", "ts": ts, "recv": ts, "refilled": i == 20}, ts + 5)
    backtest = minutes_from(minutes)
    for k in range(1, 4):
        start, end = (START + pd.Timedelta(minutes=15 * k)).value, (START + pd.Timedelta(minutes=15 * (k + 1))).value
        assert status.minutes_of(iid, start, end) == backtest(start, end) and len(backtest(start, end)) == 15


@pytest.mark.parametrize("cut", ["start", "end"])
def test_minutes_that_miss_part_of_the_first_or_last_decision_candle_are_refused(cut, instrument):
    """The engine builds its decision candles from these minutes: a candle they cover only in part (the first one's
    early minutes, or the last one's late minutes) would be judged on a hole, or never at all (CR on #170)."""
    candles, minutes = _frames(FLAT * 15)
    minutes = minutes.iloc[14:] if cut == "start" else minutes.iloc[:-1]  # from the first candle's close / short
    with pytest.raises(ValueError, match="1-minute bars of every decision candle"):
        run_backtest("rules", candles, instrument, params=to_params(DEFN), bar_minutes=15, half_spread=0,
                     exec_prices=minutes, exec_minutes=1)


def test_levels_equal_by_arithmetic_order_are_refused_as_the_same_level():
    """QA R2 round 1 F1: close + 1 and 1 + close are one level; neither can come first."""
    rule = {"first_touch": {"reach": {"add": ["close", 1]}, "before": {"add": [1, "close"]}}}
    with pytest.raises(ValueError, match="same level"):
        check_definition({**DEFN, "long": {"entry": rule}}, bar_spec="15-MINUTE-LAST-INTERNAL")
    check_definition({**DEFN, "long": {"entry": {"first_touch": {"reach": {"sub": ["close", 1]},
                                                                  "before": {"sub": [1, "close"]}}}}},
                     bar_spec="15-MINUTE-LAST-INTERNAL")  # subtraction doesn't commute: two levels
