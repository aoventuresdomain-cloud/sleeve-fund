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
    (_m((100.5, 99.5), (101.0, 100.0), (100.2, 98.0)), Touch(True, 2 * M, None)),  # X at minute 2, Y later
    (_m((100.5, 99.5), (100.2, 99.0), (101.5, 100.0)), Touch(False, None, 2 * M)),  # Y at minute 2 first
    (_m((100.5, 99.5), (101.2, 98.9)), Touch(False, 2 * M, 2 * M, same_minute=True)),  # both in minute 2
    (_m((100.9, 99.1), (100.5, 99.5)), Touch(False)),  # neither: 101 and 99 are never reached
])
def test_first_touch_follows_the_minutes_in_order_and_takes_the_adverse_level_first_within_one(minutes, expect):
    """Levels X 101 and Y 99 against the candle before's close of 100: X is reached by a high, Y by a low."""
    assert first_touch(minutes, 101.0, 99.0, 100.0) == expect


def test_a_level_below_the_close_before_is_reached_by_a_low_and_one_above_by_a_high():
    """A short's question: down to 99 before up to 101."""
    assert first_touch(_m((100.4, 99.0)), 99.0, 101.0, 100.0).holds
    assert not first_touch(_m((101.0, 99.6)), 99.0, 101.0, 100.0).holds


# --- the definition ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rule, words", [
    ({"first_touch": {"reach": 101}}, "first_touch is {reach, before}"),
    ({"first_touch": {"reach": 101, "before": "nope"}}, "unknown reference 'nope'"),
    ({"first_touch": [101, 99]}, "first_touch is {reach, before}"),
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
    ft = entry["signal"]["first_touch"]["first_touch(reach add(close,1.0), before sub(close,1.0))"]
    assert ft == {"reach_at": "2025-10-03T00:18:00+00:00", "before_at": None, "same_minute": False, "unknown": False}
    stats = res.first_touch["first_touch(reach add(close,1.0), before sub(close,1.0))"]
    assert stats["held"] == 1 and stats["same_minute"] == 0


def test_reached_y_first_doesnt_enter(instrument):
    b = FLAT * 2 + [(100.0, 98.9)] + FLAT * 6 + [(101.5, 100.0)] + [(100.0, 100.0)] * 5
    res = _run(instrument, b)
    assert not _entered(res)


def test_both_in_one_minute_takes_the_adverse_level_first_and_the_report_counts_it(instrument):
    b = FLAT * 4 + [(101.5, 98.5)] + FLAT * 10
    res = _run(instrument, b)
    assert not _entered(res)
    (stats,) = res.first_touch.values()
    assert stats["same_minute"] == 1 and stats["either_reached"] == 1


def test_a_level_reached_only_after_the_decision_time_never_counts(instrument):
    """Look-ahead: inside candle B neither level is reached; the first minute after its close reaches X (101.3).
    Judged at B's close, nothing held. Candle C then judges its own minutes from B's close."""
    b = FLAT * 15
    res = _run(instrument, b, after=[(101.3, 100.0)])
    stats = next(iter(res.first_touch.values()))
    entries = [d for d in res.decisions.values() if d["intent"] == "entry"]
    # held once, on candle C (to 00:45), whose first minute reached 101.3: never on candle B (to 00:30)
    assert stats["held"] == 1 and len(entries) == 1
    coid = next(o for o, d in res.decisions.items() if d["intent"] == "entry")
    assert res.fills.loc[coid, "ts_last"] == START + pd.Timedelta(minutes=45)  # decided at C's close, not B's
    assert entries[0]["signal"]["first_touch"][next(iter(res.first_touch))]["reach_at"] == "2025-10-03T00:31:00+00:00"


def test_a_candle_with_a_minute_missing_is_not_judged(instrument):
    b = FLAT * 2 + [(101.2, 100.0)] + FLAT * 12
    res = _run(instrument, b, drop=[22])
    assert not _entered(res)
    assert next(iter(res.first_touch.values()))["unknown"] >= 1


def test_a_first_touch_backtest_on_slower_candles_needs_the_minutes(instrument):
    candles, _ = _frames(FLAT * 15)
    with pytest.raises(ValueError, match="first_touch rule, judged on the 1-minute bars"):
        run_backtest("rules", candles, instrument, params=to_params(DEFN), bar_minutes=15, half_spread=0)


def test_the_levels_are_the_candle_befores_not_the_judged_candles_own(instrument):
    """Candle B reaches 101.2 in its 3rd minute and ends at 102.5. From candle A's close (100) the levels are 101 and
    99: X first, so it enters. From B's own close they would be 103.5 (never reached) and 101.5 (reached by the first
    minute's low of 100): no entry. Reading B's close to judge B's own path would be look-ahead."""
    b = FLAT * 2 + [(101.2, 100.0)] + [(102.5, 102.5)] * 12
    assert _entered(_run(instrument, b))


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
