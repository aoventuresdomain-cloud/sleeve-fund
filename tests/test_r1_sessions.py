"""R1 (Researcher tool gap; HoE 15 Oct, scope: the Asia and London session levels only, on PE3's completed-period
pattern): a rules block giving the last completed Asia or London session's high and low. Each session is in its own
market's local time (Advisor 15 Oct 08:44 UK): Asia is Asia/Tokyo [09:00, 17:00), London is Europe/London
[08:00, 16:00) (R1-DEF), so London's clock change moves London in UTC and leaves Asia where it is. The window and its zone are
named settings. Candles are stamped at their close and belong to the session their close falls in (the one closing
at the end is its last), as PriorPeriod's periods, and the levels move on the next close. Levels come only from
completed sessions seen from their start: the session in progress is never read, and the one the data starts in
part-way is never published. A session missing more than 10% of its candles is incomplete: it keeps the previous
levels and says so in `skipped`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.strategies.definitions import check_definition
from sleeve_fund.strategies.indicators import BLOCKS

WINDOWS = {"asia": ("Asia/Tokyo", "09:00", "17:00"), "london": ("Europe/London", "08:00", "16:00")}


def _candles(start: str, n: int, minutes: int = 15, seed: int = 7) -> pd.DataFrame:
    """Candles stamped at their close, from `start` (UTC), a random walk."""
    idx = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes, "min"), periods=n,
                        freq=f"{minutes}min")
    c = 60_000 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.002, n)))
    return pd.DataFrame({"close": c, "high": c * 1.001, "low": c * 0.999}, index=idx)


def _feed(block, df):
    out = []
    for ts, r in df.iterrows():
        block.update_ohlcv(r.close, r.high, r.low, r.close, 1.0, ts_ns=ts.value)
        out.append(dict(block.values) if block.settled else None)
    return out


def _keys(df, tz, start, end):
    """Each candle's session: the local date of its close less 1 ns, where that falls inside [start, end)."""
    local = (df.index - pd.Timedelta(1, "ns")).tz_convert(tz)
    clock = local.strftime("%H:%M")
    inside = (clock >= start) & (clock < end)
    return pd.Series(local.tz_localize(None).normalize(), index=df.index).where(inside)


def _expected(df, session):
    """By calendar grouping: each session's high and low, and at each candle the last session that ended before it
    and was seen from its start."""
    key = _keys(df, *WINDOWS[session])
    first = key.iloc[0]  # the data starts inside it part-way: never published
    agg = df[key.notna()].groupby(key[key.notna()]).agg({"high": "max", "low": "min"})
    last_at = {k: max(i for i, x in enumerate(key) if x == k) for k in agg.index}
    out = []
    for i in range(len(df)):
        done = [k for k in agg.index if last_at[k] < i and k != key.iloc[i] and k != first]
        out.append({"high": agg.loc[max(done)].high, "low": agg.loc[max(done)].low} if done else None)
    return out, key


@pytest.mark.parametrize("session, start", [("asia", "2026-10-15 03:00"), ("london", "2026-10-15 09:00")])
def test_r1_levels_are_the_last_completed_session_by_calendar_and_never_the_one_in_progress(session, start):
    df = _candles(start, 4 * 24 * 16)  # starts inside the session (Tokyo 12:00 / London 10:00); runs past 25 Oct
    blk = BLOCKS["session_levels"](session=session)
    got = _feed(blk, df)
    want, key = _expected(df, session)
    assert got == want
    assert sum(g is not None for g in got) > 4 * 24 * 12
    assert blk.skipped == []
    assert (blk.tz, blk.start, blk.end) == WINDOWS[session]


def test_r1_the_clock_change_moves_london_in_utc_and_leaves_asia():
    """25 Oct 2026, BST to GMT: London 08:00-16:00 local is 07:00-15:00 UTC on Friday 23 Oct and 08:00-16:00 UTC on
    Monday 26 Oct, 32 fifteen-minute candles each; Tokyo has no clock change, so Asia is 00:00-08:00 UTC on both."""
    df = _candles("2026-10-22 12:00", 4 * 24 * 5)
    london, asia = _keys(df, *WINDOWS["london"]), _keys(df, *WINDOWS["asia"])
    for key, day, first, last, n in ((london, "2026-10-23", "2026-10-23 07:15", "2026-10-23 15:00", 32),
                                     (london, "2026-10-26", "2026-10-26 08:15", "2026-10-26 16:00", 32),
                                     (asia, "2026-10-23", "2026-10-23 00:15", "2026-10-23 08:00", 32),
                                     (asia, "2026-10-26", "2026-10-26 00:15", "2026-10-26 08:00", 32)):
        rows = key[key == pd.Timestamp(day)].index
        assert (str(rows[0])[:16], str(rows[-1])[:16], len(rows)) == (first, last, n)
    got = _feed(BLOCKS["session_levels"](session="london"), df)
    at = dict(zip(df.index, got))
    mon = df[london == pd.Timestamp("2026-10-26")]
    end = pd.Timestamp("2026-10-26 16:00", tz="UTC")  # the session's last candle closes here: it still reads Friday
    assert at[end] != {"high": mon.high.max(), "low": mon.low.min()}
    assert at[end + pd.Timedelta(15, "min")] == {"high": mon.high.max(), "low": mon.low.min()}


@pytest.mark.parametrize("missing, published", [(3, True), (4, False)])
def test_r1_a_session_missing_more_than_a_tenth_of_its_candles_keeps_the_previous_levels_and_says_so(missing,
                                                                                                     published):
    """Asia on 21 Oct has 32 candles: 3 missing (9%) is still a session, 4 (12.5%) is not, and the levels read after
    it (08:15 UTC) stay those of the 20th."""
    df = _candles("2026-10-19 12:00", 4 * 24 * 3)
    key = _keys(df, *WINDOWS["asia"])
    day = df[key == pd.Timestamp("2026-10-21")]
    gap = df.drop(day.index[10:10 + missing])
    blk = BLOCKS["session_levels"](session="asia")
    at = dict(zip(gap.index, _feed(blk, gap)))
    prev = df[key == pd.Timestamp("2026-10-20")]
    kept = {"high": prev.high.max(), "low": prev.low.min()}
    left = gap[_keys(gap, *WINDOWS["asia"]) == pd.Timestamp("2026-10-21")]
    new = {"high": left.high.max(), "low": left.low.min()}
    assert new != kept
    assert at[pd.Timestamp("2026-10-21 08:15", tz="UTC")] == (new if published else kept)
    assert blk.skipped == ([] if published else [("2026-10-21", 32 - missing, 32)])


def test_r1_in_a_definition_with_its_warm_up_its_window_settings_and_bad_settings_refused():
    level = {"version": 1, "reason": "Above the Asia session's high, written before it is tested.",
             "blocks": {"asia": {"kind": "session_levels", "session": "asia"}},
             "long": {"entry": {"left": "close", "op": ">", "right": "asia.high"}}}
    assert check_definition(level, bar_minutes=15).warmup[None] == 2 * 24 * 4 + 1
    early = BLOCKS["session_levels"](session="london", start="07:00")
    assert (early.tz, early.start, early.end) == ("Europe/London", "07:00", "16:00")
    for bad, msg in (({"session": "tokyo"}, "asia, london"), ({"tz": "Mars/Olympus"}, "Mars/Olympus"),
                     ({"start": "9am"}, "HH:MM"), ({"end": "25:00"}, "HH:MM"),
                     ({"start": "17:00"}, "after"), ({"start": "10:00", "end": "10:00"}, "after")):
        with pytest.raises(ValueError, match=msg):
            check_definition({**level, "blocks": {"asia": {"kind": "session_levels", "session": "asia", **bad}}})


@pytest.mark.parametrize("minutes, start, end, ok", [
    (5, "08:07", "12:00", False), (5, "08:05", "12:00", True), (15, "08:10", "12:00", False),
    (5, "08:00", "12:03", False), (15, "08:00", "12:00", True),  # each end on a candle boundary, never rounded
    (15, "08:00", "08:45", False), (15, "08:00", "09:00", True), (1, "08:00", "08:03", False),
    (1, "08:00", "08:04", True),  # at least 4 candles of the idea's interval
])
def test_r1_a_window_off_the_candle_boundary_or_under_four_candles_is_refused_when_the_idea_is_loaded(minutes, start,
                                                                                                    end, ok):
    """Advisor R1-WIN: a window that doesn't start and end on the idea's candles, or holds fewer than 4 of them, is
    refused when the definition is checked against its interval, and never rounded to fit."""
    defn = {"version": 1, "reason": "A London window, written before it is tested.",
            "blocks": {"s": {"kind": "session_levels", "session": "london", "start": start, "end": end}},
            "long": {"entry": {"left": "close", "op": ">", "right": "s.high"}}}
    if ok:
        check_definition(defn, bar_minutes=minutes)
    else:
        with pytest.raises(ValueError, match="candle"):
            check_definition(defn, bar_minutes=minutes)
    check_definition(defn)  # with no interval known there is nothing to judge it against


@pytest.mark.parametrize("session", ["asia", "london"])
@pytest.mark.parametrize("minutes, ok", [(1, True), (5, True), (15, True), (30, True), (60, True), (120, False),
                                         (240, False), (1_440, False)])
def test_r1_def_the_default_sessions_load_on_candles_up_to_an_hour_and_are_refused_above(session, minutes, ok):
    """Advisor R1-DEF: London's default is 08:00-16:00, on the hour, so both defaults load on every candle up to 1h in
    BST and in GMT (the boundary check takes each of the zone's offsets); above 1h a session level is refused when the
    idea is loaded, failing closed, with a plain message."""
    defn = {"version": 1, "reason": "Above the session's high, written before it is tested.",
            "blocks": {"s": {"kind": "session_levels", "session": session}},
            "long": {"entry": {"left": "close", "op": ">", "right": "s.high"}}}
    if ok:
        check_definition(defn, bar_minutes=minutes)
    else:
        with pytest.raises(ValueError, match="session levels are read on candles of 1 hour or less"):
            check_definition(defn, bar_minutes=minutes)
