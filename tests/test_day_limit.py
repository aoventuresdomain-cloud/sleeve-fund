"""QRT1 DAY LIMIT (HoE 17 Nov; prereg-trend-intraday-continuation-16nov.md, rule "one trade per asset per UTC day" and
dry-run check d): a definition's [limits] entries_per_utc_day = N opens at most N legs, long and short together, on
decision candles closing in one UTC day, so a leg a stop closed inside the day is not re-entered that day. The
candles and days an entry rule held on but the limit kept flat are counted for the sheet. Left out, nothing
changes: existing definitions keep their hashes and trades."""

from __future__ import annotations

import copy
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.definitions import (
    DAY_NS,
    check_definition,
    definition_hash,
    to_params,
)
from sleeve_fund.strategies.rules import Rules
from tests.test_research_gaps import BASE, LONG_STOP, SHORT_STOP, _walk

DEFN = {**copy.deepcopy(BASE), "exits": {"long": {"time_stop": {"minutes": 240}, "stop": LONG_STOP},
                                         "short": {"time_stop": {"minutes": 240}, "stop": SHORT_STOP}}}
PERP = {"market": "perp", "allow_short": True}


def _limited(n):
    return {**copy.deepcopy(DEFN), "limits": {"entries_per_utc_day": n}}


@pytest.mark.parametrize("limits, words", [
    ({"entries_per_utc_day": 0}, "1 to 24"),
    ({"entries_per_utc_day": 25}, "1 to 24"),
    ({"entries_per_utc_day": 1.0}, "1 to 24"),
    ({"entries_per_utc_day": True}, "1 to 24"),
    ({"entries_per_day": 1}, "limits takes entries_per_utc_day"),
    ({}, None),
    (1, "limits takes entries_per_utc_day"),
])
def test_the_limit_is_checked(limits, words):
    d = {**copy.deepcopy(DEFN), "limits": limits}
    if words is None:
        check_definition(d)
    else:
        with pytest.raises(ValueError, match=words):
            check_definition(d)


def test_a_definition_without_limits_keeps_its_hash():
    assert definition_hash(DEFN) == definition_hash(copy.deepcopy(DEFN))
    assert definition_hash(_limited(1)) != definition_hash(DEFN)


def _run(defn, instrument, n=6000):
    return run_backtest("rules", _walk(n), instrument, params={**to_params(defn), **PERP}, bar_minutes=1,
                        half_spread=0)


def _entries_by_day(res) -> Counter:
    f = res.fills.sort_values("ts_last")
    return Counter(pd.Timestamp(f.loc[o, "ts_last"]).floor("1D") for o in f.index
                   if res.decisions[o]["intent"] == "entry")


def _stop_then_same_day_entry(res) -> int:
    """Entries on a UTC day after a stop closed a leg earlier that day."""
    f = res.fills.sort_values("ts_last")
    out, stopped = 0, set()
    for o in f.index:
        day, intent = pd.Timestamp(f.loc[o, "ts_last"]).floor("1D"), res.decisions[o]["intent"]
        if intent == "stop_loss":
            stopped.add(day)
        elif intent == "entry" and day in stopped:
            out += 1
    return out


@pytest.mark.parametrize("n", [1, 2])
def test_no_more_legs_open_in_a_utc_day_than_the_limit_a_stop_out_included(instrument, n):
    free, capped = _run(DEFN, instrument), _run(_limited(n), instrument)
    per_day = _entries_by_day(capped)
    assert max(_entries_by_day(free).values()) > n  # not vacuous: unlimited, a day opens more
    assert _stop_then_same_day_entry(free) > 0  # and re-enters after a same-day stop
    assert per_day and max(per_day.values()) <= n
    if n == 1:
        assert "stop_loss" in {d["intent"] for d in capped.decisions.values()}  # a stop did close a leg
        assert _stop_then_same_day_entry(capped) == 0
    counts = capped.rule_counts
    assert counts["day_limit"] == n and counts["day_limit_blocked"] > 0
    assert 0 < counts["day_limit_blocked_days"] <= len(per_day)
    assert "day_limit" not in free.rule_counts


def test_the_first_entry_of_each_day_is_the_one_kept(instrument):
    """Before its first entry a day is unlimited, so the limited run's entry on a day is the first one the rules
    would open from flat there: on the first day, the same fill as the unlimited run's first."""
    free, capped = _run(DEFN, instrument), _run(_limited(1), instrument)

    def first(res):
        f = res.fills.sort_values("ts_last")
        o = next(o for o in f.index if res.decisions[o]["intent"] == "entry")
        return f.loc[o, ["ts_last", "side", "avg_px"]].tolist()

    assert first(capped) == first(free)


def _rules(limit, day, n):
    r = Rules.__new__(Rules)
    r.rules = SimpleNamespace(entries_per_day=limit)
    r._day_entries, r.day_limit_blocked, r._day_limit_days = (day, n), 0, set()
    r._resume_entry_ns = None
    return r


def test_a_candle_closing_at_midnight_counts_to_the_day_it_opens():
    day = pd.Timestamp("2026-01-02", tz="UTC").value // DAY_NS
    midnight = (day + 1) * DAY_NS
    r = _rules(1, day, 1)
    assert r._day_full(midnight - 60 * 10**9)  # 23:59 the same day: full, counted
    assert not r._day_full(midnight)  # the 00:00 close is the next day's
    assert (r.day_limit_blocked, r._day_limit_days) == (1, {day})
    assert not _rules(2, day, 1)._day_full(midnight - 1) and not _rules(0, day, 5)._day_full(midnight - 1)


def test_a_resumed_leg_uses_up_its_own_day():
    """CR #360 M1: after a paper restart with a leg open, that leg counts to its entry's day, so a rule holding after
    it closes opens nothing more that day, and opens again the next."""
    day = pd.Timestamp("2026-01-02", tz="UTC").value // DAY_NS
    r = _rules(1, None, 0)
    r._resume_entry_ns = day * DAY_NS + 3 * 3_600 * 10**9
    Rules.resume_leg(r, 1, 2)
    assert r._day_entries == (day, 1)
    assert r._day_full(day * DAY_NS + 20 * 3_600 * 10**9)
    assert not r._day_full((day + 1) * DAY_NS + 3_600 * 10**9)


def test_a_leg_the_rules_open_but_no_entry_order_follows_is_counted(instrument, monkeypatch):
    """Advisor on CR #360 F3: legs the rules opened that the risk layer (or sizing, or a hold) then refused still use
    up their day; they are counted, 0 on the normal path."""
    stopless = {**_limited(1), "exits": {"time_stop": {"minutes": 240}}}
    plain = _run(stopless, instrument)
    assert len(plain.fills) and plain.rule_counts["day_limit_unsent"] == 0  # every leg sent its entry
    # the first leg held back (its stop not ready on the run's first candles): that day is spent, and counted
    assert _run(_limited(1), instrument).rule_counts["day_limit_unsent"] == 1
    monkeypatch.setattr(Rules, "_open", lambda self, *a, **k: None)  # every entry refused
    res = _run(_limited(1), instrument)
    days = len(pd.date_range(_walk(6000).index[0].floor("1D"), _walk(6000).index[-1], freq="1D"))
    assert res.fills.empty and 0 < res.rule_counts["day_limit_unsent"] <= days


def test_the_sheet_leads_with_asset_days_and_never_reads_candles_as_trades():
    from sleeve_fund.research.study import day_limit_note

    note = day_limit_note({"day_limit": 1, "day_limit_blocked": 37, "day_limit_blocked_days": 9,
                           "day_limit_unsent": 0})
    assert note.index("Asset-days with at least one blocked re-entry: 9") < note.index(
        "Candles on which the entry rule held after the day's leg: 37 (candles, not trades)")
    assert note.endswith("Rules-opened legs refused by risk: 0.")
    assert "goes to the Advisor" in day_limit_note({"day_limit": 1, "day_limit_blocked": 0,
                                                    "day_limit_blocked_days": 0, "day_limit_unsent": 2})


def test_paper_and_live_refuse_a_day_limit_until_a_restart_can_rebuild_the_days_count():
    """CR #360 F1: a flat strategy restarted after a same-day stop-out doesn't know the day is spent, so Start refuses
    [limits] as it does a studies-only block; studies run it."""
    from sleeve_fund.strategies import check_study_only, study_only_uses

    params = to_params(_limited(1))
    assert study_only_uses(params) == ["a per-day entry limit ([limits] entries_per_utc_day)"]
    with pytest.raises(ValueError, match="per-day entry limit"):
        check_study_only("rules", params)
    assert study_only_uses(to_params(DEFN)) == []


def test_a_leg_opened_and_refused_on_the_runs_last_candle_is_counted_at_the_end():
    """CR #360 F4: the leg the rules opened on the last candle has no next candle to be checked on; rule_counts checks
    it, so one that sent no entry order is still counted."""
    r = _rules(1, None, 0)
    r.funding_blocks, r.rank_blocks = list, list
    r.c, r.decisions, r.day_limit_unsent = SimpleNamespace(vol_target_size=None), {}, 0
    r._opened_with = 0  # opened with no entry orders yet, and none followed
    assert Rules.rule_counts(r)["day_limit_unsent"] == 1
    r.decisions, r._opened_with = {"o-1": {"intent": "entry"}}, 0  # this time its entry order went out
    r.day_limit_unsent = 0
    assert Rules.rule_counts(r)["day_limit_unsent"] == 0


def _hours(days=4):
    idx = pd.date_range("2025-10-03 01:00", periods=days * 24, freq="1h", tz="UTC")
    c = 100 + np.arange(len(idx)) * 0.01
    return pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 1e3}, index=idx)


def _hour_is(h):
    return {"all": [{"left": {"time": "hour"}, "op": ">=", "right": h}, {"left": {"time": "hour"}, "op": "<=", "right": h}]}


@pytest.mark.parametrize("limit", [None, 1])
def test_an_entry_the_day_limit_blocks_is_dropped_not_carried_into_the_next_day(instrument, limit):
    """Advisor (CR #360 F2): a Confirm whose event held at 23:00 on a day already traded is spent there. Unlimited, it
    opens at 23:00; limited, it does not open at 00:00 the next day either, though its confirm still holds within its
    window, and the next leg waits for a fresh event at 02:00."""
    d = {"version": 1, "reason": "A test case, written before it is run.",
         "long": {"entry": {"event": {"any": [_hour_is(2), _hour_is(23)]}, "confirm": {"left": "close", "op": ">",
                                                                                     "right": 0}, "within": 3}},
         "exits": {"time_stop": {"minutes": 60}}}
    if limit:
        d["limits"] = {"entries_per_utc_day": limit}
    res = run_backtest("rules", _hours(), instrument, params=to_params(d), bar_minutes=60, half_spread=0)
    hours = sorted({pd.Timestamp(x["signal"]["bar"]["close_ts"]).hour for x in res.decisions.values()
                    if x["intent"] == "entry"})
    assert hours == ([2] if limit else [2, 23])
    if limit:
        assert res.rule_counts["day_limit_blocked_days"] >= 2


def test_a_candle_both_sides_hold_on_in_a_full_day_drops_both_and_counts_once(instrument):
    """CR #363 M1: long and short Confirms whose events both hold at 23:00 on a day the 02:00 long already used up.
    Both are spent there (neither opens at 00:00 the next day, though each confirm still holds within its window),
    and the candle counts once to day_limit_blocked: each 23:00 is one blocked candle and one asset-day."""
    def side(event):
        return {"entry": {"event": event, "confirm": {"left": "close", "op": ">", "right": 0}, "within": 3}}

    d = {"version": 1, "reason": "A test case, written before it is run.",
         "long": side({"any": [_hour_is(2), _hour_is(23)]}), "short": side(_hour_is(23)),
         "exits": {"time_stop": {"minutes": 60}}}
    runs = {}
    for limit in (None, 1):
        dd = {**copy.deepcopy(d), **({"limits": {"entries_per_utc_day": limit}} if limit else {})}
        runs[limit] = run_backtest("rules", _hours(), instrument, params={**to_params(dd), **PERP}, bar_minutes=60,
                                   half_spread=0)

    def entries(res):
        return sorted((pd.Timestamp(x["signal"]["bar"]["close_ts"]), x["signal"]["side"]) for x in
                      res.decisions.values() if x["intent"] == "entry")

    # Unlimited, the 23:00 long opens and the short's confirm, still within its window, opens it at 01:00 the next day.
    unlimited = entries(runs[None])
    assert (23, "long") in {(t.hour, s) for t, s in unlimited} and (1, "short") in {(t.hour, s) for t, s in unlimited}
    limited = entries(runs[1])
    assert {(t.hour, s) for t, s in limited} == {(2, "long")}  # neither 23:00 side carried into the next day
    days = len({t.floor("1D") for t, _ in limited})
    counts = runs[1].rule_counts
    assert days == 4 and counts["day_limit_blocked"] == days and counts["day_limit_blocked_days"] == days
