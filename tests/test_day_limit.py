"""QRT1 DAY LIMIT (HoE 17 Nov; prereg-trend-intraday-continuation-16nov.md, rule "one trade per asset per UTC day" and
dry-run check d): a definition's [limits] entries_per_utc_day = N opens at most N legs, long and short together, on
decision candles closing in one UTC day, so a leg a stop closed inside the day is not re-entered that day. The
candles and days an entry rule held on but the limit kept flat are counted for the sheet. Left out, nothing
changes: existing definitions keep their hashes and trades."""

from __future__ import annotations

import copy
from collections import Counter
from types import SimpleNamespace

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
    return r


def test_a_candle_closing_at_midnight_counts_to_the_day_it_opens():
    day = pd.Timestamp("2026-01-02", tz="UTC").value // DAY_NS
    midnight = (day + 1) * DAY_NS
    r = _rules(1, day, 1)
    assert r._day_full(midnight - 60 * 10**9)  # 23:59 the same day: full, counted
    assert not r._day_full(midnight)  # the 00:00 close is the next day's
    assert (r.day_limit_blocked, r._day_limit_days) == (1, {day})
    assert not _rules(2, day, 1)._day_full(midnight - 1) and not _rules(0, day, 5)._day_full(midnight - 1)
