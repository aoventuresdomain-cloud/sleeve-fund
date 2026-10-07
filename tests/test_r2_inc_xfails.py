"""QA Tester 2: acceptance tests for R2-INC (a first_touch decision made on incomplete minutes), written BEFORE the
build as STRICT xfails (PE1's #175 makes each pass, then removes its mark). Copy into tests/ to run (uses the suite's
`instrument` fixture). Run from a scratch copy, never from the shared folder.

Sources: HoQA brief 6 Oct 23:21 / 23:25 / 23:28 (PE1's names and Done-when) and the Advisor's 22:41 ruling "decided on
incomplete minutes: flag it, count it in fills-vs-model".

Class. A first_touch candle SETTLED BY ITS MINUTES (both levels in range) where at least one due minute was missing at
decision time. A candle the range settles never counts. `unknown == "missing"` is a subset (missing <= incomplete).
Interfaces ASSUMED (the behaviour is what is pinned; every missing lookup becomes AssertionError("not built: ...")):
- report: `res.first_touch[path]` and `Rules.first_touch_stats()[path]` gain `incomplete` and `incomplete_share`
  (= incomplete / judged);
- lineage: `signal["first_touch"][path]` gains `missing_minutes` (ISO list, empty when complete), `judged`, `incomplete`
  (running counts, the decision's own candle included);
- paper journal: `runtime.store.event(sleeve, "info", "first_touch_incomplete", message, ts=...)`, one per candle and
  path, EVERY time, message "<path>: decided on incomplete minutes for the candle to <iso>: k of n minutes missing
  (each missing minute as its full UTC ISO time, ...); taken as true|false";
- a backtest never journals (it still counts: an interior minute hole is counted, only the events are paper-only).
"""

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.definitions import to_params

R2INC = pytest.mark.xfail(strict=True, raises=AssertionError, reason="R2-INC not built")

M = 60_000_000_000
T0 = pd.Timestamp("2025-10-03 00:00", tz="UTC")
UP, DOWN = {"add": ["close", 1]}, {"sub": ["close", 1]}
BASE = {"version": 1, "reason": "An R2-INC acceptance case, written before it is built.", "blocks": {}}
FT = {"first_touch": {"reach": UP, "before": DOWN}}
FT_SHORT = {"first_touch": {"reach": DOWN, "before": UP}}  # a short's X is below the close, its Y above
KIND = "first_touch_incomplete"
MSG = re.compile(r"^(?P<path>[\w.\[\]]+): decided on incomplete minutes for the candle to (?P<iso>\S+): "
                 r"(?P<k>\d+) of (?P<n>\d+) minutes missing \((?P<gone>[^)]*)\); taken as (?P<as>true|false)$")

# fixed 15-minute scenario: c0 flat (warm-up), c1 judged. X = 101, Y = 99 (read at c0's close = 100).
X_FIRST = {(1, 5): (101.2, 100.0), (1, 12): (100.0, 98.8)}  # X at minute 5, Y at 12 -> true
Y_FIRST = {(1, 5): (100.0, 98.8), (1, 12): (101.2, 100.0)}  # Y at 5, X at 12 -> false
SAME = {(1, 6): (101.5, 98.5)}  # both in one minute
X_ONLY = {(1, 5): (101.2, 100.0)}  # only X in range: the range settles it


# ------------------------------------------------------------------------------------------------ backtest helpers
def _minutes(n, tf=15, spikes=(), start=T0):
    idx = pd.date_range(start + pd.Timedelta(minutes=1), periods=n * tf, freq="1min")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e3}, index=idx)
    for (k, m), (hi, lo) in dict(spikes).items():
        i = k * tf + m - 1
        df.iloc[i, df.columns.get_loc("high")] = hi
        df.iloc[i, df.columns.get_loc("low")] = lo
    return df


def _without(m, drop, tf=15):
    return m.iloc[[i for i in range(len(m)) if (i // tf, i % tf + 1) not in set(drop)]]


def _decisions(m, tf=15, start=T0):
    return m.resample(f"{tf}min", closed="right", label="right", origin=start).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})


def _run(defn, m, instrument, tf=15, drop=(), exec_prices=None, start=T0, **kw):
    ex = exec_prices if exec_prices is not None else _without(m, drop, tf)
    try:
        return run_backtest("rules", _decisions(m, tf, start), instrument, params=to_params({**BASE, **defn}),
                            bar_minutes=tf, exec_prices=ex, exec_minutes=1, half_spread=0, **kw)
    except ValueError as e:
        raise AssertionError(f"not built: {e}") from e


def _stat(res, path, key):
    try:
        return res.first_touch[path][key]
    except (KeyError, AttributeError, TypeError) as e:
        raise AssertionError(f"not built: res.first_touch[{path!r}][{key!r}] ({e!r})") from e


def _entry_lineage(res, path):
    d = next((d for d in res.decisions.values() if d["intent"] == "entry"), None)
    assert d is not None, "the rule should have entered"
    return d["signal"]["first_touch"][path]


def _lin(res, path, key):
    ft = _entry_lineage(res, path)
    assert key in ft, f"not built: lineage {path!r} lacks {key!r}"
    return ft[key]


LONG = {"long": {"entry": FT}}

# name, spikes, drop, true, unknown, missing, incomplete
TABLE = [
    ("X first, a minute missing BEFORE it: unknown, incomplete", X_FIRST, {(1, 2)}, 0, 1, 1, 1),
    ("X first, a minute missing AFTER it: still true, but incomplete", X_FIRST, {(1, 7)}, 1, 0, 0, 1),
    ("Y first, a minute missing after it: still false, but incomplete", Y_FIRST, {(1, 14)}, 0, 0, 0, 1),
    ("both reached, the gap after both reaches: settled by minutes, incomplete", X_FIRST, {(1, 14)}, 1, 0, 0, 1),
    ("two minutes missing is still ONE incomplete candle", X_FIRST, {(1, 7), (1, 9)}, 1, 0, 0, 1),
    ("(a) the range settles it (only X in range): a missing minute never counts", X_ONLY, {(1, 2)}, 1, 0, 0, 0),
    ("both levels in range, every minute present: complete", X_FIRST, set(), 1, 0, 0, 0),
    ("same minute with every minute present: ambiguous, not incomplete", SAME, set(), 0, 0, 0, 0),
]


@pytest.mark.parametrize("name, spikes, drop, true, unknown, missing, incomplete", TABLE, ids=[t[0] for t in TABLE])
def test_which_candles_count_as_decided_on_incomplete_minutes(name, spikes, drop, true, unknown, missing, incomplete,
                                                              instrument):
    res = _run(LONG, _minutes(2, spikes=spikes), instrument, drop=drop)
    assert _stat(res, "long.entry", "incomplete") == incomplete, name
    assert (_stat(res, "long.entry", "true"), _stat(res, "long.entry", "unknown"),
            _stat(res, "long.entry", "missing")) == (true, unknown, missing)
    assert _stat(res, "long.entry", "missing") <= _stat(res, "long.entry", "incomplete"), "missing is a subset"


def test_minutes_that_reach_neither_level_are_inconsistent_not_incomplete(instrument):
    """The decision candle's range says both levels were reached but the (complete) minutes say neither: unknown
    'inconsistent', and nothing was missing, so it is not incomplete."""
    m = _minutes(2, spikes=X_FIRST)
    res = _run(LONG, m, instrument, exec_prices=_minutes(2))  # complete, flat minutes under spiked decision candles
    assert _stat(res, "long.entry", "unknown") == 1 and _stat(res, "long.entry", "incomplete") == 0


def test_incomplete_share_is_incomplete_over_judged_and_zero_when_nothing_was_judged(instrument):
    """Four judged candles (c1..c4); c2 is the only one decided on incomplete minutes: share 1/4."""
    sp = {(k, m): v for k in (1, 2, 3, 4) for (_, m), v in X_FIRST.items()}
    res = _run(LONG, _minutes(5, spikes=sp), instrument, drop={(2, 7)})
    assert _stat(res, "long.entry", "judged") == 4
    assert _stat(res, "long.entry", "incomplete") == 1
    assert _stat(res, "long.entry", "incomplete_share") == pytest.approx(0.25)
    none = _run(LONG, _minutes(5), instrument)
    assert _stat(none, "long.entry", "incomplete") == 0 and _stat(none, "long.entry", "incomplete_share") == 0


def test_an_entry_names_the_missing_minutes_and_carries_the_running_counts(instrument):
    """Lineage: missing_minutes is the ISO list (empty when complete); judged and incomplete are running counts with the
    decision's own candle included. c1 is incomplete and enters (X first, minute 7 missing); c2 is complete."""
    res = _run(LONG, _minutes(2, spikes=X_FIRST), instrument, drop={(1, 7)})
    assert _lin(res, "long.entry", "missing_minutes") == ["2025-10-03T00:22:00+00:00"]
    assert (_lin(res, "long.entry", "judged"), _lin(res, "long.entry", "incomplete")) == (1, 1)
    full = _run(LONG, _minutes(2, spikes=X_FIRST), instrument)
    assert _lin(full, "long.entry", "missing_minutes") == []
    assert (_lin(full, "long.entry", "judged"), _lin(full, "long.entry", "incomplete")) == (1, 0)


def test_the_running_counts_in_the_lineage_agree_with_the_report_at_the_same_candle(instrument):
    """c1 and c2 are unknown on a missing minute 2 (false, incomplete); c3 is complete and enters: the entry carries
    judged 3 and incomplete 2 (its own candle included), and the final report agrees as nothing is judged after it."""
    sp = {(k, m): v for k in (1, 2, 3) for (_, m), v in X_FIRST.items()}
    res = _run(LONG, _minutes(4, spikes=sp), instrument, drop={(1, 2), (2, 2)})
    assert (_lin(res, "long.entry", "judged"), _lin(res, "long.entry", "incomplete")) == (3, 2)
    assert (_stat(res, "long.entry", "judged"), _stat(res, "long.entry", "incomplete")) == (3, 2)


def test_a_backtest_over_an_interior_minute_hole_counts_it_and_journals_no_event(instrument):
    """PE1 (23:25): the coverage refusal checks only the first and last decision candles, so a hole inside the data is
    counted: incomplete 1, and the right share. Only the journal events are paper-only: a backtest, with or without a
    runtime, journals nothing."""
    m = _minutes(4, spikes={(k, m): v for k in (1, 2, 3) for (_, m), v in X_FIRST.items()})
    for kw in ({}, {"risk_profile": "balanced"}):
        res = _run(LONG, m, instrument, drop={(2, 3)}, **kw)
        assert _stat(res, "long.entry", "incomplete") == 1
        assert _stat(res, "long.entry", "incomplete_share") == pytest.approx(1 / _stat(res, "long.entry", "judged"))
        journal = getattr(res, "journal", None)
        if journal is not None:
            assert not [e for e in journal.events_ if e["kind"] == KIND], "a backtest journals no incomplete event"
    assert _stat(res, "long.entry", "incomplete") == 1, "not built: the interior-hole count"


def test_a_4h_candle_ending_at_midnight_in_a_backtest_names_the_minute_with_its_own_date(instrument):
    """A UTC-aligned 4h candle 20:00 to 00:00 with 23:30 missing: the lineage names 2025-10-03T23:30, a day before the
    candle's close date. (A 4h candle crossing midnight, 22:00 to 02:00, is not on the UTC grid; it is pinned on the node
    below, as a message-format case.)"""
    start = pd.Timestamp("2025-10-03 16:00", tz="UTC")
    m = _minutes(2, tf=240, spikes={(1, 30): (101.2, 100.0), (1, 100): (100.0, 98.8)}, start=start)
    res = _run(LONG, m, instrument, tf=240, drop={(1, 210)}, start=start)
    assert _lin(res, "long.entry", "missing_minutes") == ["2025-10-03T23:30:00+00:00"]


def test_a_daily_candle_in_a_backtest_names_the_minute_with_its_own_date(instrument):
    """A daily candle (2025-10-04 00:00 to 2025-10-05 00:00) with 23:30 missing: the minute is 2025-10-04T23:30, a day
    before the candle's close date."""
    m = _minutes(2, tf=1440, spikes={(1, 30): (101.2, 100.0), (1, 900): (100.0, 98.8)})
    res = _run(LONG, m, instrument, tf=1440, drop={(1, 1410)})
    assert _lin(res, "long.entry", "missing_minutes") == ["2025-10-04T23:30:00+00:00"]


# ------------------------------------------------------------------------------------------------------ paper (hub)
class _Paper:
    """The paper path as in R2's late-minute probe: a Rules strategy fed 1-minute bars through the hub Decoder, with a
    runtime whose journal records every `store.event(...)` (a runtime that is not a backtest)."""

    def __init__(self, instrument, defn, tf=15, start=T0, spec="15-MINUTE-LAST-EXTERNAL", params=None):
        from sleeve_fund.data import bar_type_for
        from sleeve_fund.paper.hub_client import Decoder, HubStatus
        from sleeve_fund.strategies.rules import Rules, RulesConfig

        self.inst, self.tf, self.start = instrument, tf, start
        self.iid = "BTCUSDT-PERP.BINANCE"
        self.bt = bar_type_for(instrument, tf)
        self.events = []
        self.status = HubStatus()
        self.dec = Decoder(spec, status=self.status)
        self.defn, self.params = {**BASE, **defn}, params or {}
        self.Rules, self.RulesConfig = Rules, RulesConfig
        self.s = self.new_strategy()

    def new_strategy(self, events=None):
        """A (re)started strategy over the same hub feed and the same durable journal."""
        s = self.Rules(self.RulesConfig(instrument_id=self.inst.id, bar_type=self.bt, assumed_taker_fee=0.001,
                                        **{**to_params(self.defn), **self.params}))
        s.minute_source = lambda a, u: self.status.minutes_of(self.iid, a, u)
        s.runtime = SimpleNamespace(name="s", backtest=False, now=lambda: None, store=SimpleNamespace(
            event=lambda *a, **kw: self.events.append(a)))
        return s

    def feed(self, k, minute, hi, lo, refilled=False):
        ts = (self.start + pd.Timedelta(minutes=self.tf * k + minute)).value
        c = (hi + lo) / 2
        self.dec({"t": "bar", "id": self.iid, "o": f"{c:.2f}", "h": f"{hi:.2f}", "l": f"{lo:.2f}", "c": f"{c:.2f}",
                  "v": "1.000", "ts": ts, "recv": ts, "refilled": refilled}, ts + 5)

    def candle(self, k, hi, lo, close=100.0):
        ts = (self.start + pd.Timedelta(minutes=self.tf * (k + 1))).value
        p = self.inst.make_price
        from nautilus_trader.model import Bar

        return Bar(self.bt, p(100.0), p(hi), p(lo), p(close), self.inst.make_qty(self.tf), ts, ts)

    def play(self, k, spikes, drop=(), s=None):
        """Feed candle k (spikes {minute: (hi, lo)}, `drop` minutes never arrive) and close it."""
        flat = (100.0, 100.0)
        for m in range(1, self.tf + 1):
            if m not in drop:
                self.feed(k, m, *spikes.get(m, flat))
        his = [v[0] for v in spikes.values()] + [100.0]
        los = [v[1] for v in spikes.values()] + [100.0]
        (s or self.s).update_indicators(self.candle(k, max(his), min(los)))

    def warm(self):
        self.play(0, {})

    def incomplete_events(self):
        return [e for e in self.events if len(e) > 2 and e[2] == KIND]


def _only(items, what):
    assert len(items) == 1, f"expected exactly one {what}, got {len(items)}"
    return items[0]


def _parsed(events):
    out = []
    for e in events:
        assert e[1] == "info" and e[0] == "s", f"an info event on the sleeve: {e[:3]}"
        mm = MSG.match(e[3])
        assert mm, f"message format: {e[3]!r}"
        out.append(mm)
    return out


X15 = {5: (101.2, 100.0), 12: (100.0, 98.8)}
Y15 = {5: (100.0, 98.8), 12: (101.2, 100.0)}


def test_paper_journals_one_info_event_for_a_decision_on_incomplete_minutes_with_the_exact_message(instrument):
    p = _Paper(instrument, LONG)
    p.warm()
    p.play(1, X15, drop={7})
    mm = _only(_parsed(p.incomplete_events()), 'first_touch_incomplete event')
    assert (mm["path"], mm["iso"], mm["k"], mm["n"], mm["gone"], mm["as"]) == (
        "long.entry", "2025-10-03T00:30:00+00:00", "1", "15", "2025-10-03T00:22:00+00:00", "true")


def test_paper_journals_every_such_candle_but_never_a_complete_or_range_settled_one(instrument):
    """(a) + 'every time': c1 incomplete, c2 complete, c3 range-settled with a minute missing, c4 incomplete again:
    exactly two events, for c1 and c4."""
    p = _Paper(instrument, LONG)
    p.warm()
    p.play(1, X15, drop={7})
    p.play(2, X15)
    p.play(3, {5: (101.2, 100.0)}, drop={2})
    p.play(4, X15, drop={9})
    got = _parsed(p.incomplete_events())
    assert [g["iso"] for g in got] == ["2025-10-03T00:30:00+00:00", "2025-10-03T01:15:00+00:00"]
    assert [g["gone"] for g in got] == ["2025-10-03T00:22:00+00:00", "2025-10-03T01:09:00+00:00"]


def test_a_candle_read_by_both_an_entry_and_an_exit_path_gives_two_events(instrument):
    d = {"long": {"entry": FT, "exit": FT}}
    p = _Paper(instrument, d)
    p.warm()
    p.play(1, X15, drop={7})
    got = _parsed(p.incomplete_events())
    assert sorted(g["path"] for g in got) == ["long.entry", "long.exit"]
    assert len({g["iso"] for g in got}) == 1


@pytest.mark.parametrize("rule, spikes, drop, taken", [
    ("entry", X15, {2}, "false"),  # unknown on an entry: false
    ("exit", X15, {2}, "true"),  # unknown in an exit: true
    ("entry", X15, {7}, "true"),  # X first, known true, gap after
    ("entry", Y15, {14}, "false"),  # Y first, known false, gap after
    ("exit", Y15, {14}, "false"),  # Y first in an exit: false (the order is known; only ambiguity resolves true)
], ids=["entry-unknown", "exit-unknown", "entry-known-true", "entry-known-false", "exit-known-false"])
def test_taken_as_matches_the_decision_made(rule, spikes, drop, taken, instrument):
    d = {"long": {"entry": FT}} if rule == "entry" else {"long": {"entry": {"left": "close", "op": ">", "right": 0},
                                                                    "exit": FT}}
    p = _Paper(instrument, d)
    p.warm()
    p.play(1, spikes, drop=drop)
    mm = _only([g for g in _parsed(p.incomplete_events()) if g["path"] == f"long.{rule}"], 'event on that path')
    assert mm["as"] == taken
    assert p.s.first_touch_stats()[f"long.{rule}"].get("incomplete") == 1, "not built: stats['incomplete']"


def test_a_late_minute_never_rewrites_the_event_or_the_decision(instrument):
    """The missing minute turns up after the decision (refilled): no second event, the first is unchanged, and the
    counts do not move."""
    p = _Paper(instrument, LONG)
    p.warm()
    p.play(1, X15, drop={2})
    before = list(p.incomplete_events())
    st = dict(p.s.first_touch_stats()["long.entry"])
    p.feed(1, 2, 100.0, 100.0, refilled=True)
    p.play(2, {})
    assert p.incomplete_events() == before and len(before) == 1
    after = p.s.first_touch_stats()["long.entry"]
    for key in ("true", "unknown", "missing", "incomplete"):
        assert after.get(key) == st.get(key), key
    assert st.get("incomplete") == 1, "not built: stats['incomplete']"


def test_the_event_alone_rebuilds_the_count_after_a_restart(instrument):
    """(d) A restart resets the in-memory counts; the journal does not. Each event carries the path, the candle end and
    the minutes, so fills-vs-model can rebuild `incomplete` per path from history: two incomplete candles before the
    restart, one after, and the rebuilt total is 3 while the restarted strategy itself only knows 1."""
    p = _Paper(instrument, LONG)
    p.warm()
    p.play(1, X15, drop={7})
    p.play(2, X15, drop={9})
    restarted = p.new_strategy()
    restarted.update_indicators(p.candle(2, 101.2, 98.8))  # the history preload: the last closed candle sets the levels
    p.play(3, X15, drop={11}, s=restarted)
    got = _parsed(p.incomplete_events())
    assert [g["iso"] for g in got] == ["2025-10-03T00:30:00+00:00", "2025-10-03T00:45:00+00:00",
                                      "2025-10-03T01:00:00+00:00"]
    rebuilt = {}
    for g in got:
        rebuilt[g["path"]] = rebuilt.get(g["path"], 0) + 1
    assert rebuilt == {"long.entry": 3}
    assert restarted.first_touch_stats()["long.entry"].get("incomplete") == 1, "the in-memory count restarts at 0"
    assert len({(g["path"], g["iso"]) for g in got}) == len(got), "(path, candle end) identifies a decision uniquely"


def test_a_long_then_a_short_on_a_perp_each_journal_under_their_own_path(instrument):
    d = {"long": {"entry": FT}, "short": {"entry": FT_SHORT}}
    p = _Paper(instrument, d, params={"market": "perp", "allow_short": True})
    p.warm()
    # the long's X (101) is reached first at minute 5, then Y (99); the short's X (99) would come second: both read
    # the same incomplete candle, so both journal it
    p.play(1, X15, drop={7})
    got = _parsed(p.incomplete_events())
    assert sorted(g["path"] for g in got) == ["long.entry", "short.entry"]
    short = next(g for g in got if g["path"] == "short.entry")
    assert short["as"] == "false", "the short's Y (101) came first: known false"
    assert next(g for g in got if g["path"] == "long.entry")["as"] == "true"


# ------------------------------------------------------------------------------------- midnight and daily (adversarial)
def _date_of(iso_or_hhmm_minute, close_iso, hhmm):
    """The date a HH:MM minute belongs to for a candle ending at close_iso and no longer than 24h: the latest such
    time at or before the close."""
    close = pd.Timestamp(close_iso)
    t = close.normalize() + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    return t if t <= close else t - pd.Timedelta(days=1)


def _unambiguous(msg, true_minute_iso):
    """The message identifies the minute without a guess: it carries the minute's full ISO time, or states its date."""
    day = true_minute_iso[:10]
    return true_minute_iso in msg or day in msg


def test_an_hourly_candle_ending_at_midnight_identifies_its_missing_minute_unambiguously(instrument):
    """HoQA 23:28: a candle ending 00:00 with 23:30 missing. 'HH:MM' with only the close date (2025-10-04) would put
    23:30 on the wrong day. The event must carry the minute's full ISO time or its date (2025-10-03)."""
    start = pd.Timestamp("2025-10-03 22:00", tz="UTC")
    p = _Paper(instrument, LONG, tf=60, start=start, spec="1-HOUR-LAST-EXTERNAL")
    p.warm()
    p.play(1, {5: (101.2, 100.0), 50: (100.0, 98.8)}, drop={30})
    event = _only(p.incomplete_events(), 'first_touch_incomplete event')
    assert "2025-10-04T00:00:00+00:00" in event[3] and "23:30" in event[3]
    assert _unambiguous(event[3], "2025-10-03T23:30:00+00:00"), f"23:30 on the wrong day: {event[3]!r}"


def _node_event(tf_minutes, start, minute_missing, path="long.entry"):
    """A first_touch node judged directly (as PE1's own journal test does), for candles the hub cannot relay (4h, 1d)."""
    from sleeve_fund.strategies.definitions import FirstTouch

    said = []
    node = FirstTouch("x", "y", "c", None, path=path)
    t0 = int(pd.Timestamp(start).value)
    mins = [(t0 + i * M, 100.0, 101.2 if i == 30 else 100.0, 98.8 if i == tf_minutes - 90 else 100.0, 100.0)
            for i in range(1, tf_minutes + 1) if i != minute_missing]
    end = t0 + tf_minutes * M
    env = SimpleNamespace(ts=end, prev={"x": 101.0, "y": 99.0, "c": 100.0}, span=(101.2, 98.8), ohlcv=None,
                          minutes=mins, minutes_due=tf_minutes, note=None, journal=lambda k, m: said.append((k, m)))
    try:
        node.tick(env)
    except (TypeError, AttributeError) as e:
        raise AssertionError(f"not built: FirstTouch node interface ({e!r})") from e
    return said, node


@pytest.mark.parametrize("start, tf, missing, true_minute, label", [
    ("2025-10-03 22:00", 240, 90, "2025-10-03T23:30:00+00:00", "a 4h candle 22:00 to 02:00"),
    ("2025-10-04 00:00", 1440, 1410, "2025-10-04T23:30:00+00:00", "a daily candle"),
], ids=["4h-over-midnight", "daily"])
def test_a_long_candle_names_a_minute_before_midnight_with_its_own_date(start, tf, missing, true_minute, label):
    """4h and daily candles are not hub-relayed, so this judges the node directly: the event must identify 23:30 on its
    own date, not the close's."""
    said, node = _node_event(tf, start, missing)
    events = [m for k, m in said if k == KIND]
    assert len(events) == 1, "a decision on incomplete minutes is journalled"
    assert "23:30" in events[0]
    assert _unambiguous(events[0], true_minute), f"23:30 on the wrong day: {events[0]!r}"
    assert node.lineage()["missing_minutes"] == [true_minute], "the lineage names the minute in full"
