"""Head of QA: edge-case acceptance tests for v2 P1-4 (multi-timeframe input) and board rule 5a (one bar-build
rule), written BEFORE hand-over as STRICT xfails. Platform Engineer 1 makes each one pass, then removes its
xfail mark, before QA round 1. Copy this file into tests/ to run it (it uses the suite's own helpers).

Sources (every trading-behaviour expectation below cites one of these):
- [D]  v2/p1-4-multi-timeframe-design.md (rules 1-7, "Done when").
- [A]  eng-board.md row 5, Advisor 19:48: same-instant slower-candle visibility OK; short slower warm-up at start =
       no new entries/adds, exits and stops still run (Done when: restarted with open position + short warm-up
       still exits on stop); empty slower candles recorded missing, never filled; daily anchor 00:00 UTC as a
       per-venue property.
- [5a] eng-board.md row 5a, HoE 20:00: every path builds longer bars via one shared pure function: from minutes
       present, carries `missing`; >10% missing = degraded (indicators update, no new entries, exits run). Parity
       test: hole inside a slower candle -> identical bar + decisions backtest vs paper.
- [R]  Advisor rulings 6 Oct 16:40 (via the coordinator; README "Advisor rulings"): 4.1 a degraded slower candle
       blocks entries and adds while it is the latest closed one, never exits; 4.2 `missing` counts underlying
       minutes, including those recorded missing inside decision candles that exist, against the slower candle's
       total minutes (a candle of unknown completeness counts as complete); 4.3 the short-warm-up entry block ends
       once enough slower candles closed live and the settled rule passes, no restart needed; 4.4 a non-00:00 daily
       anchor is allowed, fixed per venue (and part of the definition hash: P1-5 file); 4.5 "recorded as missing" =
       a journal event (source of truth), also shown on the Signals tab.

Assumed interface (what the branch claude/platform-z3xxr9-mtf at ec9c42f/9abbab6 and PR #155 already name; the
assertions do not depend on any other name):
- sleeve_fund.bars (PR #155): combine(parts, minutes) -> Built(open, high, low, close, volume, missing) | None,
  degraded(missing, minutes) -> bool, DEGRADED_ABOVE = 0.10; HistoryStore.read() frames carry `missing` and
  `degraded` columns.
- sleeve_fund.strategies.timeframes.SlowerCandles(minutes, step_minutes, blocks): update(o, h, l, c, v, ts_ns)
  returns the slower candle it closed (or None); `.last`, `.count`; a closed candle has open/high/low/close/
  volume/end and, under rule 5a, `missing` (absent decision candles; with 1-minute decision candles, minutes).
  A degraded flag is read from the candle (`degraded` attribute or method) or else from bars.degraded(missing).
- LongFlatStrategy.slower(minutes, *blocks) registers slower candles for a model (branch).
- A VenueProfile field whose name contains "anchor" holds the daily anchor (any of 0, "00:00", time(0),
  timedelta(0); an int or float is minutes after 00:00 UTC), and the strategy's slower candles follow it.
- The minutes missing inside a decision candle reach the strategy as #155 records them (run_backtest passes the
  frame's `missing`/`degraded` columns to LongFlatStrategy.mark_degraded); the slower candle adds them up.
- An empty slower candle is journaled as an event whose message says "missing", names the timeframe ("15-minute")
  and the candle's time (HH:MM, open or close); the Signals tab payload (runtime.publish_signals) says "missing".

Every import of P1-4 / #155 code is inside a test body, so on main the tests XFAIL rather than error."""

import dataclasses
import os
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

M = 60_000_000_000  # one minute, ns
H = 60 * M
H4 = 240 * M
DAY = 1440 * M
T0 = 1_791_201_600_000_000_000  # 12:00 UTC on 5 Oct 2026: a 4-hour (and hourly) boundary
PERP = {"market": "perp", "allow_short": True}


def _xf(item: str, done_when: str):
    return pytest.mark.xfail(strict=True, reason=f"{item} Done-when: {done_when}")


def _store(tmp_path):
    """The journal: Postgres when TEST_DATABASE_URL is set (as CI and test_sleeve_runtime.py do), else SQLite."""
    from sleeve_fund.store import Store

    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


def _candle(c: float):
    return (c, c + 1.0, c - 1.0, c, 1.0)


def _feed_minutes(s, closes, start=T0, skip=()):
    """1-minute decision candles closing at start + (k+1) minutes; `skip`: indexes the feed never had.
    Returns [(ts, closed slower candle or None)] for the candles fed."""
    out = []
    for k, c in enumerate(closes):
        if k in skip:
            continue
        ts = start + (k + 1) * M
        out.append((ts, s.update(*_candle(float(c)), ts)))
    return out


def _is_degraded(candle, minutes: int) -> bool:
    from sleeve_fund import bars

    flag = getattr(candle, "degraded", None)
    if callable(flag):
        return bool(flag(minutes))
    if flag is not None:
        return bool(flag)
    return bars.degraded(candle.missing, minutes)


def _ohlcvm(c):
    return (c.open, c.high, c.low, c.close, c.volume, c.missing)


# ---- rule 5a: one shared pure bar-build function ---------------------------------------------------------------

def test_slower_candles_are_built_by_the_one_shared_bar_function():
    """[5a] The slower-candle path calls the shared function (#155 sleeve_fund.bars) and gets the same bar from
    the same minutes, `missing` included, on a candle with interior holes."""
    import ast
    import inspect

    from sleeve_fund import bars
    from sleeve_fund.strategies import timeframes
    from sleeve_fund.strategies.timeframes import SlowerCandles

    tree = ast.parse(inspect.getsource(timeframes))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    from_pkg = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "sleeve_fund"
                for a in n.names}
    assert "sleeve_fund.bars" in mods or "bars" in from_pkg, "slower candles must use the one bar-build function"

    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 0.3, 60))
    skip = {3, 4, 17, 40, 41, 42, 43}  # seven interior minutes absent; the closing minute (59) is there
    s = SlowerCandles(60, 1)
    (closed,) = [c for _, c in _feed_minutes(s, closes, skip=skip) if c is not None]
    want = bars.combine([_candle(float(c)) for k, c in enumerate(closes) if k not in skip], 60)
    assert closed.end == T0 + H
    assert _ohlcvm(closed) == pytest.approx((want.open, want.high, want.low, want.close, want.volume, want.missing))
    assert closed.missing == 7


@pytest.mark.parametrize("skip, missing, degraded", [
    (set(), 0, False),
    ({10, 11, 12, 13, 14, 15}, 6, False),  # 6 of 60 = exactly 10%: not MORE than 10%, a normal bar
    ({10, 11, 12, 13, 14, 15, 16}, 7, True),  # 7 of 60: degraded
    ({59}, 1, False),  # the closing minute absent: closes late on the next one, stamped at its own end [D] rule 4
])
def test_a_slower_candle_carries_missing_and_is_degraded_only_above_ten_percent(skip, missing, degraded):
    from sleeve_fund.strategies.timeframes import SlowerCandles

    s = SlowerCandles(60, 1)
    out = [c for _, c in _feed_minutes(s, 100 + np.arange(61, dtype=float), skip=skip) if c is not None]
    assert out[0].end == T0 + H
    assert out[0].missing == missing
    assert _is_degraded(out[0], 60) is degraded


def test_missing_counts_minutes_when_slower_candles_are_built_from_15_minute_candles():
    """[R 4.2] An hour built from 15-minute candles: one absent interior candle is 15 minutes missing (25% of 60:
    degraded), not "1"; an hour whose four candles are all there and carry no completeness record is complete."""
    from sleeve_fund.strategies.timeframes import SlowerCandles

    s = SlowerCandles(60, 15)
    out = []
    for k in range(8):  # 12:15 to 14:00
        if k == 1:
            continue  # 12:30 never came
        ts = T0 + (k + 1) * 15 * M
        out.append(s.update(*_candle(100.0 + k), ts))
    first, second = [c for c in out if c is not None]
    assert (first.end, first.missing, _is_degraded(first, 60)) == (T0 + H, 15, True)
    assert (second.end, second.missing, _is_degraded(second, 60)) == (T0 + 2 * H, 0, False)


def test_minutes_recorded_missing_inside_decision_candles_add_up_in_the_slower_candle(monkeypatch, instrument):
    """[R 4.2] 15-minute decision bars as the store builds them (#155: `missing`/`degraded` columns, passed to the
    strategy by run_backtest). The 00:15-00:30 bar is missing 7 of its minutes and the 01:15-01:30 bar 2: the
    hourly candles read 7 (degraded: 7 of 60 > 10%), 2 (normal) and 0."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies import REGISTRY
    from test_replay import START

    seen: list = []
    monkeypatch.setitem(REGISTRY, "qa_slow_reader", _slow_reader_model(60, capture=seen))
    idx = pd.date_range(pd.Timestamp(START, tz="UTC") + pd.Timedelta(minutes=15), periods=12, freq="15min")
    c = 100 + 0.01 * np.sin(np.arange(len(idx)))
    df = pd.DataFrame({"open": c, "high": c + 0.01, "low": c - 0.01, "close": c, "volume": 1e6,
                       "missing": 0, "degraded": False}, index=idx)
    df.loc[idx[1], ["missing", "degraded"]] = [7, True]
    df.loc[idx[5], ["missing", "degraded"]] = [2, True]  # 2 of 15 is over 10% of the 15-minute bar: recorded
    run_backtest("qa_slow_reader", df, instrument, half_spread=0, bar_minutes=15)
    t = lambda h: (pd.Timestamp(START, tz="UTC") + pd.Timedelta(hours=h)).value  # noqa: E731
    assert seen == [(t(1), 7), (t(2), 2), (t(3), 0)]
    from sleeve_fund import bars

    assert [bars.degraded(m, 60) for _, m in seen] == [True, False, False]


def test_store_resample_and_strategy_built_candles_agree_on_missing_and_degraded(tmp_path):
    """[D] Done when (store = strategy-built, including a gap) + [5a] (one function on every path): a 7-minute
    hole inside one hour and a wholly empty hour. Both paths give the same hours, with the same `missing`, and
    neither makes up the empty one ([A] never filled)."""
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.strategies.timeframes import SlowerCandles

    rng = np.random.default_rng(3)
    idx = pd.date_range(pd.Timestamp(T0, tz="UTC"), periods=6 * 60, freq="1min")  # by open time
    c = 60_000 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx))))
    df = pd.DataFrame({"open": c * (1 + rng.normal(0, 1e-4, len(c))), "high": c * 1.001, "low": c * 0.999,
                       "close": c, "volume": rng.uniform(0.5, 2, len(c))}, index=idx)
    df = df.drop(idx[80:87].append(idx[180:240]))  # 7 minutes of hour 1 (13:20-13:27); all of hour 3 (15:00-16:00)
    store = HistoryStore(tmp_path)
    # The hub's write path: a minute nobody heard stays a hole (append() would vouch for it and store it flat).
    store.append_bars("KRAKEN", "BTC/USD", [(ts.value, *row) for ts, row in zip(df.index, df.itertuples(index=False))],
                      "live")
    stored = store.read("KRAKEN", "BTC/USD", 60)
    minutes = store.read("KRAKEN", "BTC/USD", 1)
    s = SlowerCandles(60, 1)
    built = [x for ts, r in minutes.iterrows()
             if (x := s.update(r.open, r.high, r.low, r.close, r.volume, ts.value)) is not None]
    assert [b.end for b in built] == [ts.value for ts in stored.index]
    assert T0 + 4 * H not in [b.end for b in built]  # the empty hour is in neither
    assert [b.missing for b in built] == [int(m) for m in stored["missing"]]
    assert [_is_degraded(b, 60) for b in built] == [bool(d) for d in stored["degraded"]]
    assert sorted({b.missing for b in built}) == [0, 7]


def test_an_empty_slower_candle_is_never_filled():
    """[A] An hour with no decision candles at all: no candle is made up for it (no block is fed one, `last` never
    is one). Where it is recorded [R 4.5] is test_an_empty_slower_candle_is_journaled_missing_and_shown_on_signals."""
    from sleeve_fund.strategies.indicators import Sma
    from sleeve_fund.strategies.timeframes import SlowerCandles

    sma = Sma(1)  # its count is the number of slower candles it was fed
    s = SlowerCandles(60, 1, [sma])
    closes = 100 + np.arange(4 * 60, dtype=float)
    out = [c for _, c in _feed_minutes(s, closes, skip=set(range(120, 180))) if c is not None]  # 14:00-15:00 empty
    assert [c.end for c in out] == [T0 + H, T0 + 2 * H, T0 + 4 * H]
    assert sma.count == s.count == 3 and s.last.end == T0 + 4 * H
    assert all(np.isfinite([c.open, c.high, c.low, c.close, c.volume]).all() for c in out)


def test_an_empty_slower_candle_is_journaled_missing_and_shown_on_the_signals_tab(tmp_path, monkeypatch):
    """[A] + [R 4.5] Paper (a recorded session replayed through the paper runtime): rsi_cross on 1-minute candles
    with a 15-minute trend filter, and no trade at all from 03:15 to 03:30, so that 15-minute candle is empty."""
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.research.replay import replay

    published: list = []
    publish = SleeveRuntime.publish_signals

    def recording(self, state):
        published.append((self.now(), state))
        return publish(self, state)

    monkeypatch.setattr(SleeveRuntime, "publish_signals", recording)
    s = np.arange(4 * 60 * 60)
    prices = 60_000 * (1 + 0.03 * np.sin(s / 4000) + 0.004 * np.sin(s / 110))
    hole = (s >= 195 * 60) & (s < 210 * 60)
    params = {"rsi_period": 5, "trend_sma": 3, "trend_minutes": 15, "time_stop_bars": 20}
    _record_with_hole(tmp_path / "s.jsonl.gz", prices, hole, "rsi_cross", params)
    store = _store(tmp_path)
    replay(tmp_path / "s.jsonl.gz", store=store)
    events = store.events("parity", limit=1000)
    said = [e for e in events if "missing" in e["message"].lower() and "15-minute" in e["message"]
            and ("03:15" in e["message"] or "03:30" in e["message"])]
    assert said, "the empty 03:15-03:30 candle must be journaled as missing"
    after = [state for at, state in published if pd.Timestamp(at) >= pd.Timestamp("2025-10-03 03:30", tz="UTC")]
    assert any("missing" in str(state).lower() for state in after), "the Signals tab must show it"


# ---- visibility: same instant, never before, no look-ahead -------------------------------------------------------

def test_a_slower_value_is_seen_on_the_candle_that_closes_it_never_before_and_never_from_the_future():
    """[D] rule 3 and Done when, [A] same instant, [D] rule 4 late close. The prefix check of
    tests/test_indicators_ref.py::test_no_look_ahead: the state after each decision candle is the same whether the
    candles after it are the real ones or a scramble of them."""
    from sleeve_fund.strategies.indicators import Sma
    from sleeve_fund.strategies.timeframes import SlowerCandles

    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, 16 * 12))  # two days of 15-minute candles
    skip = {15, 70}  # 16:00 never came (that 4-hour candle closes late, on 16:15) and one interior candle

    def states(seq):
        sma, out = Sma(2), []
        s = SlowerCandles(240, 15, [sma])
        for k, c in enumerate(seq):
            if k in skip:
                continue
            ts = T0 + (k + 1) * 15 * M
            s.update(*_candle(float(c)), ts)
            last = None if s.last is None else (s.last.end, s.last.close)
            out.append((ts, last, sma.value if sma.initialized else None))
        return out

    full = states(closes)
    for ts, last, _ in full:
        assert last is None or last[0] <= ts  # never a candle whose close is ahead
        if ts % H4 == 0:
            assert last is not None and last[0] == ts  # seen by the decision at the same instant
    late = [x for x in full if x[0] == T0 + H4 + 15 * M][0]
    assert late[1][0] == T0 + H4  # closed late, stamped at its own end
    for cut in (10, 15, 16, 40, 100, 150):
        future = closes[cut:].copy()
        rng.shuffle(future)
        scrambled = states(np.concatenate([closes[:cut], future + 37.0]))
        seen = [x for x in full if x[0] <= T0 + cut * 15 * M]
        assert scrambled[:len(seen)] == seen


def test_every_venue_carries_its_daily_anchor_at_00_utc_and_daily_candles_close_there():
    from sleeve_fund.strategies.timeframes import SlowerCandles
    from sleeve_fund.venues import VENUES

    def midnight(v) -> bool:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v == 0
        if isinstance(v, time):
            return (v.hour, v.minute) == (0, 0) and (v.tzinfo is None or v.utcoffset() == timedelta(0))
        if isinstance(v, timedelta):
            return v == timedelta(0)
        return isinstance(v, str) and v.strip().upper() in ("00:00", "00:00 UTC", "0", "00:00Z")

    for name, profile in VENUES.items():
        fields = {f.name: getattr(profile, f.name) for f in dataclasses.fields(profile) if "anchor" in f.name.lower()}
        assert fields, f"{name}: no daily anchor property"
        assert all(midnight(v) for v in fields.values()), (name, fields)

    s = SlowerCandles(1440, 60)
    start = T0 - 12 * H  # 00:00 UTC on 5 Oct
    out = [c for _, c in [(None, s.update(*_candle(100.0 + k), start + (k + 1) * H)) for k in range(72)]
           if c is not None]
    assert [c.end % DAY for c in out] == [0, 0, 0] and out[0].end == start + DAY


def test_a_venue_with_an_08_00_anchor_closes_its_daily_candles_at_08_00(monkeypatch, instrument):
    """[R 4.4] The instrument's venue (KRAKEN) given an 08:00 UTC daily anchor: a model reading daily candles under
    hourly decisions sees them close at 08:00, and the part day from 00:00 to 08:00 is dropped ([D] rule 4)."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.venues import VENUES
    from test_replay import START

    profile = VENUES[str(instrument.id.venue)]
    names = [f.name for f in dataclasses.fields(profile) if "anchor" in f.name.lower()]
    assert names, "no daily anchor property"
    now = getattr(profile, names[0])
    eight = (timedelta(hours=8) if isinstance(now, timedelta) else time(8, 0) if isinstance(now, time)
             else "08:00" if isinstance(now, str) else type(now)(480))
    monkeypatch.setattr(profile, names[0], eight)
    seen: list = []
    monkeypatch.setitem(REGISTRY, "qa_slow_reader", _slow_reader_model(1440, capture=seen))
    idx = pd.date_range(pd.Timestamp(START, tz="UTC") + pd.Timedelta(hours=1), periods=72, freq="1h")
    c = 100 + 0.01 * np.sin(np.arange(len(idx)))
    df = pd.DataFrame({"open": c, "high": c + 0.01, "low": c - 0.01, "close": c, "volume": 1e6}, index=idx)
    run_backtest("qa_slow_reader", df, instrument, half_spread=0, bar_minutes=60)
    day8 = lambda d: (pd.Timestamp(START, tz="UTC") + pd.Timedelta(days=d, hours=8)).value  # noqa: E731
    assert [end for end, _ in seen] == [day8(1), day8(2)]


# ---- warm-up short at start (Advisor 19:48) ----------------------------------------------------------------------

def _restart_holding_a_long(tmp_path, monkeypatch, legs, params):
    """A paper restart (recorded session, replayed through the paper runtime) of rsi_cross on 1-minute candles
    with a 15-minute trend filter, holding a long of 0.05 entered at 60,000, whose slower-candle warm-up the
    history store can't meet (it loads nothing). Returns the orders sent and the journal's events."""
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import replay_book
    from sleeve_fund.strategies.base import LongFlatStrategy
    from test_long_short import _fill, _record
    from test_replay import START

    store = _store(tmp_path)
    create = store.create_sleeve

    def create_holding_a_long(**kw):
        s = create(**kw)
        store.record_fill(kw["name"], side="BUY", qty=0.05, price=60_000.0, fee=1.5, order_id="carried",
                          trade_id="carried", ts=datetime.fromtimestamp(START / 1e9 - 60, tz=timezone.utc))
        return s

    store.create_sleeve = create_holding_a_long
    attach = LongFlatStrategy.attach_runtime

    def attach_with_a_short_store(self, runtime):  # as the paper node attaches the history store
        out = attach(self, runtime)
        self.attach_history(lambda instrument, bar_type, n: [])
        return out

    monkeypatch.setattr(LongFlatStrategy, "attach_runtime", attach_with_a_short_store)
    book = replay_book([_fill("BUY", 0.05, 60_000.0, 1.5)], 10_000.0)
    meta = {"balances": [f"{book['cash'] + book['qty'] * book['entry_px']:.2f} USD"],
            "sleeve": {"name": "mtf-restart", "strategy": "rsi_cross", "instrument": "BTC/USD",
                       "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000, "risk_profile": "balanced",
                       "params": {**PERP, "trend_sma": 3, "trend_minutes": 15, **params}, "maker_fee": "0.0002",
                       "taker_fee": "0.0005", "tick_seconds": 30}}
    path = tmp_path / "restart.jsonl.gz"
    _record(path, meta, legs)
    orders = replay(path, store=store)
    return orders, store.events("mtf-restart", limit=500)


def _says_short(events) -> bool:
    """[D] rule 6: it journals an error naming the timeframe (and how much is missing)."""
    return any(e["level"] in ("warning", "error") and "15-minute" in e["message"] for e in events)


def test_restarted_with_an_open_position_and_a_short_slower_warm_up_still_exits_on_its_stop(tmp_path, monkeypatch):
    """[A] Done when, verbatim. The long carried through the restart falls 3.5% through its 2% stop inside the
    first 45 minutes (before three 15-minute candles could close live): the stop still sells it."""
    orders, events = _restart_holding_a_long(tmp_path, monkeypatch, [(5, 0.0), (10, -0.035), (5, 0.0)],
                                             {"stop_loss": 0.02})
    assert _says_short(events)
    assert [(o["side"], o["intent"]) for o in orders if o["intent"] == "stop_loss"] == [("SELL", "stop_loss")]
    assert not [o for o in orders if o["intent"] == "entry"]


def test_a_short_slower_warm_up_blocks_new_entries_but_the_models_own_exit_still_runs(tmp_path, monkeypatch):
    """[A] "exits ... still run": not only the stop. The carried long's own exit (RSI back up to 55) is sent; the
    later RSI cross that would open a new leg is not, inside the 40 minutes before the filter could fill live.
    (Measured on the branch with the filter off: exit 00:18, new entries 00:25 and 00:33.)"""
    orders, events = _restart_holding_a_long(tmp_path, monkeypatch,
                                             [(12, -0.01), (12, 0.025), (8, -0.03), (8, 0.03)], {"time_stop_bars": 0})
    assert _says_short(events)
    assert ("SELL", "exit") in [(o["side"], o["intent"]) for o in orders]
    assert not [o for o in orders if o["intent"] == "entry"]


def test_the_short_warm_up_entry_block_lifts_once_the_slower_candles_have_closed_live(tmp_path, monkeypatch):
    """[R 4.3] The same restart with nothing loaded: the 15-minute SMA(3) is settled once three 15-minute candles
    have closed live (00:15, 00:30, 00:45). No entry before 00:45; the RSI cross after it (the trend allowing a
    long: measured on main, which has no warm-up gate, at 00:55) opens one in the same run."""
    orders, events = _restart_holding_a_long(
        tmp_path, monkeypatch, [(12, -0.01), (12, 0.025), (8, -0.03), (8, 0.03), (5, 0.0), (8, -0.04), (20, 0.05)],
        {"time_stop_bars": 0})
    assert _says_short(events)
    filled_at = pd.Timestamp("2025-10-03 00:45", tz="UTC")
    entries = [pd.Timestamp(o["ts"]) for o in orders if o["intent"] == "entry"]
    assert entries and all(t >= filled_at for t in entries)


# ---- degraded slower candles (rule 5a) ---------------------------------------------------------------------------

def test_a_degraded_slower_candle_still_feeds_its_blocks():
    from sleeve_fund.strategies.indicators import Sma
    from sleeve_fund.strategies.timeframes import SlowerCandles

    sma = Sma(1)
    s = SlowerCandles(60, 1, [sma])
    out = [c for _, c in _feed_minutes(s, 100 + np.arange(120, dtype=float), skip=set(range(70, 100))) if c]
    assert [c.end for c in out] == [T0 + H, T0 + 2 * H]  # the second hour is half missing
    assert sma.count == 2 and sma.value == out[1].close == 219.0


def _slow_reader_model(minutes=15, capture=None, weights=False):
    """A test model reading `minutes` candles: long from the first closed one until the third has closed. With
    `weights`, half in on the first and all in from the second (an add). `capture` collects (close ns, missing) of
    each slower candle as the model first sees it."""
    from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy
    from sleeve_fund.strategies.indicators import Sma

    class SlowReaderConfig(LongFlatConfig):
        pass

    class SlowReader(LongFlatStrategy):
        def __init__(self, config):
            super().__init__(config)
            self.slow = self.slower(minutes, Sma(1))

        def update_indicators(self, bar):
            last = self.slow.last
            if capture is not None and last is not None and (not capture or capture[-1][0] != last.end):
                capture.append((last.end, getattr(last, "missing", None)))

        def want_long(self, bar):
            n = self.slow.count
            return None if n == 0 else n < 3

        def target_weight(self, bar):
            if not weights:
                return super().target_weight(bar)
            n = self.slow.count
            return None if n == 0 else (0.5 if n == 1 else 1.0)

        def explain(self, bar, target):
            return (f"{self.slow.count} closed 15-minute candles", {})

    return SlowReader, SlowReaderConfig


def test_no_entry_while_a_degraded_slower_candle_is_the_latest_and_its_exit_still_runs(monkeypatch, instrument):
    """[5a] + [R 4.1]. The 00:00-00:15 candle misses 2 of its 15 minutes (13%): no entry from 00:15 to 00:29 while
    it is the latest closed slower candle; the entry comes at 00:30 when the whole 00:15-00:30 candle closes. The
    00:30-00:45 candle is degraded too, and the exit on it still runs at 00:45."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies import REGISTRY
    from test_replay import START

    monkeypatch.setitem(REGISTRY, "qa_slow_reader", _slow_reader_model())
    idx = pd.date_range(pd.Timestamp(START, tz="UTC") + pd.Timedelta(minutes=1), periods=60, freq="1min")
    c = 100 + 0.01 * np.sin(np.arange(len(idx)))
    df = pd.DataFrame({"open": c, "high": c + 0.01, "low": c - 0.01, "close": c, "volume": 1e6}, index=idx)
    t = lambda m: pd.Timestamp(START, tz="UTC") + pd.Timedelta(minutes=m)  # noqa: E731

    def run(frame):
        res = run_backtest("qa_slow_reader", frame, instrument, half_spread=0, bar_minutes=1)
        f = res.fills.sort_values("ts_last")
        return [(res.decisions[o]["intent"], ts) for o, ts in zip(f.index, f["ts_last"])]

    whole = run(df)
    assert whole == [("entry", t(15)), ("exit", t(45))]
    thin = run(df.drop([t(4), t(5), t(34), t(35)]))
    assert thin == [("entry", t(30)), ("exit", t(45))]


def test_no_add_while_a_degraded_slower_candle_is_the_latest(monkeypatch, instrument):
    """[R 4.1] A weight-sized test model: half in on the first 15-minute candle, all in from the second. With the
    00:15-00:30 candle missing 2 of its minutes, the add waits from 00:30 until the whole 00:30-00:45 candle closes."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies import REGISTRY
    from test_replay import START

    monkeypatch.setitem(REGISTRY, "qa_slow_reader", _slow_reader_model(weights=True))
    idx = pd.date_range(pd.Timestamp(START, tz="UTC") + pd.Timedelta(minutes=1), periods=60, freq="1min")
    c = 100 + 0.01 * np.sin(np.arange(len(idx)))
    df = pd.DataFrame({"open": c, "high": c + 0.01, "low": c - 0.01, "close": c, "volume": 1e6}, index=idx)
    t = lambda m: pd.Timestamp(START, tz="UTC") + pd.Timedelta(minutes=m)  # noqa: E731

    def buys(frame):
        res = run_backtest("qa_slow_reader", frame, instrument, {"rebalance_band": 0.1}, half_spread=0,
                           bar_minutes=1)
        f = res.fills.sort_values("ts_last")
        return [ts for side, ts in zip(f["side"], f["ts_last"]) if side == "BUY"]

    assert buys(df) == [t(15), t(30)]
    assert buys(df.drop([t(19), t(20)])) == [t(15), t(45)]


# ---- parity with a hole (rule 5a) --------------------------------------------------------------------------------

def test_a_hole_inside_a_slower_candle_gives_the_same_candle_and_the_same_trades_in_backtest_and_paper(tmp_path,
                                                                                                     monkeypatch):
    """[5a] parity line, on the design's own parity set-up (1-minute RSI under a 15-minute trend average, the
    recorded paper session replayed tick by tick against the backtest of the same trades as 1-minute bars). Two
    whole minutes inside the 03:15-03:30 candle have no trades at all, on both paths."""
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.replay import replay
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies import timeframes
    from test_tick_bar_parity import INST, SPREAD, START, _by_order, _same_trades

    built: list = []
    update = timeframes.SlowerCandles.update

    def recording(self, *a, **k):
        out = update(self, *a, **k)
        if out is not None:
            built.append(out)
        return out

    monkeypatch.setattr(timeframes.SlowerCandles, "update", recording)
    s = np.arange(6 * 60 * 60)
    prices = 60_000 * (1 + 0.03 * np.sin(s / 4000) + 0.004 * np.sin(s / 110))
    hole = (s >= 200 * 60) & (s < 202 * 60)  # 03:20 to 03:22: no trades
    params = {"rsi_period": 5, "trend_sma": 3, "trend_minutes": 15, "time_stop_bars": 20}

    ticks = _record_with_hole(tmp_path / "s.jsonl.gz", prices, hole, "rsi_cross", params)
    orders, fills = replay(tmp_path / "s.jsonl.gz", with_fills=True)
    paper = _by_order(fills, {o["order_id"]: o["intent"] for o in orders})
    paper_candles, built[:] = list(built), []

    bars = ticks.resample("1min", closed="left", label="right").ohlc().dropna()  # minutes with trades only
    bars["volume"] = 0.6 / BOOK_SHARE
    res = run_backtest("rsi_cross", bars, INST, params=params, starting_capital=10_000, risk_profile="aggressive",
                       bar_minutes=1, half_spread=SPREAD / 2 / float(prices[0]))
    j = res.journal
    backtest = _by_order(j.fills_, {k: o["intent"] for k, o in j.orders_.items()})
    hole_end = (pd.Timestamp(START, tz="UTC") + pd.Timedelta(minutes=210)).value  # the 03:15-03:30 candle
    end = START + len(prices) * 1_000_000_000  # paper's last minute never closes: no trade comes after it
    same = lambda cs: [_ohlcvm(c) + (c.end,) for c in cs if c.end < end]  # noqa: E731
    assert len(same(built)) >= 20 and same(built) == same(paper_candles)
    assert [c.missing for c in built if c.end == hole_end] == [2]
    assert len(paper) >= 6
    _same_trades(paper, backtest)


def _record_with_hole(path, prices, hole, strategy, params):
    """test_tick_bar_parity._record, with no trade or quote at all in the seconds `hole` marks."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.paper.recorder import Recorder
    from test_tick_bar_parity import INST, SPREAD, START

    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "parity", "strategy": strategy, "instrument": "BTC/USD",
                           "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000,
                           "risk_profile": "aggressive", "params": params, "max_notional": None, "maker_fee": "0.004",
                           "taker_fee": "0.008", "tick_seconds": 30}}
    rec.start(INST)
    stamps, kept = [], []
    for s, px in enumerate(np.round(prices, 1)):
        if hole[s]:
            continue
        t = START + s * 1_000_000_000
        rec.trade(TradeTick(INST.id, Price(px, 1), Quantity(0.01, 8), AggressorSide.BUY if s % 2 else AggressorSide.SELL,
                            TradeId(str(s)), t, t + 1000))
        rec.quote(QuoteTick(INST.id, Price(px - SPREAD / 2, 1), Price(px + SPREAD / 2, 1), Quantity(1, 8),
                            Quantity(1, 8), t + 2000, t + 3000))
        stamps.append(t)
        kept.append(px)
    rec.close()
    return pd.Series(kept, index=pd.to_datetime(stamps, utc=True))
