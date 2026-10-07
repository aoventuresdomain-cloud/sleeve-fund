"""SPREAD-PIT: strict xfails written before the build (Head of QA, 7 Oct; Advisor 23:42 and 23:44). Owner: PE1, after
GAP-LIQ-CAP. Lands before any G1 judgement, not before paper start.

The ruling (advisor-rulings.md, "23:42 6 Oct"): the half spread is a point-in-time series. Each measurement has an
effective-from time; backtest and paper read the value in force at that minute (a backtest using the latest
measurement for an older minute is a small look-ahead). Paper picks up new measurements when they land (a daily
refresh is OK). Today the backtest charges one run-level number, the latest measurement (spreads.resolve:
research/run.py:126, dashboard/preview.py:291 -> research/runner.py:169 -> instruments.py:261), and paper resolves once
at node start (paper/node.py:260).

Two measurements are recorded throughout: LOW = 0.1% effective from T1 and HIGH = 0.3% effective from T2
(Store.record_spread with ts=T). Both sit above the 0.05% moving-market floor, so a stop booked with the wrong
measurement books a different price with or without D13's floor.

The Advisor's answers (advisor-rulings.md, "GAP-LIQ-CAP / SPREAD-PIT edges", 7 Oct ~00:19-00:22): (7) a measurement is
effective from max(end of its measured window, the time it landed); history with no landing time uses the window end.
The spreads table has one time, measured_at, and paper writes it when it records (paper/runtime.py on_quote:
record_spread(..., ts=now) at the end of the hour it measured), so measured_at is both the window end and the landing
time, and effective-from = measured_at meets (7) (PE1, 7 Oct). The schema cannot record a landing later than the
window end, so no case pins one (see the README). (8) Before the first measurement: the venue's assumption, with the
0.05% floor still applied to moving-market fills. (9) Fills-vs-model counts the daily-refresh lag in its own labelled
bucket, never inside a tolerance.

Marks: SPREAD_PIT = `xfail(strict=True, raises=AssertionError, reason="SPREAD-PIT ...")`. Tests named
`test_control_...` carry no mark. Paper cases use #146's QA harness (tests/test_hub_146_qa.py) and skip on a head
without it. Each test checks its set-up first; a missing interface fails as AssertionError("not built: ...").
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPREAD_PIT = pytest.mark.xfail(strict=True, raises=AssertionError, reason="SPREAD-PIT: not built yet (Advisor 23:42)")
SPREAD_PIT_D13 = pytest.mark.xfail(strict=True, raises=AssertionError,
                                   reason="SPREAD-PIT (+D13: the backtest stop's 0.05% floor): not built yet")

START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC, the #146 harness's clock
S = 1_000_000_000
M = 60 * S
BASE = 60_000.0
LOW, HIGH, FLOOR = 0.001, 0.003, 0.0005  # 0.1% from T1, 0.3% from T2, both above the moving-market floor
VENUE, PAIR = "KRAKEN", "BTC/USD"
TOL = 5e-6  # fees are charged to the cent: 0.01 on ~5,000 of notional


def _ts(minute: float) -> datetime:
    return datetime.fromtimestamp((START + int(minute * M)) / 1e9, tz=timezone.utc)


T1 = _ts(-24 * 60)  # 2 Oct 00:00


def _store(rows):
    """A fresh in-memory store holding the measurements (half spread, effective from). Built as Store.in_memory()
    builds one, never through it: the paper cases hand their harness a store through that name."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from sleeve_fund.store import Store

    st = Store(engine=create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))
    for hs, at in rows:
        st.record_spread(VENUE, PAIR, hs, 500, ts=at)
    return st


def _series(store):
    """ASSUMED interface: spreads.series(venue, instrument, store), the measurements as a point-in-time series whose
    .at(ts) gives the half spread in force at ts (a float, or a SpreadQuote); run_backtest and run_study take it as
    half_spread."""
    from sleeve_fund import spreads

    fn = getattr(spreads, "series", None)
    if fn is None:
        raise AssertionError("not built: spreads.series(venue, instrument, store) (the half spread as a point-in-time "
                             "series)")
    try:
        return fn(VENUE, PAIR, store)
    except TypeError as exc:
        raise AssertionError(f"not built: spreads.series(venue, instrument, store) as assumed: {exc}") from exc


def _at(series_like, ts) -> float:
    at = getattr(series_like, "at", None)
    if at is None:
        raise AssertionError(f"not built: a point-in-time spread; got {series_like!r}, one number for every minute")
    v = at(ts)
    return float(getattr(v, "half_spread", v))


# ------------------------------------------------------------------------------------------------- resolve


def test_spread_pit_resolve_gives_the_measurement_in_force_at_a_minute():
    """ASSUMED: spreads.resolve(venue, instrument, store, at=ts). Before T1: the venue's assumption; from T1: LOW;
    from T2 (inclusive): HIGH. Never the later measurement for an earlier minute. T2 is the measurement's measured_at,
    the end of the hour it measured (as paper writes it): a minute inside that hour, before T2, still gets LOW, so the
    measurement never reaches back to the start of its window (Advisor 00:19 (7))."""
    from sleeve_fund import spreads
    from sleeve_fund.venues import venue

    t2 = _ts(30)
    st = _store([(LOW, T1), (HIGH, t2)])
    assumed = venue(VENUE).assumed_half_spread
    assert spreads.resolve(VENUE, PAIR, st).half_spread == HIGH, "set-up: today's resolve gives the latest, HIGH"
    want = {"before T1": assumed, "T1 + 1h": LOW, "inside T2's measured hour (T2 - 30 min)": LOW, "T2 - 1 min": LOW,
            "T2": HIGH, "T2 + 1 day": HIGH}
    at = {"before T1": T1 - timedelta(hours=1), "T1 + 1h": T1 + timedelta(hours=1),
          "inside T2's measured hour (T2 - 30 min)": t2 - timedelta(minutes=30),
          "T2 - 1 min": t2 - timedelta(minutes=1), "T2": t2, "T2 + 1 day": t2 + timedelta(days=1)}
    try:
        got = {k: spreads.resolve(VENUE, PAIR, st, at=v).half_spread for k, v in at.items()}
    except TypeError as exc:
        raise AssertionError(f"not built: spreads.resolve(venue, instrument, store, at=ts): {exc}") from exc
    assert got == want, (got, want)


def test_control_resolve_with_one_measurement_gives_it_and_with_none_the_venues_assumption():
    from sleeve_fund import spreads
    from sleeve_fund.venues import venue

    assert spreads.resolve(VENUE, PAIR, _store([])).half_spread == venue(VENUE).assumed_half_spread
    q = spreads.resolve(VENUE, PAIR, _store([(LOW, T1)]))
    assert (q.half_spread, q.source) == (LOW, "measured")


# ------------------------------------------------------------------------------------------------ backtest

BT_CASES = [pytest.param(1, False, "aggressive", id="spot-long"),
            pytest.param(-1, True, "conservative", id="perp-1x-short")]


def _minutes(side: int, n: int = 40) -> pd.DataFrame:
    """1-minute bars by close time from #146's flat per-second path (60,000, +1e-7 a second), with a 2% move against
    `side` from 00:20:30 to 00:20:50 that recovers: a 1% stop fills inside the minute to 00:21."""
    s = np.arange(n * 60)
    p = BASE * (1 + 1e-7 * s)
    p[(s >= 20.5 * 60) & (s < (20 + 50 / 60) * 60)] *= 1 - side * 0.02
    sec = pd.Series(np.round(p, 1), index=pd.to_datetime(START + s * S, unit="ns", utc=True))
    m = sec.resample("1min", closed="left", label="left").ohlc()
    m = m.set_axis(m.index + pd.Timedelta(minutes=1))
    m["volume"] = 1e6
    return m


def _probe_classes():
    """Holds `side` for the 1-minute bars closing in [enter, leave) minutes after START, flat otherwise (as
    tests/test_hub_146_qa.py's probe, copied so the backtest cases need no #146 harness)."""
    from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy

    class QAPitProbeConfig(LongFlatConfig):
        def __init__(self, *, enter: int = 5, leave: int = 60, side: int = 1, **kwargs) -> None:
            super().__init__(**kwargs)
            self.enter, self.leave, self.side = enter, leave, side

    class QAPitProbe(LongFlatStrategy):
        def __init__(self, config) -> None:
            super().__init__(config)
            self.enter, self.leave, self.side = config.enter, config.leave, config.side

        def _k(self, bar) -> int:
            return int((bar.ts_event - START) // M)

        def want_long(self, bar):
            return self.enter <= self._k(bar) < self.leave

        def want_side(self, bar):
            return self.side if self.enter <= self._k(bar) < self.leave else 0

    return QAPitProbe, QAPitProbeConfig


def _bt(side: int, perp: bool, profile: str, half_spread, bars=None, leave: int = 60):
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.venues import venue

    params = {"enter": 5, "leave": leave, "side": side, "stop_loss": 0.01}
    if perp:
        params.update(market="perp", allow_short=True)
    saved = REGISTRY.get("qa_pit_probe")
    REGISTRY["qa_pit_probe"] = _probe_classes()
    try:
        inst = venue(VENUE).instrument("BTC", "USD", price_precision=1)
        try:
            res = run_backtest("qa_pit_probe", _minutes(side) if bars is None else bars, inst, params=params,
                               starting_capital=10_000, risk_profile=profile, bar_minutes=1, half_spread=half_spread)
        except (TypeError, ValueError) as exc:
            if isinstance(half_spread, float):
                raise
            raise AssertionError(f"not built: run_backtest(half_spread=<point-in-time series>): {exc}") from exc
    finally:
        if saved is None:
            REGISTRY.pop("qa_pit_probe", None)
        else:
            REGISTRY["qa_pit_probe"] = saved
    return res, params, inst


def _paid(res, params, inst) -> dict:
    """Per intent, (fill minute, the share of the reference price paid in spread): the price moved against the order
    from its reference (the entry's decision close; the stop's trigger) plus whatever the fee carries beyond the
    taker rate, over the reference notional. The same whether the spread is in the price (D13) or in the fee."""
    from sleeve_fund import markets
    from sleeve_fund.instruments import FeeSchedule

    taker = float(markets.fees_for(params, FeeSchedule(inst.maker_fee, inst.taker_fee), VENUE).taker)
    j = res.journal
    orders = {o["order_id"]: o for o in j.orders_.values()}
    out = {}
    for f in sorted(j.fills_, key=lambda f: f["ts"]):
        o = orders[f["order_id"]]
        sig = o.get("signal") or {}
        ref = sig.get("trigger") if o["intent"] == "stop_loss" else sig.get("close")
        buy = 1 if f["side"] == "BUY" else -1
        q, px = f["qty"], f["price"]
        paid = (buy * (px - ref) * q + f["fee"] - taker * q * px) / (q * ref)
        out[o["intent"]] = (pd.Timestamp(f["ts"]).ceil("1min"), paid)
    return out


@pytest.mark.parametrize("side,perp,profile", BT_CASES)
@pytest.mark.parametrize("t2_minute,want_stop", [pytest.param(30, LOW, id="stop-between-T1-and-T2"),
                                                 pytest.param(15, HIGH, id="stop-after-T2")])
def test_spread_pit_a_backtest_charges_each_fill_the_half_spread_in_force_at_its_minute(side, perp, profile,
                                                                                        t2_minute, want_stop):
    """Entry at 00:05, stop at 00:21. With T2 at 00:30 the stop falls between T1 and T2 and pays LOW (0.1%), never the
    later HIGH (0.3%); with T2 at 00:15 it pays HIGH. The entry (00:05, before T2 either way) pays LOW. Both values
    sit above the 0.05% floor, so the floor (D13) never decides the outcome."""
    t2 = _ts(t2_minute)
    st = _store([(LOW, T1), (HIGH, t2)])
    ref, p0, i0 = _bt(side, perp, profile, 0.0)  # set-up, on today's float interface
    plain = _paid(ref, p0, i0)
    assert set(plain) == {"entry", "stop_loss"}, ("set-up: one entry and its stop", plain)
    assert plain["entry"][0] == pd.Timestamp(_ts(5)) and plain["stop_loss"][0] == pd.Timestamp(_ts(21)), plain
    assert abs(plain["entry"][1]) < TOL and min(abs(plain["stop_loss"][1]), abs(plain["stop_loss"][1] - FLOOR)) < TOL, (
        "set-up: at zero spread the entry pays nothing beyond the taker fee, and the stop nothing or, with D13, "
        "the 0.05% floor", plain)
    res, params, inst = _bt(side, perp, profile, _series(st))
    got = {k: round(v[1], 6) for k, v in _paid(res, params, inst).items()}
    want = {"entry": LOW, "stop_loss": max(want_stop, FLOOR)}
    assert set(got) == set(want) and all(abs(got[k] - want[k]) < TOL for k in want), (got, want)


@pytest.mark.parametrize("side,perp,profile", BT_CASES)
def test_spread_pit_a_backtest_stop_pays_the_0_05pc_floor_when_the_measurement_in_force_is_below_it(side, perp,
                                                                                                   profile):
    """0.01% in force from T1, nothing later: the entry pays 0.01% (no floor on entries, Advisor 21:03), the stop the
    0.05% floor (max(half spread, 0.05%) on moving-market fills)."""
    st = _store([(0.0001, T1)])
    res, params, inst = _bt(side, perp, profile, _series(st))
    got = {k: round(v[1], 6) for k, v in _paid(res, params, inst).items()}
    want = {"entry": 0.0001, "stop_loss": FLOOR}
    assert set(got) == set(want) and all(abs(got[k] - want[k]) < TOL for k in want), (got, want)


@pytest.mark.parametrize("side,perp,profile", BT_CASES)
def test_spread_pit_before_the_first_measurement_a_backtest_uses_the_venues_assumption_and_the_stop_the_floor(
        monkeypatch, side, perp, profile):
    """Advisor 00:19 (8): minutes before the first measurement use the venue's assumption, and the 0.05% floor still
    applies to moving-market fills. The venue's assumption is set to 0.01% here (below the floor; Kraken's own is 0.05%,
    equal to it, so it could not tell), and the only measurement (HIGH) lands the day after the run: the entry pays
    0.01%, the stop the 0.05% floor, and neither pays the later HIGH."""
    import dataclasses

    from sleeve_fund import venues

    monkeypatch.setitem(venues.VENUES, VENUE, dataclasses.replace(venues.venue(VENUE), assumed_half_spread=0.0001))
    st = _store([(HIGH, _ts(24 * 60))])
    from sleeve_fund import spreads

    assert spreads.resolve(VENUE, PAIR, _store([])).half_spread == 0.0001, "set-up: the venue assumes 0.01%"
    res, params, inst = _bt(side, perp, profile, _series(st))
    got = {k: round(v[1], 6) for k, v in _paid(res, params, inst).items()}
    want = {"entry": 0.0001, "stop_loss": FLOOR}
    assert set(got) == set(want) and all(abs(got[k] - want[k]) < TOL for k in want), (got, want)


def test_spread_pit_a_store_study_hands_the_engine_the_series_not_the_latest_measurement(monkeypatch):
    """research/run.py:126 resolves the latest measurement once and passes that number to every run. Built, the
    study passes the series: LOW (0.1%) for a minute between T1 and T2, HIGH (0.3%) from T2."""
    from sleeve_fund.research import run as run_mod
    from sleeve_fund.research import study as study_mod

    t1, t2 = datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 6, 1, tzinfo=timezone.utc)
    st = _store([(LOW, t1), (HIGH, t2)])
    idx = pd.date_range("2022-01-01", "2025-12-31", freq="D", tz="UTC")
    daily = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6}, index=idx)

    class History:
        def read(self, venue, pair, minutes, start=None):
            if minutes == 1440:
                return daily
            fine = daily.resample(f"{minutes}min").ffill()
            return fine if start is None else fine[fine.index >= start]

        def coverage(self, venue, pair):
            return type("Cov", (), {"last": idx[-1]})()

    seen = {}

    class Captured(Exception):
        pass

    def fake_run_study(*a, **kw):
        seen.update(kw)
        raise Captured

    monkeypatch.setattr(study_mod, "run_study", fake_run_study)
    req = run_mod.StudyRequest(strategy="ping_pong", pair=PAIR, venue=VENUE, minutes=1440)
    with tempfile.TemporaryDirectory(prefix="qa-spread-pit-") as d:
        with pytest.raises(Captured):
            run_mod.run_store_study(req, store=st, ledger_path=Path(d) / "ledger.jsonl", out_dir=Path(d),
                                    history=History())
    assert "half_spread" in seen, ("set-up: the study reached run_study", sorted(seen))
    h = seen["half_spread"]
    got = {"2025-03-01": _at(h, datetime(2025, 3, 1, tzinfo=timezone.utc)),
           "2025-07-01": _at(h, datetime(2025, 7, 1, tzinfo=timezone.utc))}
    assert got == {"2025-03-01": LOW, "2025-07-01": HIGH}, got


# ---------------------------------------------------------------------------------------------------- paper


def _harness(monkeypatch):
    try:
        import test_hub_146_qa as h

        from sleeve_fund.paper import hub_client  # noqa: F401
    except ImportError:
        pytest.skip("needs #146's outage replay and its QA harness (tests/test_hub_146_qa.py); not on this head")
    from sleeve_fund.strategies import REGISTRY

    monkeypatch.setitem(REGISTRY, "probe", h._probe_classes())
    return h


def _with_store(monkeypatch, rows):
    """The paper harness makes its journal with Store.in_memory(): hand it one already holding the measurements. The
    node's own hourly median of its quotes (SleeveRuntime.on_quote, SPREAD_EVERY) is switched off, so the two
    measurements are the only ones (the harness quotes a fixed 12-dollar spread, 0.01%)."""
    from sleeve_fund.paper import runtime as runtime_mod
    from sleeve_fund.store import Store

    st = _store(rows)
    monkeypatch.setattr(Store, "in_memory", staticmethod(lambda: st))
    monkeypatch.setattr(runtime_mod, "SPREAD_EVERY", timedelta(days=3650))
    return st


def _node_start(rows, at: datetime) -> dict:
    """ASSUMED adapter (one place to change): the strategy config as paper/node.py:260 builds it when the node starts
    at `at`: assumed_half_spread = spreads.resolve(venue, instrument, store) over the measurements landed by then.
    Built, the strategy reads the value in force at each minute from its runtime's store (spreads.resolve(...,
    at=minute)), this start value staying the fallback; if PE1 passes a series through the config instead, change this
    function only."""
    from sleeve_fund import spreads

    return {"assumed_half_spread": spreads.resolve(VENUE, PAIR, _store([r for r in rows if r[1] <= at])).half_spread}


def _replayed_stop(run, h, side: int, stop: float = 0.01):
    """(booked price, the stop level from the journaled entry) of the replayed stop."""
    entry = h.fill_px(run, "entry")
    ex = h.first_exit(run.sequence())
    # the filled order only: since #182 a watched stop is also journaled as its own order ("STOP (watched)",
    # status "triggered") beside the replayed MARKET fill (PE1, set-up)
    (o,) = [o for o in run.orders if o["intent"] == ex[0] and o.get("status") == "filled"]
    return ex, o, entry * (1 - side * stop)


def test_spread_pit_paper_started_before_t2_books_the_new_value_on_an_outage_stop_a_day_after_t2_lands(monkeypatch):
    """The bound: a node started at 00:00 on 3 Oct with LOW (0.1%) in force; HIGH (0.3%) lands at 00:10 (T2) while it
    runs. At 00:27 on 4 Oct (T2 + 24h17m, past any daily refresh) the hub is away and a 2% dip crosses the 1% stop;
    the replayed stop books the level less HIGH, the value in force at that minute, not the node-start LOW."""
    h = _harness(monkeypatch)
    D = 24 * 60
    rows = [(LOW, T1), (HIGH, _ts(10))]
    _with_store(monkeypatch, rows)
    n = D + 45
    p = h.shape(h.flat_prices(n), D + 27.5, D + 27 + 50 / 60, 0.98)
    away = (START + (D + 26) * M, START + (D + 32) * M)
    back = START + (D + 32) * M + 5 * S
    keep = set(range(0, n * 60, 15))  # one trade a 15 s outside the outage, to keep a 24-hour run quick
    gone = frozenset(s for s in range(n * 60) if s not in keep or (D + 26) * 60 <= s < (D + 32) * 60)
    run = h.paper(p, side=1, perp=False, profile="aggressive", leave=10**6, gone=gone, away=away, back_at=back,
                  extra=_node_start(rows, _ts(0)))
    ex, o, level = _replayed_stop(run, h, 1)
    assert ex[0] == "stop_loss" and h.minute(D + 32) <= ex[2] <= h.minute(D + 33), ("set-up: the stop replayed on "
                                                                                    "the hub's return", ex)
    assert h.is_modelled(o["signal"]), ("set-up: booked by the outage replay's model", o["signal"])
    assert _node_start(rows, _ts(0))["assumed_half_spread"] == LOW, "set-up: LOW in force when the node started"
    got = round(ex[3] / level - 1, 6)
    assert abs(got + HIGH) < 2e-6, {"booked_vs_level": got, "want": -HIGH}


@pytest.mark.parametrize("label, perp, profile, side", [("spot-long", False, "aggressive", 1),
                                                        ("perp-1x-short", True, "conservative", -1)],
                         ids=["spot-long", "perp-1x-short"])
def test_spread_pit_restart_replay_of_a_minute_before_t2_equals_the_backtest_at_that_minute(monkeypatch, label, perp,
                                                                                           profile, side):
    """The process is down 00:06-00:12; a 2% dip crosses the 1% stop at 00:07:30; HIGH (0.3%) lands at 00:09 (T2),
    before the restart, so the restarted node starts on HIGH. The replayed stop is booked with the half spread in
    force at 00:07 (LOW, 0.1%), and equals the backtest's stop at that minute on the same series: level less LOW in
    both."""
    h = _harness(monkeypatch)
    rows = [(LOW, T1), (HIGH, _ts(9))]
    _with_store(monkeypatch, rows)
    p = h.shape(h.flat_prices(40), 7.5, 7 + 50 / 60, h.adverse(side, 0.02))
    run = h.restart(p, 6, 12, side=side, perp=perp, profile=profile, leave=30, extra=_node_start(rows, _ts(12)))
    ex, o, level = _replayed_stop(run, h, side)
    assert ex[0] == "stop_loss" and ex[2] <= h.minute(13), ("set-up: the stop replayed on return", ex)
    assert h.is_modelled(o["signal"]), ("set-up: booked by the outage replay's model", o["signal"])
    assert _node_start(rows, _ts(12))["assumed_half_spread"] == HIGH, "set-up: the restarted node starts on HIGH"
    replay_paid = side * (1 - ex[3] / level)
    st = _store(rows)
    mins = h.minutes_of(p)
    bars = mins.set_axis(mins.index + pd.Timedelta(minutes=1))
    bars["volume"] = 1e6
    try:
        series = _series(st)
    except AssertionError as exc:
        raise AssertionError(f"{exc}; meanwhile the replay booked the level less {replay_paid:.4%} (want "
                             f"{LOW:.4%})") from None
    bt = _paid(*_bt(side, perp, profile, series, bars=bars, leave=30))
    assert bt.get("stop_loss", (None,))[0] == pd.Timestamp(_ts(8)), ("set-up: the backtest's stop in the minute to "
                                                                      "00:08", bt)
    got = {"replay": round(replay_paid, 6), "backtest": round(bt["stop_loss"][1], 6)}
    assert abs(replay_paid - LOW) < 2e-6 and abs(bt["stop_loss"][1] - LOW) < TOL, (got, {"both": LOW})


@pytest.mark.parametrize("label, perp, profile, side", [("spot-long", False, "aggressive", 1),
                                                        ("perp-1x-short", True, "conservative", -1)],
                         ids=["spot-long", "perp-1x-short"])
def test_control_the_replay_books_the_0_05pc_floor_when_the_measurement_is_below_it(monkeypatch, label, perp, profile,
                                                                                   side):
    """0.01% in force from T1 and nothing later: the restart replay books the stop at its level less the 0.05% floor
    (#146, Advisor 20:39), today and after SPREAD-PIT."""
    h = _harness(monkeypatch)
    rows = [(0.0001, T1)]
    _with_store(monkeypatch, rows)
    p = h.shape(h.flat_prices(40), 7.5, 7 + 50 / 60, h.adverse(side, 0.02))
    run = h.restart(p, 6, 12, side=side, perp=perp, profile=profile, leave=30, extra=_node_start(rows, _ts(12)))
    ex, o, level = _replayed_stop(run, h, side)
    assert ex[0] == "stop_loss" and h.is_modelled(o["signal"]), ex
    assert abs(side * (1 - ex[3] / level) - FLOOR) < 2e-6, (ex, level)


# ------------------------------------------------------------------------------------------- fills-vs-model


def _fills_vs_model(paper, model, store):
    """ASSUMED interface (one place to change): sleeve_fund.research.fills_vs_model.compare(paper_fills, model_fills,
    store=..., venue=..., instrument=...), fills as the journal's rows (paper's carrying the `half_spread` it booked
    with), returning a report whose `buckets` maps a label to {"count", "amount"}."""
    try:
        from sleeve_fund.research import fills_vs_model
    except ImportError as exc:
        raise AssertionError(f"not built: sleeve_fund.research.fills_vs_model ({exc})") from None
    fn = getattr(fills_vs_model, "compare", None)
    if fn is None:
        raise AssertionError("not built: fills_vs_model.compare(paper_fills, model_fills, store=..., venue=..., "
                             "instrument=...)")
    try:
        rep = fn(paper, model, store=store, venue=VENUE, instrument=PAIR)
    except TypeError as exc:
        raise AssertionError(f"not built: fills_vs_model.compare as assumed: {exc}") from exc
    buckets = getattr(rep, "buckets", None)
    if buckets is None and isinstance(rep, dict):
        buckets = rep.get("buckets")
    if buckets is None:
        raise AssertionError(f"not built: a fills-vs-model report with labelled buckets; got {rep!r}")
    return buckets


@SPREAD_PIT
def test_spread_pit_fills_vs_model_reports_the_daily_refresh_lag_in_its_own_labelled_bucket():
    """Advisor 00:19 (9): fills-vs-model counts the daily-refresh lag as a known divergence in its own labelled
    bucket, never absorbed into a tolerance. HIGH (0.3%) lands at 00:10 (T2); paper last refreshed at 00:00 and
    books a stop at 00:21 with LOW (0.1%), while the model books it with HIGH, the value in force at that minute: a
    difference of (HIGH - LOW) x level x qty = 11.88. Exactly one bucket, labelled for the refresh lag, holds it
    (count 1, amount 11.88), and no other bucket (tolerance, slippage, unexplained) counts it."""
    t2 = _ts(10)
    st = _store([(LOW, T1), (HIGH, t2)])
    from sleeve_fund import spreads

    level, qty, ts = 59_400.0, 0.1, _ts(21)
    assert spreads.resolve(VENUE, PAIR, st).half_spread == HIGH and t2 < ts < t2 + timedelta(hours=24), (
        "set-up: HIGH landed before the stop, inside the 24 hours a daily refresh may lag")

    def fill(hs, **extra):
        px = round(level * (1 - hs), 2)
        return {"ts": ts, "order_id": "O-stop", "intent": "stop_loss", "side": "SELL", "qty": qty, "price": px,
                "fee": round(0.0005 * qty * px, 2), **extra}

    paper, model = [fill(LOW, half_spread=LOW)], [fill(HIGH)]
    gap = (paper[0]["price"] - model[0]["price"]) * qty
    assert abs(gap - (HIGH - LOW) * level * qty) < 0.011, ("set-up: paper and model differ by the lag only", gap)
    buckets = _fills_vs_model(paper, model, st)
    lag = {k: v for k, v in buckets.items() if "refresh" in str(k).lower()}
    others = {k: v for k, v in buckets.items() if k not in lag and (v or {}).get("count")}
    got = {"lag": lag, "others": others}
    want = {"lag": {"<a label naming the refresh lag>": {"count": 1, "amount": round(gap, 2)}}, "others": {}}
    assert (len(lag) == 1 and next(iter(lag.values())).get("count") == 1
            and abs(abs(next(iter(lag.values())).get("amount", 0.0)) - gap) < 0.011 and not others), (got, want)
