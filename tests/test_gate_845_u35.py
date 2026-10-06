"""QA repros on 845df4d (stop-safety + CHOKE + GAP-LIQ), check (c) U35 with REAL engine processes (recorded sessions
replayed through the paper runtime; tests/test_choke.py uses FakePopen, so the engine's own incident and safety stop
never run there). Strict xfails (raises=AssertionError) where 845df4d fails: P1-SG4, P1-SG5. Plain tests are the
regression guards that hold on 845df4d.

P1-SG5 per the Independent Quant Advisor, 6 Oct 23:05 (MAJOR): status governs ENTRIES only. Every strategy holding a
position (running, paused, exits-only, winding down, PM-paused) keeps the daily-loss and drawdown checks (flattening
per the profile: on every profile in risk.PROFILES both flatten), the liquidation check and its stops.

Set-up: tests/test_restart_x_guards_on.py's 2x short (ping_pong, balanced, whole equity as margin), opened in session 1
(open-risk limit lifted for that, as the guards-on twins do), then process restarts ("deploys") before it is flat:
- "stopped_holder": stopless; the PM stopped it while it held (U35: status paused "exits only: ...", desired stopped),
  so it runs for its exits only, to its safety stop;
- "pm_paused": a 20% stop (inside half the ~49% to liquidation at 2x, so stop safety lets it trade); the PM paused it
  ("paused by PM: ..."), desired running;
- "refused_stopless" (SG4 only): stopless, refused by stop safety (stopless above 1x) while it holds;
- "running" (controls): the same position with nothing blocking it.
Run with PYTHONPATH=<checkout>:<checkout>/tests (imports test_restart_x_guards_on, test_degraded_155_qa); the
conftest's `no_open_risk_limit` mark lifts the limit (tests/ must be the rootdir's conftest, as in a tests/ copy).
"""
import re
from datetime import datetime, timedelta, timezone

import pytest

import test_replay
from sleeve_fund.research.replay import replay
from sleeve_fund.store import Store
from test_degraded_155_qa import _record_session, _session_meta
from test_restart_x_guards_on import NAME, OPENS_SHORT, PARAMS, whole_equity  # noqa: F401

LIFT = pytest.mark.no_open_risk_limit(reason="guards off: the 2x short must open in session 1 to be carried")
ENTRY = 60_900.0  # the price after session 1's 1.5% rise (ping_pong shorts at about 60,628)


def xf(reason):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


SG4 = xf("P1-SG4 (U35, HoE 22:14): a second deploy before flat writes no second incident and keeps the tighter stop")
SG5 = xf("P1-SG5 (Advisor 23:05, MAJOR): status governs entries only; a holder whose status blocks entries keeps the "
         "daily-loss and drawdown flatten")


# A head without the gate branch (main 9ae1f8a): no U35 exits-only holder, no GAP-LIQ. Pins that hold on 845df4d but not
# there are strict xfails on that condition only.
PRE_GATE = not hasattr(__import__("sleeve_fund.paper.runtime", fromlist=["SleeveRuntime"]).SleeveRuntime, "entry_blocked")
ON_PRE_GATE = pytest.mark.xfail(PRE_GATE, strict=True, raises=AssertionError,
                                reason="U35 / GAP-LIQ (the gate branch) not on this head")


def _built(module: str, name: str):
    import importlib
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        raise AssertionError(f"not built: {module}.{name}") from None


HOLDER_PARAMS = {"stopped_holder": PARAMS, "refused_stopless": PARAMS, "pm_paused": {**PARAMS, "stop_loss": 0.20},
                 "running": {**PARAMS, "stop_loss": 0.20}}


def _deploys(tmp_path, holder, sessions, params=None):
    """Session 1 opens the short; then `holder`'s status is set as the PM's action leaves it, and each of `sessions`
    ((start price, legs)) is a process start two hours after the one before, on the same journal."""
    params = params or HOLDER_PARAMS[holder]
    store = Store(f"sqlite:///{tmp_path}/t.db")
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        _record_session(s1, _session_meta(10_000, params), OPENS_SHORT)
        replay(s1, store=store)
        book = store.journal_book(NAME, 10_000)
        assert book["qty"] < 0, f"setup: session 1 opened no short: {book}"
        if holder == "stopped_holder":
            exits_only = _built("sleeve_fund.strategies.base", "EXITS_ONLY")
            stopped_holding = _built("sleeve_fund.supervisor", "STOPPED_HOLDING")
            store.set_desired_state(NAME, "stopped")
            store.set_status(NAME, "paused", f"{exits_only}: {stopped_holding}")
        elif holder == "pm_paused":
            store.set_status(NAME, "paused", "paused by PM: QA, hold it")  # SleeveRuntime's own words for a PM pause
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        for i, (px, legs) in enumerate(sessions, 1):
            test_replay.START = start0 + i * 2 * 3600 * 10**9
            p = tmp_path / f"s{i + 1}.jsonl.gz"
            _record_session(p, _session_meta(book["cash"] + book["qty"] * book["entry_px"], params), legs, px=px)
            replay(p, store=store)
    finally:
        test_replay.START = start0
    return store


QUIET = [(5, 0.0), (5, 0.002)]


def _two_deploys(tmp_path, why, second_move):
    return _deploys(tmp_path, why, [(ENTRY, QUIET), (ENTRY * (1.0 + second_move), QUIET)])


def _orders(store):
    return store.orders(NAME, limit=10_000)


def _kinds(store):
    return [e["kind"] for e in store.events(NAME, limit=10_000)]


# ---------------------------------------------------------------------------------------------------------------
# P1-SG4
# ---------------------------------------------------------------------------------------------------------------

@LIFT
# PE2: SG4 passes on this head (mark removed)
@pytest.mark.parametrize("why", ["stopped_holder", "refused_stopless"])
def test_sg4_a_second_deploy_before_flat_writes_no_second_incident(tmp_path, whole_equity, why):  # noqa: F811
    """U35 (HoE 22:14 checklist): a second deploy before the holder is flat raises no second incident. 845df4d: each
    process start writes the engine's own "started for its exits only ... safety stop" incident again (2 after 2
    deploys; the supervisor's own U35 incident is de-duplicated, the engine's is not)."""
    store = _two_deploys(tmp_path, why, 0.0)
    assert store.journal_book(NAME, 10_000)["qty"] < 0, "setup: still holding"
    incidents = [e["message"] for e in store.events(NAME, limit=1000) if e["kind"] == "incident"]
    assert len(incidents) == 1, incidents


@LIFT
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="PE2: set-up only: with SG5 the 8% move flattens the "
                   "holder at the daily-loss limit first, so nothing is held to deploy again (QA to retune the move); "
                   "the stop never loosening is pinned in test_restart_safety_stop")
def test_sg4_a_second_deploy_after_an_adverse_move_keeps_the_tighter_safety_stop(tmp_path, whole_equity):  # noqa: F811
    """U35 / R-S3: "an already resting tighter stop is kept, not doubled". The stopped holder's first safety stop
    (half way from the 60,900 mark to liquidation: ~75,695) must not be re-measured looser after the price moved 8%
    further against it before the second deploy. 845df4d: the second deploy sets ~78,131 (looser), with a new
    incident."""
    store = _two_deploys(tmp_path, "stopped_holder", 0.08)
    assert store.journal_book(NAME, 10_000)["qty"] < 0, "setup: still holding"
    incidents = [e for e in store.events(NAME, limit=1000) if e["kind"] == "incident"]
    levels = [float(m.replace(",", "")) for e in sorted(incidents, key=lambda e: e["id"])
              for m in re.findall(r"safety stop at ([\d,.]+)", e["message"])]
    assert levels and all(x <= levels[0] + 1e-6 for x in levels), levels  # a short's stop never moves up (looser)


# ---------------------------------------------------------------------------------------------------------------
# P1-SG5 (Advisor 23:05): the risk limits, the liquidation check and the stops run whatever the status
# ---------------------------------------------------------------------------------------------------------------

# Session 2 is quiet at 1.5% against (about 3% of equity: under every limit); session 3 starts against the short by:
# - daily_loss: 8% (about 17% of equity lost in the day: over balanced's 5% daily loss, under its 20% drawdown);
# - drawdown: 13% (about 27%: over balanced's 20% drawdown, which outranks the daily loss).
THRESHOLDS = {"daily_loss": (1.08, "risk_pause"), "drawdown": (1.13, "risk_halt")}


def _limit_run(tmp_path, holder, threshold):
    move, _ = THRESHOLDS[threshold]
    return _deploys(tmp_path, holder, [(ENTRY, QUIET), (ENTRY * move, QUIET)])


def _assert_flattened_by(store, kind, holder, threshold):
    held = store.journal_book(NAME, 10_000)["qty"]
    kinds = _kinds(store)
    closing = [o for o in _orders(store) if o["intent"] == kind and o["side"] == "BUY" and o["filled_qty"] > 0]
    assert kind in kinds and closing and abs(held) < 1e-12, (
        f"{holder} at the {threshold} limit: still holding {held}; journal kinds {sorted(set(kinds))}; "
        f"status {store.sleeve(NAME).status!r} {store.sleeve(NAME).status_reason[:80]!r}")


@LIFT
@pytest.mark.parametrize("threshold", list(THRESHOLDS))
def test_sg5_control_a_running_holder_is_flattened_at_the_risk_limits(tmp_path, whole_equity, threshold):  # noqa: F811
    """Control (holds on 845df4d): the same moves flatten the same position when nothing blocks it, through the limit's
    own flatten (a BUY of intent risk_pause / risk_halt) with its event journaled."""
    store = _limit_run(tmp_path, "running", threshold)
    _assert_flattened_by(store, THRESHOLDS[threshold][1], "running", threshold)


@LIFT
# PE2: SG5 passes on this head (mark removed)
@pytest.mark.parametrize("threshold", list(THRESHOLDS))
@pytest.mark.parametrize("holder", ["stopped_holder", "pm_paused"])
def test_sg5_a_holder_whose_status_blocks_entries_is_still_flattened_at_the_risk_limits(tmp_path, whole_equity,  # noqa: F811
                                                                                       holder, threshold):
    """Advisor 23:05: a Stopped (exits-only) 2x holder and a PM-paused holder each hit the daily-loss and the drawdown
    thresholds and are flattened (balanced: both flatten), with the event journaled (risk_pause / risk_halt) and the
    flatten order of that intent filled. 845df4d: SleeveRuntime.tick runs risk.check only while status == "running",
    so both still hold the whole short (the U35 one to its safety stop, the PM-paused one to its 20% stop)."""
    store = _limit_run(tmp_path, holder, threshold)
    _assert_flattened_by(store, THRESHOLDS[threshold][1], holder, threshold)


@LIFT
@pytest.mark.parametrize("holder", [pytest.param("stopped_holder", marks=ON_PRE_GATE),
                                    pytest.param("pm_paused", marks=ON_PRE_GATE)])  # main: a stop_loss, no GAP-LIQ
def test_sg5_a_paused_holder_still_gets_the_liquidation_check(tmp_path, whole_equity, holder):  # noqa: F811
    """Advisor 23:05: the liquidation check runs whatever the status. The price opens session 3 60% against the 2x
    short, past its safety (or 20%) stop and its ~90,490 liquidation price: the venue's liquidation is booked (a
    liquidation order fills the short, a "liquidation" event), the strategy is halted as liquidated and is flat."""
    store = _deploys(tmp_path, holder, [(ENTRY, QUIET), (ENTRY * 1.6, QUIET)])
    liq = [o for o in _orders(store) if o["intent"] == "liquidation" and o["filled_qty"] > 0]
    kinds = _kinds(store)
    assert liq and "liquidation" in kinds, (f"{holder}: no liquidation booked; kinds {sorted(set(kinds))}; "
                                            f"{[(o['intent'], o['side'], o['status']) for o in _orders(store)]}")
    s = store.sleeve(NAME)
    assert s.status == "halted" and abs(store.journal_book(NAME, 10_000)["qty"]) < 1e-12, (s.status, s.status_reason)


@LIFT
@pytest.mark.parametrize("holder", [pytest.param("stopped_holder", marks=ON_PRE_GATE), "pm_paused"])
def test_sg5_a_paused_holders_stop_still_closes_it(tmp_path, whole_equity, holder):  # noqa: F811
    """Advisor 23:05: its stops run whatever the status. The same short with a 2% stop (4% of equity at 2x: under
    every limit, so only the stop can act), held while stopped (exits only) or PM-paused; session 2 rises 3% over five
    minutes from 60,900: the restored stop (entry +2%, ~61,843) closes it with a stop_loss order, and nothing else
    trades."""
    store = _deploys(tmp_path, holder, [(ENTRY, [(5, 0.0), (5, 0.03), (5, 0.0)])],
                     params={**PARAMS, "stop_loss": 0.02})
    s2 = datetime.fromtimestamp(test_replay.START / 1e9, timezone.utc) + timedelta(hours=2)
    after = [o for o in _orders(store) if o["ts"] >= s2]  # session 2's orders (session 1 traded before it)
    stop = [o for o in after if o["intent"] == "stop_loss" and o["side"] == "BUY" and o["filled_qty"] > 0]
    assert stop and abs(store.journal_book(NAME, 10_000)["qty"]) < 1e-12, (
        f"{holder}: its stop didn't close it: {[(o['ts'], o['intent'], o['side'], o['status']) for o in after]}")
    assert all(o["intent"] == "stop_loss" for o in after if o["filled_qty"] > 0), [(o["ts"], o["intent"]) for o in after]
