"""ORPHAN-GUARD's pure parts (paper.orphan_guard); HoQA's cells are test_orphan_guard_qa."""
from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund.paper.orphan_guard import is_overdue, overdue_at, resting_entry_order

CLOSE, BAR = datetime(2026, 10, 8, 9, 0, tzinfo=UTC), timedelta(hours=1)


def test_og4_a_resting_entry_is_overdue_only_after_its_next_close_plus_two_minutes():
    due = overdue_at(CLOSE, BAR)
    assert due == CLOSE + BAR + timedelta(minutes=2)
    assert not is_overdue(CLOSE, BAR, due) and is_overdue(CLOSE, BAR, due + timedelta(microseconds=1))


def test_og1_a_bar_of_no_length_is_refused():
    with pytest.raises(ValueError):
        resting_entry_order(gtd_supported=True, decision_close=CLOSE, bar=timedelta(0))


def test_og1_a_short_bars_expiry_is_clamped_past_the_venues_minimum_lead():
    """Advisor 13 Oct: expiry = max(the next close, now + the venue's minimum lead + 30 s)."""
    lead = timedelta(minutes=10)
    now = CLOSE + timedelta(seconds=2)
    _, one_minute, _ = resting_entry_order(gtd_supported=True, decision_close=CLOSE, bar=timedelta(minutes=1),
                                           now=now, min_lead=lead)
    assert one_minute == now + lead + timedelta(seconds=30)
    _, four_hours, _ = resting_entry_order(gtd_supported=True, decision_close=CLOSE, bar=timedelta(hours=4),
                                           now=now, min_lead=lead)
    assert four_hours == CLOSE + timedelta(hours=4)


@pytest.mark.parametrize("venue,minutes,expect", [
    ("BINANCE", 60, "close"),  # an hour's bar (1-HOUR): good till its next close
    ("BINANCE", 1, "lead"),  # a minute's bar: clamped past the venue's 600 s minimum lead
    ("NOWHERE", 60, "gtc"),  # no venue profile: good-till-cancelled, with the backstop note
])
def test_og1_the_engines_resting_entry_takes_its_venues_terms(venue, minutes, expect):
    from types import SimpleNamespace

    from nautilus_trader.model import InstrumentId

    from sleeve_fund.strategies.base import LongFlatStrategy
    from sleeve_fund.venues import BINANCE_GTD_MIN_LEAD

    now = CLOSE + timedelta(seconds=3)
    fake = SimpleNamespace(
        _cfg=SimpleNamespace(instrument_id=InstrumentId.from_str(f"BTCUSDT-PERP.{venue}"),
                             bar_type=f"BTCUSDT-PERP.{venue}-" + ("1-HOUR" if minutes == 60 else f"{minutes}-MINUTE") + "-LAST-INTERNAL"),
        _gtd_refused=None, _deciding=(int(CLOSE.timestamp() * 1e9), 0), clock=SimpleNamespace(timestamp_ns=lambda: int(now.timestamp() * 1e9)))
    tif, expires, note = LongFlatStrategy._resting_entry_terms(fake)
    if expect == "gtc":
        assert tif.name == "GTC" and expires is None and "backstop" in note
    else:
        assert tif.name == "GTD" and note is None
        assert expires == (CLOSE + timedelta(minutes=minutes) if expect == "close" else
                           now + BINANCE_GTD_MIN_LEAD + timedelta(seconds=30))


@pytest.mark.parametrize("gated", [True, False])
def test_og2_the_supervisor_marks_a_crashed_or_stopped_owner_down_only_where_the_gate_is_in_force(tmp_path, gated):
    """HoE 13 Oct (the pair): a crash seen by the supervisor, or its own stop, hands the strategy's resting entries to
    the supervisor's orphan pass (owner_down); a journal without the portfolio gate has no gated entries to hand."""
    from sleeve_fund import supervisor as sup
    from sleeve_fund.paper.orphan_guard import down_since
    from sleeve_fund.store import Store

    class Dead:
        pid, returncode = 4242, 1
        poll = lambda self: 1  # noqa: E731

    class Alive(Dead):
        returncode = None
        poll = lambda self: None  # noqa: E731
        send_signal = wait = kill = lambda self, *a, **k: 0  # noqa: E731

    store = Store(f"sqlite:///{tmp_path}/sv.db", portfolio_gate=gated)
    for name in ("crashes", "stops"):
        store.create_sleeve(name=name, strategy="trend_filter", instrument="BTC/USD",
                            bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=1000)
    sv = sup.Supervisor(store)
    sv.procs["crashes"] = sup.Proc(popen=Dead())
    sv.step()
    sv._stop("stops", sup.Proc(popen=Alive()), "stopped by PM")
    sv._stop("stops", sup.Proc(), "stopped by PM")  # nothing running: no second hand-over
    for name in ("crashes", "stops"):
        assert (down_since(store, name) is not None) is gated, name
    downs = [e for e in store.events("stops", limit=100) if e["kind"] == "owner_down"]
    assert len(downs) == (1 if gated else 0)


@pytest.mark.parametrize("reason, refused", [
    ("Order would immediately expire: GTD time too early", True), ("goodTillDate must be at least 600s ahead", True),
    ("Post only order will be rejected", False), ("insufficient margin", False)])
def test_og1_a_venue_refusing_the_expiry_is_told_apart_from_other_refusals(reason, refused):
    from types import SimpleNamespace

    from nautilus_trader.model import TimeInForce

    from sleeve_fund.strategies.base import LongFlatStrategy

    gtd, gtc = SimpleNamespace(time_in_force=TimeInForce.GTD), SimpleNamespace(time_in_force=TimeInForce.GTC)
    assert LongFlatStrategy._gtd_refusal(gtd, SimpleNamespace(reason=reason)) is refused
    assert LongFlatStrategy._gtd_refusal(gtc, SimpleNamespace(reason=reason)) is False  # only a good-till-date one
    assert LongFlatStrategy._gtd_refusal(None, SimpleNamespace(reason=reason)) is False


def test_og1_after_a_refused_expiry_the_entries_rest_good_till_cancelled_and_say_why():
    """Advisor 13 Oct: never refuse resting entries for the expiry; the next goes good-till-cancelled with the
    backstop note and the venue's reason."""
    from types import SimpleNamespace

    from nautilus_trader.model import InstrumentId

    from sleeve_fund.strategies.base import LongFlatStrategy

    fake = SimpleNamespace(_cfg=SimpleNamespace(instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                                                bar_type="BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-INTERNAL"),
                           _gtd_refused="GTD time too early", _deciding=(int(CLOSE.timestamp() * 1e9), 0),
                           clock=SimpleNamespace(timestamp_ns=lambda: int(CLOSE.timestamp() * 1e9)))
    tif, expires, note = LongFlatStrategy._resting_entry_terms(fake)
    assert tif.name == "GTC" and expires is None and "backstop" in note and "GTD time too early" in note


@pytest.fixture
def jstore():
    from test_gate_ledger import _store, _strategies

    s = _store()
    _strategies(s)
    return s


def _reason(store, rid):
    from sqlalchemy import select

    from sleeve_fund.store import gate_reservations_t

    with store.engine.connect() as c:
        return c.execute(select(gate_reservations_t.c.release_reason)
                         .where(gate_reservations_t.c.decision_id == rid)).scalar()


@pytest.mark.parametrize("filled_after_down", [False, True])
def test_og3_a_sandbox_restart_keeps_what_filled_before_the_crash_and_releases_only_the_rest(
        jstore, filled_after_down):
    """Advisor 13 Oct 17:30 UK, pin 2: part filled before its process died, the entry keeps that fill (its bar not yet
    closed); the restart ends it with its sandbox and releases only the unfilled remainder as 'cancel'. Filled after
    the owner went down instead (a fill the dead sandbox journalled on its way out), it is closed (og5)."""
    from sleeve_fund.paper import orphan_guard as og

    rid = og.qa_resting_entry(jstore, "a", "o-rest", CLOSE)
    down = CLOSE + timedelta(minutes=10)
    at = down + (timedelta(seconds=1) if filled_after_down else -timedelta(minutes=1))
    jstore.book_fill("a", side="BUY", qty=0.04, price=60000.0, fee=0.1, order_id="o-rest", trade_id="t1", ts=at)
    og.owner_down(jstore, "a", down, "crash")
    og.supervisor_pass(jstore, down + timedelta(seconds=5))
    og.restart_reconcile(jstore, "a", down + timedelta(minutes=2), venue_open=[], in_process="sandbox")
    assert _reason(jstore, rid) == "cancel" and not jstore.open_cancels("a")
    assert sum(float(f["qty"]) for f in jstore.fills("a")) == pytest.approx(0.04)
    closes = [o for o in jstore.orders("a") if o["order_id"] != "o-rest"]
    assert [(c["side"], float(c["qty"])) for c in closes] == ([("SELL", 0.04)] if filled_after_down else [])
    assert "journal fills 0.04" in jstore.events("a", limit=100)[0]["message"] or any(
        "journal fills 0.04" in e["message"] for e in jstore.events("a", limit=100) if e["kind"] == "sandbox_ended")


def test_og3_a_venue_outside_the_process_stays_blocked_at_restart_until_its_own_lookup_answers(jstore):
    """Advisor 13 Oct 17:30 UK, pin 3 (condition 2): the same crash on a venue that isn't the in-process sandbox
    (testnet, live): the restart's "not found" releases nothing; only the venue's lookup does."""
    from sleeve_fund.paper import orphan_guard as og
    from sleeve_fund.paper.runtime import entry_blocked

    rid = og.qa_resting_entry(jstore, "a", "o-rest", CLOSE)
    down = CLOSE + timedelta(minutes=10)
    og.owner_down(jstore, "a", down, "crash")
    og.supervisor_pass(jstore, down + timedelta(seconds=5))
    now = down + timedelta(minutes=2)
    og.restart_reconcile(jstore, "a", now, venue_open=[], in_process=None)
    for m in range(3, 60, 5):
        og.supervisor_pass(jstore, down + timedelta(minutes=m))
    assert _reason(jstore, rid) is None and [c["order_id"] for c in jstore.open_cancels("a")] == ["o-rest"]
    assert entry_blocked(jstore, "a", down + timedelta(minutes=60))[0]
    og.venue_lookup(jstore, "o-rest", terminal="canceled", fills=[], at=down + timedelta(minutes=61))
    assert _reason(jstore, rid) == "cancel" and not entry_blocked(jstore, "a", down + timedelta(minutes=62))[0]


def test_in_process_kind_is_read_from_the_venues_type_never_a_setting():
    """Advisor 13 Oct 17:30 UK, condition 1: only the paper node's Sandbox client or a replay's BacktestEngine count;
    anything else (a testnet or live client, a config naming a sandbox) is outside the process."""
    from types import SimpleNamespace

    from nautilus_trader.adapters.sandbox import SandboxExecutionClientFactory
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig

    from sleeve_fund.paper.orphan_guard import in_process_kind

    assert in_process_kind(SandboxExecutionClientFactory()) == "sandbox"
    assert in_process_kind(BacktestEngine(BacktestEngineConfig())) == "backtest"
    assert in_process_kind(SimpleNamespace(kind="sandbox")) is None and in_process_kind("sandbox") is None


@pytest.mark.parametrize("lock_free", [True, False])
def test_og3_with_no_owner_down_a_restart_releases_only_once_its_lock_proves_the_old_process_gone(
        jstore, tmp_path, lock_free):
    """Advisor 13 Oct 17:55 UK: the supervisor restarted too, so no owner_down was journalled. A free single-instance
    lock proves the process before this one, and its sandbox, are gone: released. Held (an orphan process a deploy
    left running, whose sandbox could still fill): nothing released, entries blocked, one alert."""
    from sleeve_fund.paper import orphan_guard as og
    from sleeve_fund.paper.runtime import entry_blocked
    from sleeve_fund.paper.single_lock import instance_lock

    rid = og.qa_resting_entry(jstore, "a", "o-rest", CLOSE)
    old = instance_lock("a", tmp_path)  # the process before this one
    if lock_free:
        old.close()  # it is gone: the OS dropped its lock
    mine = instance_lock("a", tmp_path)
    assert (mine is not None) is lock_free
    now = CLOSE + timedelta(minutes=20)
    for s in (0, 30):
        og.restart_reconcile(jstore, "a", now + timedelta(seconds=s), venue_open=[], in_process="sandbox",
                             sole_instance=mine is not None)
    told = [e for e in jstore.events("a", limit=100) if e["kind"] == "previous_process_unconfirmed"]
    if lock_free:
        assert _reason(jstore, rid) == "cancel" and not told
        assert not entry_blocked(jstore, "a", now + timedelta(minutes=1))[0]
    else:
        assert _reason(jstore, rid) is None and len(told) == 1
        assert entry_blocked(jstore, "a", now + timedelta(minutes=1))[0]
    for fh in (old, mine):
        if fh is not None:
            fh.close()


def test_a_refused_duplicates_exit_never_hands_the_running_process_entries_to_the_supervisor(tmp_path):
    from sleeve_fund import supervisor as sup
    from sleeve_fund.paper.orphan_guard import down_since
    from sleeve_fund.paper.single_lock import EXIT_DUPLICATE
    from sleeve_fund.store import Store

    class Dup:
        pid, returncode = 4243, EXIT_DUPLICATE
        poll = lambda self: EXIT_DUPLICATE  # noqa: E731

    store = Store(f"sqlite:///{tmp_path}/sv.db", portfolio_gate=True)
    store.create_sleeve(name="s", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=1000)
    sv = sup.Supervisor(store)
    sv.procs["s"] = sup.Proc(popen=Dup())
    sv.step()
    assert down_since(store, "s") is None


def test_og3_outside_the_process_a_leftover_the_venue_no_longer_lists_blocks_until_its_lookup(jstore):
    """HoQA og3-adv-1b: after a restart on a venue outside the process, a gated entry still open in the journal that
    the venue doesn't list may have filled while it was down: unknown, so it is held as a "not found" and blocks that
    strategy (only) until the venue's lookup settles it."""
    from sleeve_fund.paper import orphan_guard as og
    from sleeve_fund.paper.runtime import entry_blocked

    rid = og.qa_resting_entry(jstore, "a", "o-left", CLOSE)
    now = CLOSE + timedelta(minutes=20)
    og.restart_reconcile(jstore, "a", now, venue_open=[], in_process=None)
    (c,) = jstore.open_cancels("a")
    assert c["order_id"] == "o-left" and c["not_found_at"] is not None
    assert entry_blocked(jstore, "a", now + timedelta(seconds=1))[0]
    assert not entry_blocked(jstore, "b", now + timedelta(seconds=1))[0]
    og.venue_lookup(jstore, "o-left", terminal="canceled", fills=[], at=now + timedelta(minutes=1))
    assert _reason(jstore, rid) == "cancel" and not entry_blocked(jstore, "a", now + timedelta(minutes=2))[0]


def _open_cancel(store, name="s1"):
    """A gated entry left resting for `name`, its cancel asked and answered "not found": unconfirmed."""
    from sleeve_fund.paper import orphan_guard as og

    og.qa_resting_entry(store, name, "o-rest", CLOSE)
    og.owner_down(store, name, CLOSE + timedelta(minutes=10), "crash")
    og.supervisor_pass(store, CLOSE + timedelta(minutes=10, seconds=5))
    og.cancel_confirmed(store, "o-rest", not_found=True)
    assert store.open_cancels(name)


def test_f241_3_a_strategy_reset_is_refused_while_a_cancel_is_open_and_runs_once_the_venue_settles_it(
        tmp_path, monkeypatch):
    """CR F241-3 (HoE MUST): split_run would put the open cancel away with the run and lift the fresh run's entry block
    while its order may still rest. So the reset is refused, saying why; asked again once settled, it runs. Its process
    isn't proved gone (another holds its lock), so nothing settles by itself (CR MUST 2, TEC 15 Oct)."""
    from sleeve_fund import supervisor as sup
    from sleeve_fund.paper import orphan_guard as og
    from sleeve_fund.paper.single_lock import instance_lock
    from sleeve_fund.store import Store

    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))
    held = instance_lock("s1")

    store = Store(f"sqlite:///{tmp_path}/r.db", portfolio_gate=True)
    store.create_sleeve(name="s1", strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True})
    _open_cancel(store)
    store.set_desired_state("s1", "stopped")
    store.request_reset("s1", "Test finished")
    sv = sup.Supervisor(store, python="true")
    sv.reset_pending()
    assert store.pending_reset("s1") is None and not store.reset_runs() and store.open_cancels("s1")
    assert store.last_event("s1", ("reset_refused",))["message"].endswith(sup.OPEN_CANCEL_REFUSAL)
    og.venue_lookup(store, "o-rest", terminal="canceled", fills=[], at=CLOSE + timedelta(hours=3))
    store.request_reset("s1", "Again, now settled")
    sv.reset_pending()
    held.close()
    assert store.pending_reset("s1") is None and len(store.reset_runs()) == 1


def test_f241_4_a_cancel_asked_between_the_check_and_split_run_refuses_the_reset_and_restarts_the_strategy(
        tmp_path, monkeypatch):
    """DA F241-4: its process may ask for a cancel after reset_pending's check and before it is stopped. split_run
    then refuses and changes nothing; the supervisor refuses the reset as F241-3 does, and a running strategy runs
    again, rather than the supervisor's loop raising. Its process isn't proved gone (another holds its lock), so
    nothing settles by itself (CR MUST 2, TEC 15 Oct)."""
    from sleeve_fund import supervisor as sup
    from sleeve_fund.paper.single_lock import instance_lock
    from sleeve_fund.store import Store

    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))
    held = instance_lock("s1")

    store = Store(f"sqlite:///{tmp_path}/r.db", portfolio_gate=True)
    store.create_sleeve(name="s1", strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True})
    store.set_desired_state("s1", "running")
    store.request_reset("s1", "Test finished")
    assert store.pending_reset("s1")["restart"], "set-up: a reset of a running strategy restarts it"
    sv = sup.Supervisor(store, python="true")
    monkeypatch.setattr(sv, "_stop", lambda name, proc, why: _open_cancel(store, name))  # the race, at the stop
    sv.reset_pending()
    held.close()
    assert store.pending_reset("s1") is None and not store.reset_runs() and store.open_cancels("s1")
    assert store.last_event("s1", ("reset_refused",))["message"].endswith(sup.OPEN_CANCEL_REFUSAL)
    assert store.sleeve("s1").desired_state == "running"


def test_f241_3_a_book_reset_or_a_reset_after_liquidation_never_lifts_an_open_cancels_block(tmp_path):
    """The other resets don't move the cancel (no split_run): the book reset is refused while the entry rests, and a
    reset after liquidation leaves the cancel, and the block, as they were."""
    from sleeve_fund.paper.runtime import entry_blocked
    from sleeve_fund.store import BookResetRefused, Store

    store = Store(f"sqlite:///{tmp_path}/b.db", portfolio_gate=True)
    store.create_sleeve(name="s1", strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True})
    _open_cancel(store)
    with pytest.raises(BookResetRefused):
        store.book_reset("PM resets the book")
    store.event("s1", "error", "liquidation", "Liquidated: the price 80,000 gapped through 81,000")
    store.event("s1", "error", "incident", "Liquidation: the half-liquidation stop did not protect the position")
    iid = store.last_event("s1", ("incident",))["id"]
    store.write_incident_note(iid, author="Head of Engineering", why_stop_did_not_protect="QA gap")
    store.command("s1", "reset_after_liquidation", "PM read the incident", actor="PM", incident=iid)
    assert store.open_cancels("s1") and entry_blocked(store, "s1", CLOSE + timedelta(hours=3))[0]


# ---- CR MUST 2 (HoE ruling, CR option 1, 15 Oct): a Stop with a resting entry never wedges the PM's Reset ------------

def _stopped_with_a_resting_entry(tmp_path, monkeypatch):
    """The PM stopped `s1` while a post-only entry rested: _stop journalled owner_down, the next pass asked its cancel,
    and on paper nothing answers it, since the simulated venue lived in the stopped process."""
    from sleeve_fund.paper import orphan_guard as og
    from sleeve_fund.store import Store

    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))
    for var in ("EXEC_MODE", "EXEC_HOST"):
        monkeypatch.delenv(var, raising=False)
    store = Store(f"sqlite:///{tmp_path}/r.db", portfolio_gate=True)
    store.create_sleeve(name="s1", strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True})
    og.qa_resting_entry(store, "s1", "o-rest", CLOSE)
    og.owner_down(store, "s1", CLOSE + timedelta(minutes=10), "stop")
    og.supervisor_pass(store, CLOSE + timedelta(minutes=10, seconds=5))
    assert store.open_cancels("s1"), "set-up: the cancel is asked and open"
    store.set_desired_state("s1", "stopped")
    store.request_reset("s1", "Test finished")
    return store


def test_cr_must2_a_reset_after_a_stop_with_a_resting_entry_goes_through_and_settles_the_cancel_sandbox_ended(
        tmp_path, monkeypatch):
    from sleeve_fund import supervisor as sup
    from sleeve_fund.paper import orphan_guard as og

    store = _stopped_with_a_resting_entry(tmp_path, monkeypatch)
    sup.Supervisor(store, python="true").reset_pending()
    assert store.pending_reset("s1") is None and len(store.reset_runs()) == 1
    (run,) = store.reset_runs()
    assert not store.open_cancels("s1") and not store.open_cancels(run)
    assert "o-rest" in store.last_event(run, (og.SANDBOX_ENDED,))["message"]  # settled, and put away with the run
    assert store.sleeve("s1").desired_state == "stopped"  # the PM's Stop holds: Reset never starts it


def test_cr_must2_with_the_instance_lock_still_held_the_reset_is_still_refused(tmp_path, monkeypatch):
    from sleeve_fund import supervisor as sup
    from sleeve_fund.paper.single_lock import instance_lock

    store = _stopped_with_a_resting_entry(tmp_path, monkeypatch)
    held = instance_lock("s1")  # a process still holds it: nothing proves the old one is gone
    try:
        sup.Supervisor(store, python="true").reset_pending()
    finally:
        held.close()
    assert not store.reset_runs() and store.open_cancels("s1")
    assert store.last_event("s1", ("reset_refused",))["message"].endswith(sup.OPEN_CANCEL_REFUSAL)


@pytest.mark.parametrize("mode", ["demo", "OFF"])
def test_cr_must2_with_the_order_client_on_nothing_settles_by_itself(tmp_path, monkeypatch, mode):
    """A real venue's cancel is the venue's to answer; an unknown mode fails closed the same way (N-KH1)."""
    from sleeve_fund import supervisor as sup

    store = _stopped_with_a_resting_entry(tmp_path, monkeypatch)
    monkeypatch.setenv("EXEC_MODE", mode)
    sup.Supervisor(store, python="true").reset_pending()
    assert not store.reset_runs() and store.open_cancels("s1")
    assert store.last_event("s1", ("reset_refused",))["message"].endswith(sup.OPEN_CANCEL_REFUSAL)


def test_cr_must2_the_f241_4_race_on_a_stopped_paper_process_settles_and_the_reset_runs(tmp_path, monkeypatch):
    """The F241-4 race with the process proved gone (its lock free, the order client off): the cancel asked at the
    stop is settled sandbox_ended and the reset goes through, restarting a running strategy as asked."""
    from sleeve_fund import supervisor as sup
    from sleeve_fund.store import Store

    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("EXEC_MODE", raising=False)
    store = Store(f"sqlite:///{tmp_path}/r.db", portfolio_gate=True)
    store.create_sleeve(name="s1", strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True})
    store.set_desired_state("s1", "running")
    store.request_reset("s1", "Test finished")
    sv = sup.Supervisor(store, python="true")
    monkeypatch.setattr(sv, "_stop", lambda name, proc, why: _open_cancel(store, name))  # the race, at the stop
    sv.reset_pending()
    assert store.pending_reset("s1") is None and len(store.reset_runs()) == 1 and not store.open_cancels("s1")
    assert store.sleeve("s1").desired_state == "running"
