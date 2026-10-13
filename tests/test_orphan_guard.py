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
