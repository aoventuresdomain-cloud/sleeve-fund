"""A3 (Advisor 05:47; HoE 15 Oct): a Stop's cancel goes out at its acceptance. On Postgres the journal pushes every
decision on a Strategy (Store.decide, NOTIFY on GATE_CHANNEL) as it commits; the Strategy's process listens on a thread
of its own that only sets a flag, which the engine reads every GATE_PUSH_SECONDS and then reads the gate at once. The
gate read every GATE_WATCH_SECONDS stays for a lost push, and its period is the setting raced_fill_window_ms."""

import os
import threading
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from sleeve_fund.store import Store

PG = os.environ.get("TEST_DATABASE_URL")
needs_pg = pytest.mark.skipif(not (PG or "").startswith("postgresql"), reason="Postgres only with TEST_DATABASE_URL")


@pytest.fixture
def pg_store():
    schema = f"a3_{uuid.uuid4().hex[:10]}"
    admin = create_engine(PG)
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {schema}"))
    eng = create_engine(PG, connect_args={"options": f"-csearch_path={schema}"})
    yield Store(engine=eng)
    eng.dispose()
    with admin.begin() as c:
        c.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    admin.dispose()


def _listening(name):
    from sqlalchemy.engine import make_url

    from sleeve_fund.strategies.base import listen_gate

    pushed, done = threading.Event(), threading.Event()
    dsn = make_url(PG).set(drivername="postgresql").render_as_string(hide_password=False)
    threading.Thread(target=listen_gate, args=(dsn, name, pushed, done), daemon=True).start()
    assert pushed.wait(5), "the listener never connected"  # it reads once on connecting
    pushed.clear()
    return pushed, done


@needs_pg
def test_a3a_a_stop_is_pushed_to_its_strategys_process_as_it_commits(pg_store):
    pushed, done = _listening("s1")
    try:
        pg_store.decide("pm", "stop", "stopped by PM", "s1")
        assert pushed.wait(5), "no push for the Stop"  # the path, not wall-clock ms (HoQA): those are journaled
    finally:
        done.set()


@needs_pg
@pytest.mark.parametrize("how", ["halt", "pause", "desired_stopped", "fund_halt"])
def test_a3a2_a_halt_a_pause_a_stop_and_the_funds_own_block_all_push(pg_store, how):
    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.portfolio.gate import PortfolioState

    pg_store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USDT-PERP",
                           bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000)
    led = DbLedger(pg_store, lambda: None)
    led.ensure_state() if how == "fund_halt" else None
    pushed, done = _listening("s1")
    try:
        if how == "halt":
            pg_store.set_status("s1", "halted", "drawdown past the limit")
        elif how == "pause":
            pg_store.set_status("s1", "paused", "daily loss")
        elif how == "desired_stopped":
            pg_store.set_desired_state("s1", "stopped")
        else:
            led.set_state(PortfolioState(halted="fund drawdown past the limit"))
        assert pushed.wait(5), f"no push for {how}"
    finally:
        done.set()


@needs_pg
def test_a3a3_an_asset_legs_process_wakes_for_its_strategy(pg_store):
    pushed, done = _listening("multi~ETH")
    try:
        pg_store.decide("pm", "stop", "stopped by PM", "multi")
        assert pushed.wait(5)
    finally:
        done.set()


@needs_pg
def test_a3b_another_strategys_decision_and_a_fund_decision_push_nothing_to_it(pg_store):
    pushed, done = _listening("s1")
    try:
        pg_store.decide("pm", "stop", "stopped by PM", "s2")
        pg_store.decide("pm", "change_settings", "fund limits", None)
        assert not pushed.wait(1.5)
    finally:
        done.set()


def test_a3c_on_sqlite_a_decision_pushes_nothing_and_is_journaled_as_before():
    store = Store.in_memory()
    store.decide("pm", "stop", "stopped by PM", "s1")
    assert store.decisions("s1", action="stop")[0]["reason"] == "stopped by PM"


def test_a3d_a_push_reads_the_gate_at_once_and_no_push_reads_nothing():
    from sleeve_fund.strategies.base import GATE_PUSH, LongFlatStrategy

    reads = []
    strat = SimpleNamespace(_gate_pushed=threading.Event(), _on_gate_watch=lambda e: reads.append(e),
                            _resting_openers=lambda: ([1], []),
                            clock=SimpleNamespace(timer_names=lambda: [GATE_PUSH], cancel_timer=lambda n: None))
    LongFlatStrategy._on_gate_push(strat, "tick")
    assert reads == []
    strat._gate_pushed.set()
    LongFlatStrategy._on_gate_push(strat, "tick")
    assert reads == ["tick"] and not strat._gate_pushed.is_set()  # read once per push


def test_a3e_the_raced_fill_window_is_a_setting_with_todays_period_by_default():
    from sleeve_fund.strategies import base

    def seconds(ms):
        return base.LongFlatStrategy._gate_watch_seconds(SimpleNamespace(_cfg=SimpleNamespace(raced_fill_window_ms=ms)))

    assert seconds(None) == base.GATE_WATCH_SECONDS == 0.5
    assert seconds(180) == pytest.approx(0.18)


@pytest.mark.parametrize("bad", [0, 9, 5_001, 12.5])
def test_a3f_a_window_outside_10ms_to_5s_is_refused(bad, instrument):
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies.base import LongFlatConfig

    with pytest.raises(ValueError, match="raced_fill_window_ms"):
        LongFlatConfig(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-INTERNAL"), assumed_taker_fee=0.001,
                       raced_fill_window_ms=bad)


def test_a3g_a_push_after_a_reconnect_reads_the_gate_and_cancels_only_what_still_rests():
    """The listener reads once on (re)connecting; a cancel already done is not done again: the gate read cancels only
    the opening orders still resting, and with none it stops watching."""
    from sleeve_fund.strategies.base import GATE_PUSH, GATE_WATCH, LongFlatStrategy

    cancelled, timers = [], {GATE_WATCH, GATE_PUSH}
    strat = SimpleNamespace(_resting_openers=lambda: (), _cancel_resting_entries=lambda: cancelled.append(1),
                            _report=lambda *a: (_ for _ in ()).throw(AssertionError(a)),
                            clock=SimpleNamespace(timer_names=lambda: list(timers), cancel_timer=timers.discard))
    LongFlatStrategy._on_gate_watch(strat, None)
    assert cancelled == [] and timers == set()
