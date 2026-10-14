"""F-HALTREF-2 (HoE 14 Oct): a Strategy the supervisor flattens because the portfolio halted says so. Its status reads
"flattened by the portfolio halt: <why>", not "flattened by PM"; a PM flatten still reads "flattened by PM". The order
intent stays pm_flatten either way: only the words change."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from sleeve_fund import supervisor
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.store import HALT_FLATTEN, Store, flattened_words, reset_holds_t

WHY = "the book is 15.2% below its reference 30,000.00, at the 15% portfolio drawdown limit"
T = datetime(2026, 10, 14, 12, tzinfo=UTC)
HELD = {"equity": 10_000, "cash": 7_000, "qty": 0.05, "price": 60_000}


@pytest.fixture
def store(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/h.db")
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="balanced")
    return store


def _runtime(store, t):
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.008)
    return rt


def test_the_words_name_who_asked():
    assert flattened_words(f"{HALT_FLATTEN}{WHY}") == f"flattened by the portfolio halt: {WHY}"
    assert flattened_words("Book kill switch: drawdown") == "flattened by PM: Book kill switch: drawdown"
    assert not flattened_words(f"{HALT_FLATTEN}{WHY}").endswith(".")  # the banner adds its own full stop (FE)


def test_the_supervisor_flattens_with_the_halt_reason(store, monkeypatch):
    monkeypatch.setattr(Store, "holds_any", lambda self, name: True)
    supervisor.Supervisor(store)._halt_every_strategy(WHY, T)
    assert [c["reason"] for c in store.pending_commands("s1")] == [f"{HALT_FLATTEN}{WHY}"]


@pytest.mark.parametrize("reason, words", [
    (f"{HALT_FLATTEN}{WHY}", f"flattened by the portfolio halt: {WHY}"),
    ("Book kill switch: drawdown", "flattened by PM: Book kill switch: drawdown"),
], ids=["halt", "pm"])
def test_the_status_and_the_sells_reason_say_who_flattened_it(store, reason, words):
    t = [T]
    rt = _runtime(store, t)
    store.command("s1", "flatten", reason, actor="supervisor" if reason.startswith(HALT_FLATTEN) else "PM")
    assert rt.tick(**HELD) == "flatten"
    s = store.sleeve("s1")
    assert s.status == "paused" and s.status_reason == words
    assert rt.flatten_why == ("pm_flatten", words[0].upper() + words[1:])  # the intent is unchanged


@pytest.mark.parametrize("reason, words", [
    (f"{HALT_FLATTEN}{WHY}", f"flattened by the portfolio halt: {WHY}"),
    ("Book kill switch: drawdown", "flattened by PM: Book kill switch: drawdown"),
], ids=["halt", "pm"])
def test_a_flatten_pending_at_a_reset_keeps_its_words(tmp_path, reason, words):
    """P1-KR-2's hold: a flatten not yet applied when the reset is asked for is kept as the pause it would have set."""
    store = Store(f"sqlite:///{tmp_path}/r.db")
    store.create_sleeve(name="bn-held", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp"}, venue="binance")
    store.set_status("bn-held", "running")
    store.command("bn-held", "flatten", reason, actor="supervisor")
    store.request_reset("bn-held", "Test finished")
    supervisor.Supervisor(store, python="true").reset_pending()
    s = store.sleeve("bn-held")
    assert s.status == "paused" and s.status_reason.startswith(words)


@pytest.mark.parametrize("reason, words", [
    (f"{HALT_FLATTEN}{WHY}", f"flattened by the portfolio halt: {WHY}"),
    ("Book kill switch: drawdown", "flattened by PM: Book kill switch: drawdown"),
], ids=["halt", "pm"])
def test_a_flatten_asked_for_during_a_reset_keeps_its_words(tmp_path, reason, words):
    """Store._hold_on_reset: a flatten asked for while a reset is under way is recorded as the hold the fresh run
    starts with."""
    store = Store(f"sqlite:///{tmp_path}/d.db")
    store.create_sleeve(name="bn-mid", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp"}, venue="binance")
    store.request_reset("bn-mid", "Test finished")  # under way until the supervisor completes it
    store.command("bn-mid", "flatten", reason, actor="supervisor")
    with store.engine.connect() as c:
        held = [r.status_reason for r in c.execute(select(reset_holds_t.c.status_reason))]
    assert held == [words]


@pytest.mark.parametrize("reason, said, restart", [
    (f"{HALT_FLATTEN}{WHY}", f"Flattened by the portfolio halt, sent again after a restart: {WHY}",
     "restarted while paused by the portfolio halt; stays paused until resumed"),
    ("PM flatten", "Flattened by PM, sent again after a restart: PM flatten",
     "restarted while paused by the PM; stays paused until resumed"),
], ids=["halt", "pm"])
def test_a_flatten_cut_short_by_a_restart_still_says_who_asked(store, reason, said, restart):
    t = [T]
    rt = _runtime(store, t)
    store.command("s1", "flatten", reason)
    assert rt.tick(**HELD) == "flatten"
    t[0] += timedelta(minutes=1)
    rt = _runtime(store, t)  # the sell never filled
    assert store.last_event("s1", ("restart",))["message"] == restart
    assert rt.tick(**HELD) == "flatten" and rt.flatten_why == ("pm_flatten", said)
