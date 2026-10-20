"""B-20 (supervisor rows): inside a fund-held book a Strategy has no capital of its own, so the reset's decision and
event and the deploy log's book line must not state a per-Strategy balance ("started afresh at 0"). A journal from
before the bridge keeps its wording. SQLite, and Postgres with TEST_DATABASE_URL."""

from __future__ import annotations

from test_halt_ref import T0, _mark, store  # noqa: F401

from sleeve_fund import supervisor
from sleeve_fund.supervisor import Supervisor, book_line


def _held(store):  # noqa: F811
    store.fund_flow("deposit", 100_000, actor="pm", reason="QA", rebase=True, now=T0)
    _mark(store, T0)
    assert supervisor.bridge_start(store) is not None


def _strategy(store, name="timed-test"):  # noqa: F811
    store.create_sleeve(name=name, strategy="candle_step", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, desired_state="stopped", venue="binance")


def test_the_deploy_log_book_line_states_no_strategy_balance_in_a_fund_held_book(store):  # noqa: F811
    _held(store)
    _strategy(store)
    line = book_line(store)
    assert line == "the fund holds the cash: timed-test (stopped, no history)"
    assert " 0 (" not in line


def test_a_reset_in_a_fund_held_book_does_not_say_it_started_afresh_at_0(store):  # noqa: F811
    _held(store)
    _strategy(store)
    store.request_reset("timed-test", "Test finished")
    Supervisor(store, python="true").reset_pending()
    assert store.pending_reset("timed-test") is None
    decided = store.decisions("timed-test")[0]["reason"]
    said = next(e["message"] for e in store.events("timed-test") if e["kind"] == "reset")
    for text in (decided, said):
        assert "at 0" not in text and "the fund's cash, with no capital of its own" in text
