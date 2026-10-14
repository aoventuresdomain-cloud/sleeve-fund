"""Bridge Drop A, the fund-held book (PE2): the supervisor's poll applies the fund's flows in journal order (CAP5-RESET:
an inflow never overtakes a pending outflow; a re-base only onto an empty book), the book is cash put in + every
Strategy's P&L from 0 + positions, and every poll checks that identity (Advisor RECON-B, MUST). SQLite, and Postgres
with TEST_DATABASE_URL."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

from test_halt_ref import T0, _make, _mark, _settle, _state, _trade, store  # noqa: F401

from sleeve_fund import supervisor

MIN = timedelta(minutes=1)


def _flow(store, kind, amount, at, rebase=False):
    return store.fund_flow(kind, amount, actor="pm", reason="QA", rebase=rebase, now=at)


def _ended(store) -> list:
    return [(f["kind"], f["applied_at"] is not None, f["refused_at"] is not None) for f in store.fund_flows()]


def _rebased(store, amount=100_000, at=T0):
    _flow(store, "deposit", amount, at, rebase=True)
    return _mark(store, at)


def test_a_rebase_onto_an_empty_book_starts_it_at_exactly_the_deposit(store):  # noqa: F811
    st = _rebased(store)
    assert _ended(store) == [("deposit", True, False)]
    assert (st.equity, st.hwm, st.reference(), st.day_start_equity) == (D(100_000),) * 4
    assert not st.halted and st.carried_hwm_drawdown is None
    assert supervisor.fund_book(store).total == D(100_000) == supervisor.fund_equity(store)
    assert _mark(store, T0 + MIN).equity == D(100_000)


def test_a_rebase_onto_a_book_that_isnt_empty_is_refused_naming_whats_left(store):  # noqa: F811
    _make(store, "a", 10_000)
    _settle(store, "a", T0)
    _mark(store, T0)
    _flow(store, "deposit", 100_000, T0 + MIN, rebase=True)
    _mark(store, T0 + MIN)
    (f,) = store.fund_flows()
    assert f["refused_at"] is not None and "a 10,000.00" in f["refuse_reason"]
    assert supervisor.bridge_start(store) is None


def test_a_plain_flow_before_the_reset_is_refused(store):  # noqa: F811
    _flow(store, "deposit", 5_000, T0)
    _mark(store, T0)
    assert _ended(store) == [("deposit", False, True)]


def test_a_strategy_made_after_the_reset_books_from_0_and_moves_no_capital(store):  # noqa: F811
    _rebased(store)
    _make(store, "x", 10_000)  # its starting balance is not the fund's capital any more
    assert not store.joining("x") and store.fund_capital() == D(100_000)
    _trade(store, "x", 250.37, T0 + MIN)
    book = supervisor.fund_book(store)
    assert (book.capital, book.pnl, book.positions) == (D(100_000), D("250.37"), D(0))
    assert _mark(store, T0 + 2 * MIN).equity == D("100250.37")


def test_a_withdrawal_within_free_cash_settles_and_moves_the_book_without_a_loss(store):  # noqa: F811
    _rebased(store)
    _flow(store, "withdraw", 30_000, T0 + MIN)
    st = _mark(store, T0 + MIN)
    assert _ended(store)[-1] == ("withdraw", True, False)
    assert st.equity == st.hwm == D(70_000) and store.fund_capital() == D(70_000)


def test_an_inflow_never_overtakes_a_pending_outflow(store):  # noqa: F811
    _rebased(store)
    _flow(store, "withdraw", 150_000, T0 + MIN)  # more than the free cash: waits
    _flow(store, "deposit", 50_000, T0 + MIN)
    _mark(store, T0 + MIN)
    assert _ended(store)[1:] == [("withdraw", False, False), ("deposit", False, False)]
    assert store.fund_capital() == D(100_000)
    waits = [e["message"] for e in store.events(None, limit=50) if e["kind"] == "fund_flow_waits"]
    assert len(waits) == 1 and "every flow after it" in waits[0]


def test_a_broken_identity_marks_nothing_and_alerts_once(store, monkeypatch):  # noqa: F811
    _rebased(store)
    _make(store, "x", 10_000)
    _trade(store, "x", 100, T0 + MIN)
    _mark(store, T0 + MIN)
    real = store.strategy_cash_pnl
    monkeypatch.setattr(store, "strategy_cash_pnl", lambda name, since=None, at=None: real(name, since, at) + D("0.02"))
    sup = supervisor.Supervisor(store)
    for i in (2, 3):
        sup.mark_portfolio(T0 + i * MIN)
        assert _state(store).mark_ts == T0 + MIN  # not marked: entries stop once the mark is 60 s old
    alerts = [e["message"] for e in store.events(None, limit=50) if e["kind"] == "book_identity_broken"]
    assert len(alerts) == 1 and "0.02" in alerts[0]


def test_empty_book_pockets_names_a_strategy_still_counted_and_none_when_empty(store):  # noqa: F811
    assert supervisor.empty_book_pockets(store) == []
    _make(store, "a", 5_000)
    _settle(store, "a", T0)
    assert any(p.startswith("a 5,000.00") for p in supervisor.empty_book_pockets(store))
