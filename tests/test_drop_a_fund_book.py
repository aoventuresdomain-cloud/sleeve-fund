"""Bridge Drop A, the fund-held book (PE2): the supervisor's poll applies the fund's flows in journal order (CAP5-RESET:
an inflow never overtakes a pending outflow; a re-base only onto an empty book), the book is cash put in + every
Strategy's P&L from 0 + positions, and every poll checks that identity (Advisor RECON-B, MUST). SQLite, and Postgres
with TEST_DATABASE_URL."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

import pytest

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
    _make(store, "x", 0)  # no capital of its own: the fund holds the cash
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


def _entry(store, at):
    """One entry idea through the journal's gate, as paper sends it."""
    from sleeve_fund.gate_ledger import DbLedger, TaggedIntent
    from sleeve_fund.portfolio.gate import check_order

    px = D(60_000)
    it = TaggedIntent("BTC", 1, D("0.001"), px, px, px / 50, D("0.001"), D("0.001"), intent_id=f"i{at:%H%M%S}",
                      kind="entry", bar_ts=at)
    return check_order(DbLedger(store, lambda: (supervisor.fund_book(store).total, ())), "x", it, None, at).decision


def _broken(store, monkeypatch):
    real = store.strategy_cash_pnl
    monkeypatch.setattr(store, "strategy_cash_pnl", lambda name, since=None, at=None: real(name, since, at) + D("0.02"))
    return lambda: monkeypatch.setattr(store, "strategy_cash_pnl", real)


def _alerts(store):
    return [e["message"] for e in store.events(None, limit=100) if e["kind"] == "book_identity_broken"]


def test_a_broken_identity_holds_entries_on_the_same_poll_with_one_alert_and_exits_still_go(store, monkeypatch):
    from sleeve_fund.paper.portfolio_gate import paper_gate

    _rebased(store)
    _make(store, "x", 0)
    _trade(store, "x", 100, T0 + MIN)
    _mark(store, T0 + MIN)
    assert _entry(store, T0 + MIN).outcome == "approved"
    _broken(store, monkeypatch)
    st = _mark(store, T0 + 2 * MIN)
    assert st.mark_ts == T0 + 2 * MIN  # the book is still marked: exits, stops and close-now run on it
    d = _entry(store, T0 + 2 * MIN)  # the same poll's next entry
    assert d.outcome == "rejected" and "book check failed" in d.reason and "0.02" in d.reason
    assert paper_gate(store)("x", {"reduce_only": True, "what": "exit"}, 0) is None  # an exit is never gated
    _mark(store, T0 + 3 * MIN)
    assert len(_alerts(store)) == 1


def test_the_hold_is_latched_survives_a_restart_and_only_a_resume_on_a_whole_book_lifts_it(store, monkeypatch):
    from sleeve_fund.store import Store

    _rebased(store)
    _make(store, "x", 0)
    _trade(store, "x", 100, T0 + MIN)
    fix = _broken(store, monkeypatch)
    _mark(store, T0 + MIN)
    with pytest.raises(ValueError, match="still doesn't add up by"):
        supervisor.resume_identity_hold(store, "pm", "checked it")
    fix()
    _mark(store, T0 + 2 * MIN)  # a clean poll alone
    again = Store(engine=store.engine, portfolio_gate=True)  # a restart
    assert _entry(again, T0 + 2 * MIN).outcome == "rejected"
    words = supervisor.resume_identity_hold(again, "pm", "the double count was the test's", now=T0 + 3 * MIN)
    assert "resumed" in words and again.decisions(action="book_identity_resume")
    _mark(again, T0 + 3 * MIN)
    assert _entry(again, T0 + 3 * MIN).outcome == "approved"
    assert len(_alerts(again)) == 1


def test_a_create_with_its_own_capital_is_refused_once_the_fund_holds_the_cash(store):
    """CUT-A (a): from the PM's re-base on, a Strategy has no capital of its own."""
    _rebased(store)
    with pytest.raises(ValueError, match="no capital of its own"):
        _make(store, "x", 10_000)
    _make(store, "y", 0)


def test_the_bridge_switches_once_at_the_rebase_and_not_before(store):  # noqa: F811
    _make(store, "a", 10_000)
    _settle(store, "a", T0)
    assert supervisor.bridge_start(store) is None and supervisor.fund_equity(store) == D(10_000)  # legacy, unchanged
    assert supervisor.book_part(store, store.sleeve("a")) == D(10_000)
    store.archive("a")
    _mark(store, T0)
    _mark(store, T0 + MIN)  # the poll takes "a" out of the book
    st = _rebased(store, at=T0 + 2 * MIN)
    assert supervisor.bridge_start(store) == T0 + 2 * MIN and st.equity == D(100_000)
    _flow(store, "deposit", 5_000, T0 + 3 * MIN, rebase=True)  # a second re-base onto a book that isn't empty
    _mark(store, T0 + 3 * MIN)
    assert supervisor.bridge_start(store) == T0 + 2 * MIN and store.fund_flows()[-1]["refused_at"] is not None


def test_empty_book_pockets_names_a_strategy_still_counted_and_none_when_empty(store):  # noqa: F811
    assert supervisor.empty_book_pockets(store) == []
    _make(store, "a", 5_000)
    _settle(store, "a", T0)
    assert any(p.startswith("a 5,000.00") for p in supervisor.empty_book_pockets(store))


def test_sizing_equity_is_the_marked_book_less_open_gains_and_none_when_stale(store):  # noqa: F811
    _rebased(store)
    _make(store, "x", 0)
    store.record_fill("x", side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id="b1", trade_id="tb1", ts=T0 + MIN)
    store.record_equity("x", equity=200.0, cash=-1_000.0, qty=1.0, price=1_200.0, benchmark=0.0, ts=T0 + MIN)
    _mark(store, T0 + MIN)  # the book: 100,000 - 1,000 + 1,200 = 100,200, a 200 gain open
    assert supervisor.sizing_equity(store, T0 + MIN) == D(100_000)
    store.record_equity("x", equity=-300.0, cash=-1_000.0, qty=1.0, price=700.0, benchmark=0.0, ts=T0 + 2 * MIN)
    _mark(store, T0 + 2 * MIN)  # a 300 loss open counts in full
    assert supervisor.sizing_equity(store, T0 + 2 * MIN) == D(99_700)
    assert supervisor.sizing_equity(store, T0 + 4 * MIN) is None  # the mark is over 60 s old


def test_the_interim_open_risk_book_is_the_funds_once_it_holds_the_cash(store):  # noqa: F811
    """CUT-A (d): open_risk.account_book reads no Strategy's own equity after the re-base."""
    from sleeve_fund import open_risk

    _rebased(store)
    _make(store, "x", 0)
    _make(store, "y", 0)
    book, risk, through = open_risk.account_book(store, "x", 0.0, lambda s: None)
    assert (book, risk, through) == (100_000.0, 0.0, [])


def test_a_second_rebase_counts_capital_from_its_own_flow_not_one_applied_at_the_same_poll_before_it(store):  # noqa: F811
    """PE1's reset a3: a withdrawal emptying the book and the next re-base land at the same poll; the new book is the
    re-base's deposit alone, never the withdrawal ahead of it in journal order."""
    _rebased(store)
    _flow(store, "withdraw", 100_000, T0 + MIN)
    _flow(store, "deposit", 50_000, T0 + MIN, rebase=True)
    st = _mark(store, T0 + MIN)
    assert [e[1] for e in _ended(store)] == [True, True, True]
    assert supervisor.fund_book(store).capital == D(50_000) == st.equity


def _covered(store, name, intent, at, oid):
    """A fill on an order of `intent` that left `name` 300 below zero, and the insurance credit naming it."""
    store.record_order(name, order_id=oid, side="SELL", qty=1.0, intent=intent, reason="QA", signal={}, ts=at)
    store.record_fill(name, side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id=f"{oid}-in", trade_id=f"{oid}-in",
                      ts=at)
    store.record_fill(name, side="SELL", qty=1.0, price=700.0, fee=0.0, order_id=oid, trade_id=f"{oid}-t", ts=at)
    store.record_insurance(name, price=700.0, amount=300.0, order_id=oid, trade_id=f"{oid}-t", ts=at)


def test_an_insurance_credit_naming_no_liquidation_fill_holds_entries_with_one_alert(store):  # noqa: F811
    """MUST-FUNDCASH-LOSS (HoE): a 0-capital Strategy's stop-out loss credited back as insurance hides from the cash
    leg (both readers count insurance); the insurance leg holds new entries, alerts once, and a Resume lifts it for
    that credit."""
    _rebased(store)
    _make(store, "x", 0)
    _covered(store, "x", "stop_loss", T0 + MIN, "s1")
    _mark(store, T0 + 2 * MIN)
    _mark(store, T0 + 3 * MIN)
    assert len(_alerts(store)) == 1 and "x: an insurance credit of 300.00 names no liquidation fill" in _alerts(store)[0]
    assert _entry(store, T0 + 3 * MIN).outcome == "rejected"
    with pytest.raises(ValueError, match="must name"):
        store.record_insurance("x", price=1.0, amount=1.0, ts=T0 + 3 * MIN)
    supervisor.resume_identity_hold(store, "pm", "QA: the credit is QD's bug, fixed", now=T0 + 4 * MIN)
    _mark(store, T0 + 5 * MIN)
    assert _entry(store, T0 + 5 * MIN).outcome == "approved" and len(_alerts(store)) == 1


def test_an_insurance_credit_for_a_liquidation_fill_is_no_break(store):  # noqa: F811
    _rebased(store)
    _make(store, "x", 0)
    _covered(store, "x", "liquidation", T0 + MIN, "l1")
    _mark(store, T0 + 2 * MIN)
    assert _alerts(store) == [] and supervisor.insurance_breaks(store) == []
