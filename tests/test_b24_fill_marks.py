"""B-24 part 2 (Advisor B24-LEVERAGE, 20 Oct 2026), through the store: the paper gate's decision journals the five
figures, each fill of the gated entry marks them at the order's average price so far (price_source "fill"), a fill that
slippage takes past T4 raises an error event and stays booked (cell j), and an order no decision holds is never marked.

Figures are worked by hand: the fund is marked at 20,000 and holds nothing, so the decision's book is 0 gross, 0 net,
Strategy a flat, spot (margin leverage 1). Run beside test_gate_ledger.py and test_p21b_wiring_qa.py, on SQLite and
PG (TEST_DATABASE_URL)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

import pytest
from sqlalchemy import select, text
from test_gate_ledger import _drop_schemas, _store, _strategies  # noqa: F401
from test_p21b_wiring_qa import (
    EQ,
    T0,
    _approved,
    _entry_intent,
    _hold,
    _ns,
    paper_gate,
    portfolio_pass,
)

from sleeve_fund.money import money
from sleeve_fund.store import events_t, gate_decisions_t, gate_fill_marks_t

pytestmark = pytest.mark.no_maintenance_tables

FIVE = ("margin_leverage", "share_of_fund", "gross_after", "net_underlying_after", "held_before")


@pytest.fixture
def store():
    s = _store()
    _strategies(s)
    return s


def _fill(store, qty, px, trade_id, oid="o-1"):
    store.book_fill("a", side="BUY", qty=qty, price=px, fee=0.1, order_id=oid, trade_id=trade_id, ts=T0)


def _figures(mark):
    return {k: money(mark[k]) for k in FIVE[:4]}


def _events(store, kind):
    with store.engine.connect() as c:
        return list(c.execute(select(events_t).where(events_t.c.kind == kind)))


def _breaches(store):
    return _events(store, "portfolio_fill_past_t4")


def test_the_decision_journals_all_five_figures_once(store):
    rid = _approved(store)  # 0.1 at 60,000 = 6,000 of 20,000
    with store.engine.connect() as c:
        row = c.execute(select(gate_decisions_t).where(gate_decisions_t.c.id == rid)).one()._mapping
    assert {k: money(row[k]) for k in FIVE} == {"margin_leverage": D(1), "share_of_fund": D("0.3"),
                                                "gross_after": D("0.3"), "net_underlying_after": D("0.3"),
                                                "held_before": D(0)}


def test_each_fill_marks_the_figures_at_the_average_price_so_far(store):
    rid = _approved(store)
    _fill(store, 0.03, 60000.0, "t1")  # 1,800 / 20,000
    _fill(store, 0.07, 61000.0, "t2")  # 0.1 at (1,800 + 4,270) / 0.1 = 60,700: 6,070 / 20,000
    marks = store.fill_marks(rid)
    assert [(m["trade_id"], m["price_source"], money(m["price"])) for m in marks] == [
        ("t1", "fill", D(60000)), ("t2", "fill", D(60700))]
    assert _figures(marks[0]) == {"margin_leverage": D(1), "share_of_fund": D("0.09"), "gross_after": D("0.09"),
                                  "net_underlying_after": D("0.09")}
    assert _figures(marks[1]) == {"margin_leverage": D(1), "share_of_fund": D("0.3035"), "gross_after": D("0.3035"),
                                  "net_underlying_after": D("0.3035")}
    _fill(store, 0.07, 61000.0, "t2")  # the same fill again: booked once, marked once
    assert len(store.fill_marks(rid)) == 2
    assert _breaches(store) == []


def test_j_a_fill_slippage_takes_past_t4_is_alerted_and_stays_booked(store):
    """Approved at 60,000 (net 0.3x of the seeded 0.5x); filled at 101,000: 10,100 / 20,000 = net 0.505x."""
    rid = _approved(store)
    _fill(store, 0.1, 101000.0, "t1")
    [mark] = store.fill_marks(rid)
    assert money(mark["net_underlying_after"]) == D("0.505")
    [ev] = _breaches(store)
    assert ev.level == "error" and ev.sleeve == "a" and ev.message.startswith("o-1: ") and "0.5" in ev.message
    assert money(store.held_qty("a")) == D("0.1")


@pytest.mark.parametrize("db", [True, False], ids=["db-error", "python-error"])
def test_a_t4_check_that_cannot_run_is_an_error_event_and_the_fill_stays_booked(store, monkeypatch, db):
    """HoE 21:19 UK MUST, CR2 MUST (21:25 UK, its probe): unreadable compliance tests never undo a fill, and never pass
    silently. A database error is the PG case: it aborts the fill's transaction unless the read has its own savepoint."""
    rid = _approved(store)
    real = store.compliance

    def unreadable(version=None, conn=None):
        if conn is None:
            return real(version)
        if db:
            conn.execute(text("select * from no_such_table_b24"))
        raise RuntimeError("no tests")

    monkeypatch.setattr(store, "compliance", unreadable)
    _fill(store, 0.1, 60000.0, "t1")
    assert money(store.held_qty("a")) == D("0.1") and len(store.fill_marks(rid)) == 1
    [ev] = _events(store, "portfolio_t4_check_skipped")
    assert ev.level == "error" and ev.sleeve == "a" and ev.message.startswith("o-1: fill t1: T4 not checked")


def test_a_mark_that_cannot_be_written_is_an_error_event_and_the_fill_stays_booked(store, monkeypatch):
    """CR2 follow-up 1: no raise in the mark undoes the fill; T4 is still checked on its figures (here past it)."""
    rid = _approved(store)

    def broken(*a, conn=None, **k):
        conn.execute(text("select * from no_such_table_b24"))

    monkeypatch.setattr(store, "record_fill_mark", broken)
    _fill(store, 0.1, 101000.0, "t1")
    assert money(store.held_qty("a")) == D("0.1") and store.fill_marks(rid) == []
    [ev] = _events(store, "portfolio_fill_mark_failed")
    assert ev.level == "error" and ev.message.startswith("o-1: fill t1 not marked")
    assert len(_breaches(store)) == 1


def test_a_short_perp_adding_to_a_short_journals_its_signed_position(store):
    """CR2 follow-up 2: held_before keeps its sign, and the fill's share of fund is the combined short. Strategy a
    holds 0.05 short (as the engine sends it) and sells 0.1 more at 60,000 on a 3x perp: held_before -0.05; the fill
    leaves 0.15 short = 9,000 of 20,000 = 0.45, the fund 0.3 gross and -0.3 net."""
    portfolio_pass(store, T0, EQ)
    intent = {**_entry_intent(), "side": -1, "leverage": 3, "held": D("-0.05")}
    g = paper_gate(store)("a", intent, _ns(T0 + timedelta(seconds=1)))
    assert g.decision.outcome == "approved"
    rid = g.checked.reservation
    store.record_order("a", order_id="o-1", side="SELL", qty=0.1, intent="entry", reason="qa", reservation=rid)
    store.book_fill("a", side="SELL", qty=0.1, price=60000.0, fee=0.1, order_id="o-1", trade_id="t1", ts=T0)
    with store.engine.connect() as c:
        row = c.execute(select(gate_decisions_t).where(gate_decisions_t.c.id == rid)).one()
    assert money(row.held_before) == D("-0.05") and money(row.share_of_fund) == D("0.45")
    [mark] = store.fill_marks(rid)
    assert _figures(mark) == {"margin_leverage": D(3), "share_of_fund": D("0.45"), "gross_after": D("0.3"),
                              "net_underlying_after": D("-0.3")}


def test_an_order_no_decision_holds_and_an_exit_are_never_marked(store):
    _hold(store)  # an entry recorded without the gate: no decision, no reservation
    store.record_order("a", order_id="x-1", side="SELL", qty=0.1, intent="exit", reason="qa")
    store.book_fill("a", side="SELL", qty=0.1, price=60000.0, fee=0.1, order_id="x-1", trade_id="tx", ts=T0)
    with store.engine.connect() as c:
        assert list(c.execute(select(gate_fill_marks_t))) == []
    assert _breaches(store) == []
