"""P2-1b W2 through the paper engine: an entry waits for the portfolio gate over the journal, its order row holds the
reservation, its fill settles it, and a gated entry still resting at the next bar close is cancelled by its own process
(Advisor 8 Oct 06:10 UK). Kept apart from test_portfolio_paper: the exposure-gate strategy's offline fixtures are
autouse."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

import pandas as pd
from sqlalchemy import select
from test_exposure_gate_xfails import M, NAME, STD_T0, Plan, _events, _offline, _orders, run, store  # noqa: F401

from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.paper.portfolio_gate import paper_gate
from sleeve_fund.portfolio.gate import PortfolioState
from sleeve_fund.research.portfolio import gate_intent  # noqa: F401 - the intent the engine sends is gate_intent's
from sleeve_fund.store import Store, gate_decisions_t, gate_reservations_t

END = STD_T0.to_pydatetime() + timedelta(hours=1)  # marked through the whole run


def _gated(journal, equity=D(20_000)) -> Store:
    paper = Store(engine=journal.engine, portfolio_gate=True)  # the same journal, as paper opens it
    DbLedger(paper, lambda: None).set_state(PortfolioState(mark_ts=END, equity=equity, hwm=equity, day=END.date(),
                                                           day_start_equity=equity))
    return paper


def _decisions(journal) -> list:
    d, r = gate_decisions_t, gate_reservations_t
    with journal.engine.connect() as c:
        return list(c.execute(select(d.c.id, d.c.order_id, d.c.outcome, r.c.release_reason)
                              .outerjoin(r, r.c.decision_id == d.c.id).order_by(d.c.id)))


def test_a_paper_entry_is_checked_its_order_holds_the_reservation_and_its_fill_releases_it(
        tmp_path, store, monkeypatch):  # noqa: F811
    paper = _gated(store)
    run(tmp_path, paper, Plan(t0=STD_T0, minutes=6, windows=[(M(STD_T0, 2), M(STD_T0, 6), 1)]), monkeypatch)
    entries = [o for o in _orders(paper) if o["intent"] == "entry"]
    assert entries, "no entry sent"
    rows = _decisions(paper)
    assert [(r.order_id, r.outcome, r.release_reason) for r in rows] == [
        (entries[0]["order_id"], "approved", "fill")], rows  # one check, linked to its order, settled by its fill


def test_a_paper_entry_the_portfolio_refuses_is_never_sent_and_says_why(tmp_path, store, monkeypatch):  # noqa: F811
    paper = _gated(store, equity=D(1))  # a book this small has no room for the smallest order
    run(tmp_path, paper, Plan(t0=STD_T0, minutes=6, windows=[(M(STD_T0, 2), M(STD_T0, 6), 1)]), monkeypatch)
    assert not [o for o in _orders(paper) if o["intent"] == "entry"]
    assert _decisions(paper), "the entry never reached the gate"
    told = [e["message"] for e in _events(paper) if e["kind"] == "entry_refused_by_portfolio"]
    assert len(told) == 1 and "above the 0.50x limit" in told[0], told  # once while it lasts, with the gate's reason


def test_a_gated_entry_still_resting_at_the_next_bar_close_is_cancelled_and_its_reservation_released(
        tmp_path, store, monkeypatch):  # noqa: F811
    """A resting stop entry (the harness's B3 breakout, never reached) given the gate's reservation as an engine's
    gated entry holds it: the strategy's own process cancels it at the next close and the confirmed cancel releases
    the rest of the reservation as 'cancel'."""
    paper = _gated(store)
    held = {}

    def attach(rt, kw):
        (o,) = [o for o in _orders(paper) if o["intent"] == "entry"]
        g = paper_gate(paper)(rt.name, {"side": 1, "qty": D("0.01"), "close": D(60_000), "price": D(60_000),
                                        "stop_frac": 0.02, "what": "entry", "step": D("0.001"),
                                        "min_qty": D("0.001"), "instrument": "BTC", "leverage": None,
                                        "atr_pct": None}, pd.Timestamp(rt.now()).value)
        g.attach(o["order_id"])
        held.update(order=o["order_id"], rid=g.checked.reservation)

    plan = Plan(t0=STD_T0, minutes=5, rest=[(M(STD_T0, 2), 1, 70_000.0, 0.01)],
                hooks=[(M(STD_T0, 2) + pd.Timedelta(seconds=30), attach)])
    run(tmp_path, paper, plan, monkeypatch)
    assert held, "set-up: the hook never ran"
    order = {o["order_id"]: o for o in _orders(paper)}[held["order"]]
    assert order["status"] == "canceled", order
    assert [r.release_reason for r in _decisions(paper) if r.id == held["rid"]] == ["cancel"]


def test_a_gated_entry_is_cancelled_at_the_next_close_even_when_that_bar_is_not_decided_on(
        tmp_path, store, monkeypatch):  # noqa: F811
    """CR W2-1: the next bar held, gapped or refused as stale still closes the bar the entry may rest into, so the
    cancel runs before those checks return."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    paper, held = _gated(store), {}
    skipped = {_ns_of(M(STD_T0, m)) for m in (3, 4, 5)}  # every bar after the decision's, refused as stale
    real = LongFlatStrategy._accept
    monkeypatch.setattr(LongFlatStrategy, "_accept",
                        lambda self, bar: False if int(bar.ts_event) in skipped else real(self, bar))

    def attach(rt, kw):
        (o,) = [o for o in _orders(paper) if o["intent"] == "entry"]
        g = paper_gate(paper)(rt.name, _qa_intent(), pd.Timestamp(rt.now()).value)
        g.attach(o["order_id"])
        held.update(order=o["order_id"])

    plan = Plan(t0=STD_T0, minutes=5, rest=[(M(STD_T0, 2), 1, 70_000.0, 0.01)],
                hooks=[(M(STD_T0, 2) + pd.Timedelta(seconds=30), attach)])
    run(tmp_path, paper, plan, monkeypatch)
    order = {o["order_id"]: o for o in _orders(paper)}[held["order"]]
    assert order["status"] == "canceled", order


def _ns_of(t) -> int:
    return pd.Timestamp(t).value


def _qa_intent() -> dict:
    return {"side": 1, "qty": D("0.01"), "close": D(60_000), "price": D(60_000), "stop_frac": 0.02, "what": "entry",
            "step": D("0.001"), "min_qty": D("0.001"), "instrument": "BTC", "leverage": None, "atr_pct": None}


def test_a_crashed_strategys_resting_entry_is_cancelled_by_the_supervisor_and_answered_at_its_restart(
        tmp_path, store, monkeypatch):  # noqa: F811
    """ORPHAN-GUARD og2 + og3 through the paper engine (HoE 13 Oct, the pair): the process dies with a gated entry
    resting; the supervisor marks it down and asks the cancel; the restart answers it ("not found": a sandbox's orders
    end with its process) and its entries stay blocked until the quiet window confirms it, which releases the
    reservation as 'cancel'."""
    from sleeve_fund.paper import orphan_guard as og

    paper, held = _gated(store), {}

    def attach(rt, kw):
        (o,) = [o for o in _orders(paper) if o["intent"] == "entry"]
        g = paper_gate(paper)(rt.name, _qa_intent(), pd.Timestamp(rt.now()).value)
        g.attach(o["order_id"])
        held.update(order=o["order_id"], rid=g.checked.reservation)

    # The process dies inside the bar after the decision's: its bar-close cancel never runs.
    run(tmp_path, paper, Plan(t0=STD_T0, minutes=3, rest=[(M(STD_T0, 2), 1, 70_000.0, 0.01)],
                              hooks=[(M(STD_T0, 2) + pd.Timedelta(seconds=30), attach)]), monkeypatch)
    assert held, "set-up: the hook never ran"
    oid = held["order"]
    assert {o["order_id"]: o for o in _orders(paper)}[oid]["status"] not in ("canceled", "expired"), "set-up"

    down = M(STD_T0, 3).to_pydatetime()
    og.owner_down(paper, NAME, down, "crash, exit code 1")
    assert og.supervisor_pass(paper, down + timedelta(seconds=5)) == [oid]  # og2: asked within the 60 s
    assert og.resting_block(paper, NAME, down + timedelta(seconds=5))

    run(tmp_path, paper, Plan(t0=STD_T0 + pd.Timedelta(minutes=5), minutes=2, tag="r"), monkeypatch)
    assert og.down_since(paper, NAME) is None  # the restart marked it up
    assert [c["order_id"] for c in paper.open_cancels(NAME)] == [oid]  # answered "not found", not yet confirmed
    assert not [o for o in _orders(paper) if o["intent"] == "entry" and o["order_id"] != oid]  # blocked meanwhile

    og.supervisor_pass(paper, M(STD_T0, 9).to_pydatetime())  # past the quiet window after the answer
    assert not paper.open_cancels(NAME)
    assert og.resting_block(paper, NAME, M(STD_T0, 9).to_pydatetime()) is None
    assert {o["order_id"]: o for o in _orders(paper)}[oid]["status"] in ("canceled", "expired")
    assert [r.release_reason for r in _decisions(paper) if r.id == held["rid"]] == ["cancel"]
