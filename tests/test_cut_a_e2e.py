"""CUT-A end to end (HoE Drop A, QD's red-first cell): a Strategy with a starting balance of 0 inside the fund goes mark
-> idea -> size -> fill -> fee -> next mark -> page render through the paper engine. Kept apart from
test_cut_a_fund_equity: the exposure-gate harness's offline fixtures are autouse."""
from __future__ import annotations

from decimal import Decimal as D

import pandas as pd
import pytest
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.portfolio.gate import PortfolioState
from sleeve_fund.store import Store



# ---- End to end: a Strategy with a starting balance of 0 inside the fund ---------------------------------------------
@pytest.mark.parametrize("headroom", [10, 1])
def test_a_zero_balance_strategy_sizes_off_the_fund_fills_pays_its_fee_marks_and_renders(tmp_path, monkeypatch,
                                                                                         headroom):
    """mark -> idea -> size -> fill -> fee -> next mark -> page. The fund holds the 20,000 (the PM's re-base deposit);
    the Strategy starts at 0 and its simulated account is funded virtually from the fund (PE2, x10 headroom as paper,
    x1 too: no decision reads the funding). Never halted, paused or refused for a reason of its own, and every mark
    keeps its own equity (from 0) = journal cash + its position at the mark."""
    import test_exposure_gate_xfails as h
    from fastapi.testclient import TestClient
    from test_portfolio_paper_entry import END

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.supervisor import Supervisor

    store = Store(f"sqlite:///{tmp_path}/e2e.db", portfolio_gate=True)
    params = {**h.PERP, "stop_loss": h.STOP}
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params=params)
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)  # the PM's re-base, before the session
    store.settle_flow(store.fund_flow("deposit", 20_000, actor="pm", reason="QA: the fund's capital", rebase=True,
                                      now=before), before)
    Supervisor(store).mark_portfolio(END)  # one mark covers the session (fresh: it is never older than the decision)
    plan = h.Plan(t0=h.STD_T0, minutes=6, windows=[(h.M(h.STD_T0, 2), h.M(h.STD_T0, 6), 1)],
                  balance=20_000.0 * headroom)
    h.run(tmp_path, store, plan, monkeypatch)

    kinds = {e["kind"] for e in h._events(store)}
    assert not kinds & {"risk_halt", "risk_pause", "reconcile_mismatch", "entry_refused_fund_equity",
                        "entry_refused_open_risk", "handler_failed"}, kinds
    entries = [o for o in h._orders(store) if o["intent"] == "entry"]
    assert entries and entries[0]["status"] == "filled", entries
    assert float(entries[0]["signal"]["budget"]) > 0.3 * 20_000, entries[0]["signal"]  # off the fund, never 0
    fills = h._fills(store)
    assert fills and float(fills[0]["fee"]) > 0, fills
    assert store.sleeve(h.NAME).status == "stopped"  # the replay's end stops it; never halted or paused on the way
    book, last = store.journal_book(h.NAME, 0), store.last_equity(h.NAME)
    held = float(book["qty"]) * float(last["price"])  # the journal keeps spot-style cash: the buy's notional came out
    assert float(last["equity"]) == pytest.approx(float(book["cash"]) + held, abs=0.05), (last, book)
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    page = TestClient(app_mod.create_app(store)).get(f"/sleeves/{h.NAME}", auth=("pm", "test-pw"))
    assert page.status_code == 200 and "Traceback" not in page.text and "division" not in page.text.lower()


@pytest.mark.parametrize("headroom", [10, 1])
def test_a_fund_held_losing_trade_books_no_insurance_and_the_fund_falls_by_exactly_the_loss(tmp_path, monkeypatch,
                                                                                            headroom):
    """MUST-FUNDCASH-LOSS (HoQA GOLD-FUNDCASH F3): a Strategy made with 0 is below zero after any loss, which is the
    fund's loss. The insurance fund covers only a simulated account taken past its balance, never an ordinary stop-out:
    nothing is credited back and the fund's total falls by exactly what the trade lost, fees included."""
    import test_exposure_gate_xfails as h
    from test_portfolio_paper_entry import END

    from sleeve_fund import supervisor
    from sleeve_fund.supervisor import Supervisor

    store = Store(f"sqlite:///{tmp_path}/loss.db", portfolio_gate=True)
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params={**h.PERP, "stop_loss": h.STOP})
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 20_000, actor="pm", reason="QA: the fund's capital", rebase=True,
                                      now=before), before)
    Supervisor(store).mark_portfolio(END)
    t0 = h.STD_T0
    plan = h.Plan(t0=t0, minutes=18, windows=[(h.M(t0, 2), h.M(t0, 17), 1)],
                  price=lambda s: 60_000 + s * 0.01 if s < 720 else 58_000.0, balance=20_000.0 * headroom)
    h.run(tmp_path, store, plan, monkeypatch)

    book = store.journal_book(h.NAME, 0)
    assert float(book["qty"]) == 0 and float(book["cash"]) < -100, book  # stopped out, flat, at a real loss
    assert float(store.insurance_total(h.NAME)) == 0.0
    assert "insurance_fund" not in {e["kind"] for e in h._events(store)}
    assert float(supervisor.fund_book(store).total) == pytest.approx(20_000 + float(book["cash"]), abs=0.01)
    assert abs(supervisor.identity_gap(store)) <= D("0.01")


@pytest.mark.parametrize("age, sized", [(59, True), (61, False)])
def test_stale_60_a_fund_mark_59_s_old_sizes_the_entry_and_61_s_old_refuses_it(tmp_path, monkeypatch, age, sized):
    """STALE-60 (Advisor, replaces Q5's "two polls"): the entry sizes off a fund mark up to 60 s old; older is refused
    as stale, never sized off. The book is marked `age` seconds before the entry's decision (09:52:00, the window's
    first bar close). The gate's own stale block is the same 60 s, and sizing asks first, so the refusal is sizing's."""
    import test_exposure_gate_xfails as h

    store = Store(f"sqlite:///{tmp_path}/stale.db", portfolio_gate=True)
    fund, decided = D(20_000), h.M(h.STD_T0, 2).to_pydatetime()
    at = decided - pd.Timedelta(seconds=age)
    DbLedger(store, lambda: None).set_state(PortfolioState(mark_ts=at, equity=fund, hwm=fund, day=at.date(),
                                                           day_start_equity=fund))
    plan = h.Plan(t0=h.STD_T0, minutes=4, windows=[(h.M(h.STD_T0, 2), h.M(h.STD_T0, 4), 1)])
    h.run(tmp_path, store, plan, monkeypatch)
    entries = [o for o in h._orders(store) if o["intent"] == "entry"]
    refused = [e["message"] for e in h._events(store) if e["kind"] == "entry_refused_fund_equity"]
    if sized:
        assert entries and entries[0]["ts"] == decided and not refused, (entries, refused)
    else:
        assert not entries and refused and "fund equity stale" in refused[0], (entries, refused)
