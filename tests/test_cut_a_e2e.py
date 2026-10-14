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
@pytest.mark.xfail(strict=True, reason="needs PE2's funding of the simulated account from the fund (SIZE-FUND Q3): "
                   "until then the reconcile reads its journal cash of 0 against the account's 20,000 and halts it")
def test_a_zero_balance_strategy_sizes_off_the_fund_fills_pays_its_fee_marks_and_renders(tmp_path, monkeypatch):
    """mark -> idea -> size -> fill -> fee -> next mark -> page. Its simulated account posts margin from the fund's free
    cash (PE2, SIZE-FUND Q3), here 20,000; its journal starts at 0. Never halted, paused or refused for a reason of its
    own, and every mark keeps equity = cash + what its position is worth."""
    import test_exposure_gate_xfails as h
    from fastapi.testclient import TestClient
    from test_portfolio_paper_entry import END

    from sleeve_fund.dashboard import app as app_mod

    store = Store(f"sqlite:///{tmp_path}/e2e.db", portfolio_gate=True)
    fund = D(20_000)
    DbLedger(store, lambda: None).set_state(PortfolioState(mark_ts=END, equity=fund, hwm=fund, day=END.date(),
                                                           day_start_equity=fund))
    params = {**h.PERP, "stop_loss": h.STOP}
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params=params)
    plan = h.Plan(t0=h.STD_T0, minutes=6, windows=[(h.M(h.STD_T0, 2), h.M(h.STD_T0, 6), 1)], balance=20_000.0)
    h.run(tmp_path, store, plan, monkeypatch)

    entries = [o for o in h._orders(store) if o["intent"] == "entry"]
    assert entries and entries[0]["status"] == "filled", entries
    signal = entries[0]["signal"]
    budget = float(signal["budget"])
    assert budget > 0.3 * 20_000, signal  # off the fund's 20,000, never a starting balance of 0
    fills = h._fills(store)
    assert fills and float(fills[0]["fee"]) > 0, fills
    kinds = {e["kind"] for e in h._events(store)}
    assert not kinds & {"risk_halt", "risk_pause", "entry_refused_fund_equity"}, kinds
    assert store.sleeve(h.NAME).status == "running"
    book, last = store.journal_book(h.NAME, 0), store.last_equity(h.NAME)
    unrealised = float(book["qty"]) * (float(last["price"]) - float(book["entry_px"] or 0.0))
    assert float(last["equity"]) - 20_000 == pytest.approx(float(book["cash"]) + unrealised, abs=0.05), (last, book)
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    page = TestClient(app_mod.create_app(store)).get(f"/sleeves/{h.NAME}", auth=("pm", "test-pw"))
    assert page.status_code == 200 and "Traceback" not in page.text and "division" not in page.text.lower()


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
