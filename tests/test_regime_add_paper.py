"""P2-5 on paper: an entry sent while a position is held is journaled "add" in gate_decisions only; its order row keeps
intent "entry" (DA 13 Oct 18:53 UK: store.INTENTS has no "add", and the entry readers and dashboard labels look for
"entry"). Kept apart: the exposure-gate strategy's offline fixtures are autouse."""
from __future__ import annotations

from decimal import Decimal as D

import pandas as pd
from sqlalchemy import select
from test_exposure_gate_xfails import STD_T0, M, Plan, _offline, _orders, run, store  # noqa: F401
from test_portfolio_paper_entry import _gated

from sleeve_fund.store import INTENTS, gate_decisions_t
from sleeve_fund.strategies.base import LongFlatStrategy


def test_on_paper_an_add_is_add_in_the_gate_journal_only_and_its_order_row_stays_an_entry(
        tmp_path, store, monkeypatch):  # noqa: F811
    paper, seen = _gated(store), {}
    real = LongFlatStrategy._send_entry

    def send(self, side, qty, reason, signal, what="entry", sent=None, **kw):  # B-30 (TEC): on_sent passes through
        seen.setdefault("strategy", self)
        return real(self, side, qty, reason, signal, what, sent, **kw)

    monkeypatch.setattr(LongFlatStrategy, "_send_entry", send)

    def add(rt, kw):
        from nautilus_trader.model import OrderSide

        s = seen["strategy"]
        assert s._position_qty() >= s._min_qty(), "set-up: no position held to add to"
        s._send_entry(OrderSide.BUY, D("0.002"), "QA add", {"close": 60_000.0})

    # 1% risk at the 2% stop sizes the entry (~9,200 off the 20,000 fund, CUT-A) inside the 0.5x net limit, leaving
    # the add its room (TEC-QD-CUTA: sized at the risk profile's 66% the entry alone fills the limit).
    plan = Plan(t0=STD_T0, minutes=6, windows=[(M(STD_T0, 2), M(STD_T0, 6), 1)], params={"risk_per_trade": 0.01},
                hooks=[(M(STD_T0, 4) + pd.Timedelta(seconds=30), add)])
    run(tmp_path, paper, plan, monkeypatch)
    with paper.engine.connect() as c:
        kinds = [r.intent for r in c.execute(select(gate_decisions_t.c.intent).order_by(gate_decisions_t.c.id))]
    assert kinds == ["entry", "add"], kinds
    assert "add" not in INTENTS and {o["intent"] for o in _orders(paper)} <= set(INTENTS)
    assert [o["intent"] for o in _orders(paper) if o["side"] == "BUY"] == ["entry", "entry"]
