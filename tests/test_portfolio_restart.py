"""P2-1b W1, CR F229-1 / HoQA MF229-1: a paper restart on a journal under the portfolio gate, through the paper engine
(replay, so CHOKE reads the portfolio as a paper process does). Kept apart from test_portfolio_paper: the exposure-gate
strategy's offline fixtures are autouse."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

import pandas as pd
import pytest
from test_exposure_gate_xfails import (M, STD_T0, Plan, _events, _offline, _orders,  # noqa: F401
                                      _restart_balance, run, store)

from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.portfolio.gate import PortfolioState
from sleeve_fund.store import Store

T1 = STD_T0 + pd.Timedelta(hours=1)
FRESH = T1.to_pydatetime() + timedelta(minutes=30)  # marked through the whole restart
BOOK = {"mark_ts": FRESH, "equity": D(20_000), "hwm": D(20_000), "day": FRESH.date(), "day_start_equity": D(20_000)}
STATES = {  # the fund's state the restarted strategy finds, and its CHOKE label
    "unmarked": (None, "Portfolio unchecked"),  # a supervisor restart before its first mark (every journal till CASH-2)
    "paused": (PortfolioState(**BOOK, paused_until=FRESH + timedelta(days=1), paused="3.2% down on the day"), "Portfolio paused"),
    "halted": (PortfolioState(**BOOK, halted="16% under the high-water mark"), "Portfolio halted"),
}


@pytest.mark.parametrize("fund", list(STATES))
def test_a_paper_restart_holding_a_position_keeps_its_stop_unless_the_portfolio_is_halted(
        fund, tmp_path, store, monkeypatch):  # noqa: F811
    """An unmarked book (limits unchecked) and the daily pause (which keeps positions) are, like P1-SG21's stale feed,
    not blocks a position was held under: the model's restored stop stays and no incident is raised. A portfolio halt
    flattens, so a position restored under one is held while nothing may open: the safety stop and an incident."""
    one = Plan(t0=STD_T0, minutes=10, tag="r1", windows=[(M(STD_T0, 2), M(STD_T0, 10), 1)])
    run(tmp_path, store, one, monkeypatch)
    assert any(o["intent"] == "entry" for o in _orders(store)), "set-up: no position before the restart"
    paper = Store(engine=store.engine, portfolio_gate=True)  # the same journal, as paper opens it
    state, label = STATES[fund]
    if state is not None:
        DbLedger(paper, lambda: None).set_state(state)
    two = Plan(t0=T1, minutes=5, tag="r2", windows=[(T1 - pd.Timedelta(minutes=10), M(T1, 5), 1)],
               holes=[(0, 20)], balance=_restart_balance(paper))
    run(tmp_path, paper, two, monkeypatch)
    blocks = [e["message"] for e in _events(paper, ("entry_blocked",)) if e["ts"] >= T1]
    assert any(label in m for m in blocks), f"set-up: CHOKE didn't name {label}: {blocks}"
    after = [e["message"][:200] for e in _events(paper, ("incident",)) if e["ts"] >= T1]
    if fund == "halted":
        assert len(after) == 1 and "held while nothing may open" in after[0] and label in after[0], after
    else:
        assert not after, after
    assert not [o for o in _orders(paper) if o["ts"] >= T1 and o["intent"] == "entry"], "opened while blocked"
