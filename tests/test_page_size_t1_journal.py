"""PAGE-SIZE-T1 (2 of 2), through the paper gate and its journal: the decision row carries t1_used_before, t1_cap,
risk_per_unit and size_share exactly as decide() had them; T1 off leaves all four NULL. Worked by hand: the fund is
marked at 20,000 (seeded T1 4% = 800), Strategy a asks 0.1 at 60,000 with a 2% stop and no fee: 1,200 a unit, 120 in
all, 0.15 of T1's room. Run beside test_gate_ledger.py and test_p21b_wiring_qa.py, on SQLite and PG."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D

import pytest
from sqlalchemy import select
from test_gate_ledger import _drop_schemas, _store, _strategies  # noqa: F401
from test_p21b_wiring_qa import _approved

from sleeve_fund.money import money
from sleeve_fund.portfolio import compliance
from sleeve_fund.store import gate_decisions_t

pytestmark = pytest.mark.no_maintenance_tables

FOUR = ("t1_used_before", "t1_cap", "risk_per_unit", "size_share")


@pytest.fixture
def store():
    s = _store()
    _strategies(s)
    return s


def _row(store, rid):
    with store.engine.connect() as c:
        return c.execute(select(gate_decisions_t).where(gate_decisions_t.c.id == rid)).one()._mapping


def test_the_decision_row_carries_what_t1_compared(store):
    row = _row(store, _approved(store))
    assert {k: money(row[k]) for k in FOUR} == {"t1_used_before": D(0), "t1_cap": D(800),
                                                "risk_per_unit": D(1200), "size_share": D("0.15")}


def test_t1_off_leaves_the_four_null(store, monkeypatch):
    real = compliance.load
    monkeypatch.setattr(compliance, "load", lambda *a, **k: replace(real(*a, **k), asset_open_risk=None))
    row = _row(store, _approved(store))
    assert [row[k] for k in FOUR] == [None] * 4


def test_a_later_cap_change_leaves_the_entrys_row_as_recorded(store):
    """Advisor: the row is what T1 compared when the entry was judged; the PM halving T1 afterwards changes nothing in
    it (the page reads the row, never today's cap)."""
    rid = _approved(store)
    before = {k: _row(store, rid)[k] for k in FOUR}
    compliance.save(store, replace(compliance.load(store), asset_open_risk=0.02), "pm", "qa: T1 halved")
    assert compliance.load(store).asset_open_risk == 0.02
    assert {k: _row(store, rid)[k] for k in FOUR} == before and money(before["t1_cap"]) == D(800)
