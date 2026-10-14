"""Bridge Drop A: the gate checks every idea against the PM's compliance tests in force, read inside its lock at
evaluation (BRIDGE-3-QA), and journals the version and the test that refused or trimmed it (DA 0027:
gate_decisions.compliance_version, test_hit). On SQLite, and on Postgres with TEST_DATABASE_URL (the CellLedger
harness)."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D

from sqlalchemy import select
from test_gate_ledger import _marked
from test_portfolio_gate import PX, T0

from sleeve_fund.paper.portfolio_gate import paper_gate
from sleeve_fund.portfolio import compliance
from sleeve_fund.portfolio.gate import check_order
from sleeve_fund.portfolio.limits import Intent
from sleeve_fund.store import gate_decisions_t, gate_reservations_t

RISK = PX * D("0.2")  # a stop 20% away: 12,000 a unit, so T1 binds before gross, margin and T3


def _wide(qty: str, asset: str = "BTC") -> Intent:
    return Intent(asset, 1, D(qty), PX, PX / 5, RISK, D("0.001"), D("0.001"))


def _rows(ledger) -> list:
    d = gate_decisions_t.c
    with ledger.engine.connect() as c:
        return list(c.execute(select(d.outcome, d.approved_qty, d.limit_hit, d.test_hit, d.compliance_version,
                                     d.profile_version).order_by(d.seq)))


def test_an_idea_is_checked_against_the_seeded_version_and_the_row_names_it():
    ledger = _marked()
    c = check_order(ledger, "a", _wide("0.01"), None, T0)
    assert (c.decision.outcome, c.decision.approved_qty) == ("approved", D("0.01"))
    r = _rows(ledger)[-1]
    assert (r.compliance_version, r.test_hit, r.limit_hit, r.profile_version) == (1, None, None, 1)


def test_t1_trims_to_four_percent_of_the_book_and_the_row_names_the_test_not_a_limit():
    """Book 20,000: T1 4% is 800 of loss at the stops, 800 / 12,000 a unit = 0.0666 -> 0.066 at the 0.001 step."""
    ledger = _marked()
    c = check_order(ledger, "a", _wide("1"), None, T0)
    assert (c.decision.outcome, c.decision.approved_qty) == ("trimmed", D("0.066"))
    r = _rows(ledger)[-1]
    assert (r.test_hit, r.limit_hit, r.compliance_version) == ("asset_stop_loss", None, 1)


def test_a_pm_edit_between_the_idea_and_the_gate_applies_at_evaluation():
    """T1 to 2% after the first idea: the next one is trimmed to 400 / 12,000 = 0.0333 -> 0.033 under version 2."""
    ledger = _marked()
    check_order(ledger, "a", _wide("0.001"), None, T0)
    v2 = compliance.save(ledger.store, replace(compliance.load(ledger.store), asset_open_risk=0.02), "pm",
                         note="QA: tighter per-asset loss")
    c = check_order(ledger, "b", _wide("1", "ETH"), None, T0)
    assert v2 == 2 and c.decision.approved_qty == D("0.033")
    assert [(r.compliance_version, r.test_hit) for r in _rows(ledger)] == [(1, None), (2, "asset_stop_loss")]


def test_an_asset_no_group_maps_is_refused_naming_it_under_the_group_test():
    ledger = _marked()
    c = check_order(ledger, "a", _wide("0.001", "DOGE"), None, T0)
    assert c.decision.outcome == "rejected" and "DOGE" in c.decision.reason and c.reservation is None
    r = _rows(ledger)[-1]
    assert (r.test_hit, r.limit_hit, r.compliance_version) == ("type_stop_loss", None, 1)


def test_with_the_group_test_off_an_unmapped_asset_trades_under_the_others():
    ledger = _marked()
    compliance.save(ledger.store, replace(compliance.load(ledger.store), group_open_risk=None), "pm",
                    note="QA: group test off")
    c = check_order(ledger, "a", _wide("0.001", "DOGE"), None, T0)
    assert c.decision.outcome == "approved"


def test_tests_that_cant_be_read_refuse_the_entry_and_reserve_nothing(monkeypatch):
    ledger = _marked()

    def broken(*a, **k):
        raise RuntimeError("relation compliance_tests does not exist")

    monkeypatch.setattr(ledger.store, "compliance", broken)
    c = check_order(ledger, "a", _wide("0.001"), None, T0)
    assert c.decision.outcome == "rejected" and "compliance tests couldn't be read" in c.decision.reason
    assert [r.outcome for r in _rows(ledger)] == ["error"]
    with ledger.engine.connect() as conn:
        assert not list(conn.execute(select(gate_reservations_t)))


def test_paper_reads_the_compliance_tests_and_a_direct_profile_still_wins():
    from sleeve_fund.risk import PORTFOLIO

    ledger = _marked()
    assert paper_gate(ledger.store).profile is None
    check_order(ledger, "a", _wide("1"), PORTFOLIO, T0)  # an explicit profile: no T1; its 5% fund loss binds
    r = _rows(ledger)[-1]
    assert (r.compliance_version, r.test_hit, r.limit_hit) == (None, None, "open_risk")
