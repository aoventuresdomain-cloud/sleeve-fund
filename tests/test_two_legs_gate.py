"""P2-8 two legs, the gate (Advisor 13 Oct 23:55 UK, ruling A): a two-leg entry is one position. Both legs are decided
under one lock, the same share of each or neither, open risk is the sum of the legs with no hedge credit, and net
exposure nets two legs on one underlying."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

from sleeve_fund.portfolio import Intent
from sleeve_fund.portfolio.gate import MemoryLedger, check_order, check_pair, mark_book
from sleeve_fund.portfolio.limits import Book, decide_pair
from sleeve_fund.risk import PORTFOLIO

T0 = datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)
PX = D("60000")
STEP = D("0.001")


def _leg(qty, side, *, risk=PX * D("0.10"), margin=PX, underlying="BTC", min_qty=STEP):
    """Stopless by default: risk at the 10% floor of notional (open_risk.position_risk)."""
    return Intent(underlying, side, D(qty), PX, margin, risk, STEP, min_qty)


def _marked(equity="20000"):
    ledger = MemoryLedger(equity=D(equity))
    mark_book(ledger, D(equity), T0, PORTFOLIO)
    return ledger


def test_a_pair_within_every_limit_reserves_both_legs_in_one_check():
    ledger = _marked()
    a, b = check_pair(ledger, "carry", _leg("0.05", 1), _leg("0.05", -1), PORTFOLIO, T0)
    assert (a.decision.outcome, b.decision.outcome) == ("approved", "approved")
    assert (a.decision.approved_qty, b.decision.approved_qty) == (D("0.05"), D("0.05"))
    assert {r.holding.notional for r in ledger.reservations()} == {D("3000"), D("-3000")}
    assert [r.strategy for r in ledger.records] == ["carry", "carry"]


def test_open_risk_is_the_sum_of_the_legs_with_no_hedge_credit():
    """Each leg alone would pass (600 of a 1,000 cap); together they count 1,200, so both are trimmed by one share."""
    assert check_order(_marked(), "one", _leg("0.1", 1), PORTFOLIO, T0).decision.outcome == "approved"
    # The perp leg at 2x, so margin (6,000 + 3,000 of a 10,000 cap) doesn't bind first.
    a, b = check_pair(_marked(), "carry", _leg("0.1", 1), _leg("0.1", -1, margin=PX / 2), PORTFOLIO, T0)
    assert a.decision.outcome == b.decision.outcome == "trimmed"
    assert a.decision.limit_hit == b.decision.limit_hit == "open_risk"
    assert a.decision.approved_qty == b.decision.approved_qty == D("0.083")  # 1,000 / 1,200 of 0.1, to the step


def test_two_legs_on_one_underlying_net_against_each_other_and_gross_counts_both():
    small = PX * D("0.001")  # a near stop, so open risk doesn't bind
    alone = check_order(_marked(), "one", _leg("0.2", 1, risk=small, margin=PX / 5), PORTFOLIO, T0).decision
    assert alone.outcome == "trimmed" and alone.limit_hit == "net_instrument"  # 12,000 net > 0.5 x 20,000
    a, b = check_pair(_marked(), "carry", _leg("0.2", 1, risk=small, margin=PX / 5),
                      _leg("0.2", -1, risk=small, margin=PX / 5), PORTFOLIO, T0)
    assert a.decision.outcome == b.decision.outcome == "approved"  # net 0; gross 24,000 = 1.2x, inside 1.5x
    a, b = check_pair(_marked(), "carry", _leg("0.3", 1, risk=small, margin=PX / 5),
                      _leg("0.3", -1, risk=small, margin=PX / 5), PORTFOLIO, T0)
    assert a.decision.limit_hit == "gross" and a.decision.approved_qty == b.decision.approved_qty == D("0.25")


def test_a_hedge_ratio_is_held_when_the_pair_is_trimmed():
    a, b = check_pair(_marked(), "pair", _leg("0.1", 1), _leg("0.2", -1, underlying="ETHBTC"), PORTFOLIO, T0)
    assert a.decision.outcome == "trimmed"
    assert b.decision.approved_qty == a.decision.approved_qty * 2


def test_a_fund_at_its_cap_rejects_the_pair_whole_and_reserves_nothing():
    ledger = _marked()
    full = check_order(ledger, "other", _leg("0.166", 1), PORTFOLIO, T0).decision
    assert full.outcome == "approved"  # 996 of the 1,000 open-risk cap
    before = list(ledger.reservations())
    a, b = check_pair(ledger, "carry", _leg("0.05", 1), _leg("0.05", -1), PORTFOLIO, T0)
    assert a.decision.outcome == b.decision.outcome == "rejected"
    assert a.decision.approved_qty == b.decision.approved_qty == 0
    assert (a.reservation, b.reservation) == (None, None) and ledger.reservations() == before


def test_either_leg_below_its_minimum_rejects_both():
    a, b = check_pair(_marked(), "carry", _leg("0.05", 1), _leg("0.05", -1, min_qty=D("0.1")), PORTFOLIO, T0)
    assert a.decision.outcome == b.decision.outcome == "rejected" and a.decision.limit_hit == "below_min"
    assert a.reservation is None and b.reservation is None


def test_a_blocked_portfolio_rejects_both_legs():
    ledger = _marked()
    later = T0 + timedelta(minutes=5)  # the fund's mark is stale
    a, b = check_pair(ledger, "carry", _leg("0.01", 1), _leg("0.01", -1), PORTFOLIO, later)
    assert a.decision.outcome == b.decision.outcome == "rejected" and not ledger.reservations()


def test_a_check_that_fails_rejects_both_legs_and_alerts():
    class Broken(MemoryLedger):
        def lock(self):
            raise RuntimeError("no lock")

    ledger = Broken(equity=D("20000"))
    a, b = check_pair(ledger, "carry", _leg("0.01", 1), _leg("0.01", -1), PORTFOLIO, T0)
    assert a.decision.outcome == b.decision.outcome == "rejected"
    assert [r.decision.limit_hit for r in ledger.records] == ["lock_error", "lock_error"]
    assert ledger.alerts and ledger.alerts[0][0] == "portfolio_check_failed"


def test_unusable_figures_fail_closed():
    a, b = decide_pair(Book(D("20000")), _leg("0.1", 1), _leg("0.1", -1, risk=D(0)), PORTFOLIO)
    assert a.outcome == b.outcome == "rejected" and a.approved_qty == b.approved_qty == 0
