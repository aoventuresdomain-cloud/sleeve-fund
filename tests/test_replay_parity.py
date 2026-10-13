"""P2-7: paper replay parity. Paper's journaled gate decisions against the portfolio backtest's over the same bars,
each mismatch class flagged, and the fill-cost line the ACT-DRIFT trigger reads (above 2.5 bp over 200+ fills)."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pandas as pd
import pytest
from test_portfolio_run import GROSS_60, _spec

from sleeve_fund.gate_ledger import Journaled
from sleeve_fund.portfolio.limits import Decision
from sleeve_fund.research import portfolio
from sleeve_fund.research.portfolio_run import run_portfolio
from sleeve_fund.research.replay_parity import (
    DRIFT_MIN_FILLS,
    Decided,
    adverse_bp,
    compare,
    drift,
    from_backtest,
    from_journal,
    uncorrected,
)

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _j(seq, strategy, bar, outcome="approved", req="1", appr="1", limit=None, stage="submit", side=1, price="100"):
    return Journaled(seq, strategy, f"{strategy}-{seq}", "entry", bar, stage, bar + timedelta(seconds=2), "BTC",
                     D(price), Decision(outcome, D(appr), D(req), limit, ""), 1)


def _b(strategy, bar, outcome="approved", req="1", appr="1", limit=None):
    return {"strategy": strategy, "ts": bar + timedelta(microseconds=1), "bar_ts": bar, "outcome": outcome,
            "approved": D(appr), "requested": D(req), "limit": limit, "reason": "", "side": 1, "price": D("100")}


def _day(n):
    return T0 + timedelta(days=n)


@pytest.fixture(scope="module")
def run():
    return run_portfolio([_spec("a", 2), _spec("b", 10, {"market": "perp", "allow_short": True})], GROSS_60)


def test_the_backtests_decision_log_carries_the_deciding_bars_close(run):
    assert run.decisions
    closes = {name: set(leg.prices.index) for name, leg in zip(("a", "b"), (_spec("a", 2), _spec("b", 10)))}
    for d in run.decisions:
        assert d["bar_ts"] == d["ts"] - timedelta(microseconds=1)
        assert pd.Timestamp(d["bar_ts"]) in closes[d["strategy"]]
        assert d["side"] in (1, -1) and d["price"] > 0


def test_a_journal_identical_to_the_backtest_gives_no_mismatches(run):
    assert {d["outcome"] for d in run.decisions} >= {"approved", "trimmed"}  # the tight gross limit bit somewhere
    journal = [Journaled(i, d["strategy"], f"i{i}", "entry", d["bar_ts"], "submit", d["ts"], "BTC", d["price"],
                         Decision(d["outcome"], d["approved"], d["requested"], d["limit"], ""), 1)
               for i, d in enumerate(run.decisions)]
    report = compare(from_journal(journal), from_backtest(run.decisions))
    assert report.clean and report.matched == report.paper == report.backtest == len(run.decisions)


def test_each_mismatch_class_is_flagged():
    paper = from_journal([
        _j(1, "a", _day(0)),                                              # matches
        _j(2, "a", _day(1)),                                              # paper only
        _j(3, "b", _day(1), outcome="trimmed", appr="0.5", limit="gross"),  # outcome, limit, approved
        _j(4, "b", _day(2), appr="0.4", outcome="trimmed", limit="gross"),  # approved qty only
        _j(5, "a", _day(3), req="2", appr="2"),                           # requested (sizing) and approved
        _j(6, "a", _day(4), outcome="rejected", appr="0", limit="margin"),  # limit only
    ])
    bt = from_backtest([
        _b("a", _day(0)),
        _b("b", _day(1)),
        _b("b", _day(2), appr="0.5", outcome="trimmed", limit="gross"),
        _b("a", _day(3)),
        _b("a", _day(4), outcome="rejected", appr="0", limit="gross"),
        _b("b", _day(5)),                                                 # backtest only
    ])
    report = compare(paper, bt)
    got = {(m.strategy, m.bar_ts): m.what for m in report.mismatches}
    assert got == {("a", _day(1)): ("paper only",),
                   ("b", _day(1)): ("outcome", "limit", "approved"),
                   ("b", _day(2)): ("approved",),
                   ("a", _day(3)): ("requested", "approved"),
                   ("a", _day(4)): ("limit",),
                   ("b", _day(5)): ("backtest only",)}
    assert report.matched == 1 and (report.paper, report.backtest) == (6, 6)
    assert report.counts() == {"paper only": 1, "backtest only": 1, "outcome": 1, "limit": 2, "approved": 3,
                               "requested": 1}


def test_two_decisions_on_one_bar_pair_in_order_and_an_extra_is_one_sided():
    paper = from_journal([_j(1, "a", _day(0)), _j(2, "a", _day(0), req="3", appr="3")])
    bt = from_backtest([_b("a", _day(0))])
    report = compare(paper, bt)
    assert report.matched == 1 and [m.what for m in report.mismatches] == [("paper only",)]
    assert report.mismatches[0].paper.requested == D("3")


def test_fill_stage_backstop_rows_are_not_compared_and_the_window_bounds_both_sides():
    paper = from_journal([_j(1, "a", _day(0), stage="fill"), _j(2, "a", _day(1)), _j(3, "a", _day(9))])
    bt = from_backtest([_b("a", _day(-1)), _b("a", _day(1))])
    assert len(paper) == 2
    report = compare(paper, bt, start=_day(0), end=_day(5))
    assert report.clean and report.matched == 1 and (report.paper, report.backtest) == (1, 1)


def test_adverse_drift_matches_a_hand_calculation():
    # buy expected 100, filled 100.03: 3 bp against. Sell expected 200, filled 199.98: 1 bp against.
    # Buy expected 50, filled 49.995: 1 bp in favour (-1). Mean (3 + 1 - 1) / 3 = 1 bp.
    assert adverse_bp(1, D("100"), D("100.03")) == pytest.approx(3.0)
    assert adverse_bp(-1, D("200"), D("199.98")) == pytest.approx(1.0)
    assert adverse_bp(1, D("50"), D("49.995")) == pytest.approx(-1.0)
    line = drift([(1, D("100"), D("100.03")), (-1, D("200"), D("199.98")), (1, D("50"), D("49.995"))])
    assert line.n == 3 and line.mean_bp == pytest.approx(1.0) and not line.act
    assert drift([]) == drift([]) and drift([]).n == 0 and drift([]).mean_bp is None


def test_the_act_drift_trigger_needs_both_the_mean_above_two_and_a_half_bp_and_two_hundred_fills():
    over = (1, D("10000"), D("10002.6"))  # 2.6 bp against
    at = (1, D("10000"), D("10002.5"))  # exactly 2.5 bp: not above
    assert drift([over] * DRIFT_MIN_FILLS).act
    assert not drift([over] * (DRIFT_MIN_FILLS - 1)).act
    assert not drift([at] * DRIFT_MIN_FILLS).act


def test_drift_is_measured_against_the_expected_fill_without_any_act_drift_in_force(monkeypatch):
    monkeypatch.setattr(portfolio, "ACT_DRIFT_BP", 3.0)
    close = D("10000")
    assert uncorrected(close, 1, 0.0005) == portfolio.fill_price(close, 1, 0.0005, drift_bp=0.0)
    assert uncorrected(close, 1, 0.0005) < portfolio.fill_price(close, 1, 0.0005)


@pytest.mark.parametrize("side,expected", [(0, D("1")), (1, D("0"))])
def test_a_side_or_expected_fill_that_cant_be_measured_is_refused(side, expected):
    with pytest.raises(ValueError):
        adverse_bp(side, expected, D("1"))


def test_decided_compares_by_value():
    assert Decided("a", T0, "approved", D("1"), D("1"), None) == Decided("a", T0, "approved", D("1.0"), D("1"), None)
